import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import os

from basicsr.models.archs.ECCNet_arch import ECCNet
from basicsr.models.archs.SCNet_arch import SCNet


class AdaptiveDenoisePipeline:
    def __init__(self, classifier_path, denoiser_paths, device='cuda', use_fp16=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.class_map = {0: 'biology', 1: 'iron', 2: 'leaf', 3: 'lego', 4: 'word'}

        # 启用性能优化
        torch.backends.cudnn.benchmark = True
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')

        # 1. 加载分类模型
        print("Loading classifier...")
        self.classifier = self.load_classifier(classifier_path)
        self.classifier.to(self.device)
        if self.use_fp16:
            self.classifier = self.classifier.half()
            print("Classifier converted to FP16")
        self.classifier.eval()
        print("Classifier loaded.")

        self.denoisers = {}
        print("Loading denoisers...")
        for noise_type, path in denoiser_paths.items():
            print(f"  - Loading {noise_type} denoiser from {path}")
            model = self.load_denoiser(path, noise_type)
            model.to(self.device)
            if self.use_fp16:
                model = model.half()
            model.eval()
            self.denoisers[noise_type] = model
        print("All denoisers loaded.")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        self._warmup_models()

    def _warmup_models(self):
        print("Warming up models...")
        dummy_input = torch.randn(1, 1, 220, 220, device=self.device)
        if self.use_fp16:
            dummy_input = dummy_input.half()

        with torch.no_grad():
            _ = self.classifier(dummy_input)
            for denoiser in self.denoisers.values():
                _ = denoiser(dummy_input)
        print("Models warmed up.")

    def load_classifier(self, path):
        model = ECCNet(
            img_channel=1,
            width=18,
            enc_blk_nums=[1, 1, 1, 2],
            num_classes=5,
            GCE_CONVS_nums=[2, 2, 2, 2]
        )
        checkpoint = torch.load(path, map_location='cpu')
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
        return model

    def load_denoiser(self, path, noise_type):
        model = SCNet(
            img_channel=1,
            width=84,
            middle_blk_num=24,
            enc_blk_nums=[6, 8, 10, 12],
            dec_blk_nums=[4, 4, 6, 8],
            GCE_CONVS_nums=[4, 4, 4, 4]
        )
        checkpoint = torch.load(path, map_location='cpu')

        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        return model

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')
        image_tensor = self.transform(image).unsqueeze(0)
        if self.use_fp16:
            image_tensor = image_tensor.half()
        return image_tensor.to(self.device, non_blocking=True)

    def denoise_single(self, image_path):
        image_tensor = self.preprocess_image(image_path)

        with torch.no_grad():

            classifier_output = self.classifier(image_tensor)
            _, predicted_idx = torch.max(classifier_output, 1)
            noise_type = self.class_map[predicted_idx.item()]
            print(f"classify: {noise_type}")

            selected_denoiser = self.denoisers[noise_type]
            denoised_image_tensor = selected_denoiser(image_tensor)

        return denoised_image_tensor

    def denoise_batch(self, image_paths):
        if len(image_paths) == 0:
            return []

        batch_tensors = []
        for path in image_paths:
            tensor = self.preprocess_image(path)
            batch_tensors.append(tensor)

        batch_tensor = torch.cat(batch_tensors, dim=0)

        results = []
        with torch.no_grad():
            classifier_outputs = self.classifier(batch_tensor)
            _, predicted_indices = torch.max(classifier_outputs, 1)

            for i, predicted_idx in enumerate(predicted_indices):
                noise_type = self.class_map[predicted_idx.item()]
                selected_denoiser = self.denoisers[noise_type]

                single_tensor = batch_tensor[i:i + 1]
                denoised_tensor = selected_denoiser(single_tensor)
                results.append(denoised_tensor)

        return results

    def benchmark_fps(self, test_image_path, num_iterations=100):
        print(f"\nStart: {num_iterations}")

        _ = self.denoise_single(test_image_path)

        start_time = time.time()
        for i in range(num_iterations):
            _ = self.denoise_single(test_image_path)

        end_time = time.time()
        total_time = end_time - start_time
        fps = num_iterations / total_time

        print(f"Performance: {fps:.2f} FPS")
        print(f"Toral Times: {total_time:.2f}s")
        return fps


class OptimizedAdaptiveDenoisePipeline(AdaptiveDenoisePipeline):

    def __init__(self, classifier_path, denoiser_paths, device='cuda', use_fp16=True,
                 enable_tensor_cores=True):
        super().__init__(classifier_path, denoiser_paths, device, use_fp16)

        if enable_tensor_cores and self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def denoise_optimized(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if len(image_paths) == 1:
            return [self.denoise_single(image_paths[0])]

        batch_tensors = [self.preprocess_image(path) for path in image_paths]
        batch_tensor = torch.cat(batch_tensors, dim=0)

        results = [None] * len(image_paths)

        with torch.no_grad():
            classifier_outputs = self.classifier(batch_tensor)
            _, predicted_indices = torch.max(classifier_outputs, 1)

            class_groups = {}
            for i, (pred_idx, tensor) in enumerate(zip(predicted_indices, batch_tensors)):
                noise_type = self.class_map[pred_idx.item()]
                if noise_type not in class_groups:
                    class_groups[noise_type] = []
                class_groups[noise_type].append((i, tensor))

            for noise_type, items in class_groups.items():
                denoiser = self.denoisers[noise_type]
                indices = [item[0] for item in items]
                tensors = [item[1] for item in items]

                if len(tensors) > 1:

                    for idx, tensor in zip(indices, tensors):
                        results[idx] = denoiser(tensor)
                else:
                    for idx, tensor in zip(indices, tensors):
                        results[idx] = denoiser(tensor)

        return results


def test_performance():
    denoiser_paths = {
        'iron': '',
        'leaf': '',
        'lego': '',
        'word': '',
        'biology': ''
    }

    configs = [
        {"use_fp16": False, "name": "FP32"},
        {"use_fp16": True, "name": "FP16"},
    ]

    test_image = ''

    for config in configs:
        print(f"\n{'=' * 50}")
        print(f"test: {config['name']}")
        print(f"{'=' * 50}")

        try:
            pipeline = OptimizedAdaptiveDenoisePipeline(
                classifier_path='',
                denoiser_paths=denoiser_paths,
                use_fp16=config['use_fp16']
            )

            fps = pipeline.benchmark_fps(test_image, num_iterations=50)
            results = pipeline.denoise_optimized(test_image)

            if config['use_fp16']:
                from torchvision.utils import save_image
                output_path = ''
                if pipeline.use_fp16:
                    results[0] = results[0].float()
                save_image(results[0], output_path)
                print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Config {config['name']} Test error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_performance()

    denoiser_paths = {
        'iron': '',
        'leaf': '',
        'lego': '',
        'word': '',
        'biology': ''
    }

    pipeline = OptimizedAdaptiveDenoisePipeline(
        classifier_path='',
        denoiser_paths=denoiser_paths,
        use_fp16=True
    )

    input_image_path = ''
    print(f"\n ing: {input_image_path} ...")

    start_time = time.time()
    final_result_tensor = pipeline.denoise_single(input_image_path)
    end_time = time.time()

    processing_time = end_time - start_time
    print(f"FPS: {1.0 / processing_time:.2f}")

    from torchvision.utils import save_image

    output_image_path = ''
    if pipeline.use_fp16:
        final_result_tensor = final_result_tensor.float()
    save_image(final_result_tensor, output_image_path)

    print(f"\n Saved: {output_image_path}")