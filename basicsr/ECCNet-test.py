import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from basicsr.models.archs.ECCNet_arch import ECCNet
import torch.nn.functional as F
from basicsr.utils import (get_root_logger, get_time_str, make_exp_dirs)

MODEL_WEIGHTS_PATH = ''
TEST_DATA_DIR = ''
NUM_CLASSES =  5
IMG_CHANNEL = 1
INPUT_SIZE =  [220, 220]
BATCH_SIZE = 16
NUM_WORKERS = 8

def init_test_loggers(opt_name="TestFromFolder"):
    experiments_root = f'./experiments/{opt_name}'
    opt = {
        'name': opt_name,
        'is_train': False,
        'path': {
            'experiments_root': experiments_root,
            'results_root': os.path.join(experiments_root, 'results'),
            'log': os.path.join(experiments_root, 'log'),
            'models': os.path.join(experiments_root, 'models'),
        }
    }
    make_exp_dirs(opt)
    log_file = os.path.join(opt['path']['log'],
                            f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(f"Testing model: {MODEL_WEIGHTS_PATH}")
    logger.info(f"Test data dir: {TEST_DATA_DIR}")
    logger.info(f"Number of classes: {NUM_CLASSES}")
    return logger


def create_test_dataloader(data_dir, input_size, batch_size, num_workers, img_channel):
    if img_channel == 1:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
    elif img_channel == 3:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unsupported img_channel: {img_channel}")

    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    logger = logging.getLogger('basicsr')
    logger.info(f'Found {len(test_dataset)} images in {data_dir}')
    logger.info(f'Classes found: {test_dataset.classes}')

    if len(test_dataset.classes) != NUM_CLASSES:
        logger.warning(f"Number of classes found in data ({len(test_dataset.classes)}) "
                       f"does not match NUM_CLASSES setting ({NUM_CLASSES}). "
                       f"Please check your data directory structure or NUM_CLASSES.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader, test_dataset.classes


def main():
    logger = init_test_loggers()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    test_loader, class_names = create_test_dataloader(
        TEST_DATA_DIR, INPUT_SIZE, BATCH_SIZE, NUM_WORKERS, IMG_CHANNEL
    )
    logger.info(f"Using classes: {class_names}")

    model_net = ECCNet(
        img_channel=IMG_CHANNEL,
        width=36,
        enc_blk_nums=[1, 1, 1, 2],
        num_classes=NUM_CLASSES,
        GCE_CONVS_nums=[2, 2, 2, 2]
    )

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        logger.error(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")
        raise FileNotFoundError(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")

    try:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location='cpu')
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
            logger.info("Loaded weights from 'params' key.")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.info("Loaded weights from 'state_dict' key.")
        else:
            state_dict = checkpoint
            logger.info("Loaded weights directly from file.")

        model_net.load_state_dict(state_dict, strict=True)
        logger.info(f"Successfully loaded model weights from {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_net = model_net.to(device)
    logger.info(f"Model moved to device: {device}")

    logger.info(f'Start testing on {len(test_loader.dataset)} images...')
    model_net.eval()

    total_correct = 0
    total_samples = 0

    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long, device='cpu')
    class_total = torch.zeros(NUM_CLASSES, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model_net(data)
            _, predicted = torch.max(outputs, 1)

            correct_or_not = (predicted == labels)

            total_correct += correct_or_not.sum().item()
            total_samples += labels.size(0)

            batch_class_total = torch.bincount(labels, minlength=NUM_CLASSES)
            class_total += batch_class_total.cpu()

            if correct_or_not.any():
                correct_labels = labels[correct_or_not]
                batch_class_correct = torch.bincount(correct_labels, minlength=NUM_CLASSES)
                class_correct += batch_class_correct.cpu()
    final_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    per_class_acc_str = "\nPer-class Accuracy:\n"
    for i in range(NUM_CLASSES):
        total = class_total[i].item()
        correct = class_correct[i].item()
        if total > 0:
            acc = 100.0 * correct / total
            per_class_acc_str += f"  {class_names[i]}: {acc:.2f}% ({correct}/{total})\n"
        else:
            per_class_acc_str += f"  {class_names[i]}: No samples found.\n"

    result_str = f'\nFinal Test Accuracy: {final_acc:.2f}% ({total_correct}/{total_samples})'
    logger.info(result_str)
    logger.info(per_class_acc_str)
    print(result_str)
    print(per_class_acc_str)

if __name__ == '__main__':
    main()