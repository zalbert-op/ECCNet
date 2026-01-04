# ------------------------------------------------------------------------
# Modified from CGNet (https://github.com/Ascend-Research/CascadedGaze)
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from basicsr.models.archs.arch_util import LayerNorm2d
except ImportError:
    class LayerNorm2d(nn.Module):
        def __init__(self, num_features, eps=1e-5, affine=True):
            super(LayerNorm2d, self).__init__()
            self.num_features = num_features
            self.eps = eps
            self.affine = affine
            if self.affine:
                self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
                self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

        def forward(self, x):
            mean = x.mean(dim=1, keepdim=True) 
            var = x.var(dim=1, keepdim=True, unbiased=False) 
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                return self.weight * x_norm + self.bias 
            else:
                return x_norm

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, stride=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size,
                                   stride=stride, padding=padding,
                                   groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class GlobalContextExtractor(nn.Module):
    def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[1, 1, 1], padding=1, bias=False):
        super().__init__()
        self.convs = nn.ModuleList([
            depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
            for kernel_size, stride in zip(kernel_sizes, strides)
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        context_features = []
        for conv in self.convs:
            out = F.gelu(conv(x))
            pooled = self.avgpool(out)
            context_features.append(pooled)
        return torch.cat(context_features, dim=1)


class CascadedGazeBlock(nn.Module):
    def __init__(self, c, GCE_Conv=2, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        self.c = c
        self.dw_channel = c * DW_Expand
        self.GCE_Conv = GCE_Conv
        self.conv1 = nn.Conv2d(c, self.dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, 3,
                               padding=1, groups=self.dw_channel, bias=True)
        self.GCE = GlobalContextExtractor(c,
                                          kernel_sizes=[3, 3, 5] if GCE_Conv == 3 else [3, 3],
                                          strides=[1, 1, 1])
        context_dim = c * (3 if GCE_Conv == 3 else 2)
        self.context_fusion = nn.Sequential(
            nn.Conv2d(context_dim, c, 1, bias=True),
            nn.BatchNorm2d(c),
            nn.ReLU()
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel, self.dw_channel // 4, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.dw_channel // 4, self.dw_channel, 1, bias=True),
            nn.Sigmoid()
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x) 

        context = self.GCE(identity) 
        context = self.context_fusion(context) 

        attention = self.sca(x) 
        x = x * attention 
        x = self.dropout1(x)

        x = self.sg(x) 

        x = identity + x * self.beta + context 
        y = x 
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma 

class ECCNet(nn.Module):
    def __init__(self, img_channel=1, width=24, enc_blk_nums=[2, 2, 4, 6],
                 num_classes=2, GCE_CONVS_nums=[2, 2, 2, 2], drop_rate=0.2):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(img_channel, width, 3, padding=1, bias=True),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Dropout2d(drop_rate / 4)
        )
 
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.feature_channels = []
        current_channels = width
        for i, num_blocks in enumerate(enc_blk_nums):
        
            blocks = []
            for _ in range(num_blocks):
                blocks.append(
                    CascadedGazeBlock(
                        current_channels,
                        GCE_Conv=GCE_CONVS_nums[i]
                    )
                )
            self.encoders.append(nn.Sequential(*blocks))
       
            self.feature_channels.append(current_channels)
            
            if i < len(enc_blk_nums) - 1:
                down_sample = nn.Sequential(
                    nn.MaxPool2d(2),  
                    nn.Conv2d(current_channels, current_channels * 2, 3, padding=1, bias=True),
                    nn.BatchNorm2d(current_channels * 2),
                    nn.ReLU(),
                    nn.Dropout2d(drop_rate)
                )
                self.downs.append(down_sample)
                current_channels *= 2
       
        self.feature_adapters = nn.ModuleList()
        for c in self.feature_channels:
            self.feature_adapters.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, 64, 1, bias=True), 
                nn.BatchNorm2d(64),
                nn.ReLU()
            ))
        total_features = 64 * len(self.feature_channels) 
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate / 2),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.intro(x) 
        feature_maps = []

        for i in range(len(self.encoders)): 
            x = self.encoders[i](x) 
            feature_maps.append(x)
            if i < len(self.downs): 
                x = self.downs[i](x) 
        fused_features = []
    
        for i, feat in enumerate(feature_maps):
            adapted = self.feature_adapters[i](feat)
            fused_features.append(adapted)

        fused = torch.cat(fused_features, dim=1) 
        fused = fused.view(fused.size(0), -1) 

        out = self.classifier(fused) 
        return out

if __name__ == '__main__':
    try:
        from ptflops import get_model_complexity_info
        flops_available = True
        print("ptflops is available.")
    except ImportError:
        flops_available = False
        print("ptflops not available. Skipping FLOPs calculation.")

    img_channel = 1
    width = 18
    enc_blks = [2, 2, 4, 6]
    GCE_CONVS_nums = [2, 2, 2, 2]
    num_classes = 5

    model = ECCNet(
        img_channel=img_channel,
        width=width,
        enc_blk_nums=enc_blks,
        num_classes=num_classes,
        GCE_CONVS_nums=GCE_CONVS_nums
    )
    print(f"Model created with num_classes={num_classes}")
    print(f"Expected feature channels for adapters: {model.feature_channels}")

    inp_shape = (img_channel, 220, 220)
    if flops_available:
        try:
            macs, params = get_model_complexity_info(
                model, inp_shape,
                as_strings=True, 
                print_per_layer_stat=False 
            )
            print(f"Model MACs: {macs}, Params: {params}")
        except Exception as e:
            print(f"Failed to compute FLOPs: {type(e).__name__} : {e}")
            import traceback
            macs, params = None, None
    else:
        macs, params = None, None

    test_input = torch.randn(2, *inp_shape)
    try:
        output = model(test_input)
        print(f"Output shape: {output.shape}")
        print(f"Output values (softmax, first sample): {torch.softmax(output, dim=1)[0]}")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Error during forward pass: {type(e).__name__} : {e}")
        import traceback
        traceback.print_exc()
