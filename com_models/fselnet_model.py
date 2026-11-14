
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from .FSEL_modules import DRP_1, DRP_2, DRP_3, JDPM, ETB



'''
backbone: resnet50
'''


class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=128):
        super(Network, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=False, in_chans=3, features_only=True)

        self.dePixelShuffle = torch.nn.PixelShuffle(2)

        self.up = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )
        self.channel = channels
        self.ETB_5 = ETB(2048+channels, channels)
        self.ETB_4 = ETB(1024+channels, channels)
        self.ETB_3 = ETB(512+channels, channels)
        self.ETB_2 = ETB(256+channels, channels)

        self.JDPM = JDPM(2048, channels)

        self.DRP_1 = DRP_1(channels, channels)
        self.DRP_2 = DRP_2(channels, channels)
        self.DRP_3 = DRP_3(channels,channels)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats

        p1 = self.JDPM(x4)
        x5_4 = p1
        x5_4_1 = x5_4.expand(-1, self.channel, -1, -1)

        x4   = self.ETB_5(torch.cat((x4,x5_4_1),1))
        x4_up = self.up(self.dePixelShuffle(x4))

        x3   = self.ETB_4(torch.cat((x3,x4_up),1))
        x3_up = self.up(self.dePixelShuffle(x3))

        x2   = self.ETB_3(torch.cat((x2,x3_up),1))
        x2_up = self.up(self.dePixelShuffle(x2))

        x1   = self.ETB_2(torch.cat((x1,x2_up),1))

        x4 = self.DRP_1(x4,x5_4)
        x3 = self.DRP_1(x3,x4)
        x2 = self.DRP_2(x2,x3,x4)
        x1 = self.DRP_3(x1,x2,x3,x4)

        p0 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)

        return p0, f4, f3, f2, f1

###################################################################
# ################ C3Net 框架适配包装器 (Wrapper) ##################
###################################################################
class fselnet_Wrapper(nn.Module):
    """
    用于将 FSELNet 桥接到 C3Net 训练框架的包装器。
    ResNet needs input sizes divisible by 32; 320x320 can run directly.
    """
    def __init__(self, config=None):
        super(fselnet_Wrapper, self).__init__()
        # 实例化原始网络
        self.model = Network(channels=128)
        self.input_divisor = 32
        
        # 【修改点】：删除了 self.head_final 等多余的预测头，因为原网络直接输出 1 通道结果
        
    def forward(self, x):
        # 记录原始输入尺寸 (例如 320x320)
        orig_size = x.shape[2:]
        
        # ResNet 结构要求输入能被 32 完美整除，否则解码器 skip connection 会报错
        H, W = orig_size
        safe_h = ((H + self.input_divisor - 1) // self.input_divisor) * self.input_divisor
        safe_w = ((W + self.input_divisor - 1) // self.input_divisor) * self.input_divisor
        if (safe_h, safe_w) != orig_size:
            x_resized = F.interpolate(x, size=(safe_h, safe_w), mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # 获取原网络输出的 5 个层级的预测图 (已经是 1 通道了！)
        p0, f4, f3, f2, f1 = self.model(x_resized)
        
        # 直接使用 f1, f2, f3 作为 Logits，插值拉伸回原始尺寸
        final_mask = F.interpolate(f1, size=orig_size, mode='bilinear', align_corners=False)
        object_mask = F.interpolate(f2, size=orig_size, mode='bilinear', align_corners=False)
        edge_mask = F.interpolate(f3, size=orig_size, mode='bilinear', align_corners=False)
        
        # 组装成 C3Net 引擎期望的字典格式
        output_dict = {
            'final_mask': final_mask,
            'object_mask': object_mask,
            'edge_mask': edge_mask
        }
        
        return output_dict
        
