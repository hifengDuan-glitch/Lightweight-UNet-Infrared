import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 基础注意力模块 (原版 - 完整保留)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ==========================================
# 创新点 1: 坐标感知注意力 (CoordAtt & 双模态开关)
# ==========================================
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=16, use_inno1=False):
        super(CoordAtt, self).__init__()
        self.use_inno1 = use_inno1

        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))

        if self.use_inno1:
            self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
            self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        if self.use_inno1:
            # 开启创新点1：双模态相加 (AvgPool + MaxPool)
            x_h = self.pool_h_avg(x) + self.pool_h_max(x)
            x_w = self.pool_w_avg(x).permute(0, 1, 3, 2) + self.pool_w_max(x).permute(0, 1, 3, 2)
        else:
            # 单模态普通 CoordAtt
            x_h = self.pool_h_avg(x)
            x_w = self.pool_w_avg(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h


# ==========================================
# 核心解耦: 可配置特征提取块 (消除随机种子偏移版)
# ==========================================
class ConfigurableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_inno1=False, use_inno2=False):
        super(ConfigurableBlock, self).__init__()
        self.use_inno1 = use_inno1
        self.use_inno2 = use_inno2

        if self.use_inno2:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=2, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.act = nn.SiLU(inplace=True)  # 严格对齐原版命名，统一用一个实例

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

        # -------------------------------------------------------------
        # 【关键修正】: 必须先声明注意力模块，再声明 shortcut！
        # 否则 self.modules() 遍历顺序会变，导致 kaiming_normal_ 给错权重，引发性能差异！
        # -------------------------------------------------------------
        if self.use_inno1:
            self.coord_att = CoordAtt(out_channels, reduction=16, use_inno1=True)  # 名字对齐原版
        else:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False if self.use_inno2 else True),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_inno2:
            out = self.act(out)
            out = self.conv3(out)
            out = self.bn3(out)

        # 执行注意力调制
        if self.use_inno1:
            out = self.coord_att(out)
        else:
            out = self.ca(out) * out
            out = self.sa(out) * out

        out += residual

        out = self.act(out)
        return out


# ==========================================
# 最终解耦: MSHNet (严格对齐初始化分布)
# ==========================================
class MSHNet(nn.Module):
    def __init__(self, input_channels, use_inno1=False, use_inno2=False, use_inno3=False):
        super().__init__()
        self.use_inno1 = use_inno1
        self.use_inno2 = use_inno2
        self.use_inno3 = use_inno3

        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        in_ch = input_channels + 2 if self.use_inno3 else input_channels
        self.conv_init = nn.Conv2d(in_ch, param_channels[0], 1, 1)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block_num=1)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block_num=param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block_num=param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block_num=param_blocks[2])

        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block_num=param_blocks[3])

        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block_num=param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block_num=param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block_num=param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block_num=1)

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        if self.use_inno3:
            self.fusion_dw = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=False)
            self.fusion_pw = nn.Conv2d(4, 1, kernel_size=1, bias=True)
        else:
            self.final = nn.Conv2d(4, 1, 3, 1, 1)

        # 执行严格的防爆初始化
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, block_num=1):
        layer = []
        layer.append(ConfigurableBlock(in_channels, out_channels, use_inno1=self.use_inno1, use_inno2=self.use_inno2))
        for _ in range(block_num - 1):
            layer.append(ConfigurableBlock(out_channels, out_channels, use_inno1=self.use_inno1, use_inno2=self.use_inno2))
        return nn.Sequential(*layer)

    def _initialize_weights(self):
        # 1. 主干网络：标准正态分布初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 2. 坐标系偏置清零 (仅在开启创新点 3 时执行)
        if self.use_inno3:
            nn.init.constant_(self.conv_init.weight[:, -2:, :, :], 0)

        # ---------------------------------------------------------
        # 3. 先验概率初始化 (Prior Probability Initialization)
        # ---------------------------------------------------------
        prior_prob = 0.01
        bias_value = -4.59  # -log((1 - 0.01) / 0.01)

        out_layers = [self.output_0, self.output_1, self.output_2, self.output_3]
        if self.use_inno3:
            out_layers.append(self.fusion_pw)
        else:
            out_layers.append(self.final)

        for out_layer in out_layers:
            nn.init.normal_(out_layer.weight, std=0.01)
            if out_layer.bias is not None:
                nn.init.constant_(out_layer.bias, bias_value)

    def forward(self, x, warm_flag=False):
        if self.use_inno3:
            B, C, H, W = x.shape
            # 原汁原味的零均值化
            y_index = (torch.arange(0, H, 1, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W) / H) - 0.5
            x_index = (torch.arange(0, W, 1, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W) / W) - 0.5
            model_input = torch.cat([x, x_index, y_index], dim=1)
        else:
            model_input = x

        x_e0 = self.encoder_0(self.conv_init(model_input))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)

            aligned_masks = torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1)

            if self.use_inno3:
                fused = self.fusion_dw(aligned_masks)
                output = self.fusion_pw(fused)
            else:
                output = self.final(aligned_masks)

            return [mask0, mask1, mask2, mask3], output

        else:
            output = self.output_0(x_d0)
            return [], output