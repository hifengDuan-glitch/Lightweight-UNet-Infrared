import torch
import time

# 尝试导入 thop 库用于计算 FLOPs 和 Params
try:
    from thop import profile
    from thop import clever_format

    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# 导入你刚刚定义的模型
from model import MSHNet


def main():
    print("=" * 60)
    print(" 初始化轻量级红外小目标检测网络 (MSHNet) ")
    print("=" * 60)

    # 1. 实例化完整版提出的模型
    # 开启所有创新点标志位，模拟论文中最终的轻量化架构
    model = MSHNet(input_channels=1, use_inno1=True, use_inno2=True, use_inno3=True)
    model.eval()

    # 2. 生成虚拟输入数据 [Batch, Channel, Height, Width]
    # 典型红外小目标检测单通道输入
    dummy_input = torch.randn(1, 1, 256, 256)
    print(f"[数据] 输入张量维度: {dummy_input.shape}")

    # 3. 前向传播测试
    print("\n[运行] 正在进行前向传播测试 (Forward Pass)...")
    start_time = time.time()
    with torch.no_grad():
        # 注意: 你的 forward 函数支持 warm_flag，我们开启它以测试完整的多尺度输出路径
        masks, output = model(dummy_input, warm_flag=True)
    end_time = time.time()

    # 验证输出
    print(f"[结果] 最终输出掩码维度: {output.shape}")
    if masks:
        print(f"[结果] 多尺度辅助掩码数量: {len(masks)} (对应 Decoder 0-3)")
    print(f"[性能] 单张图像 (CPU) 推理耗时: {(end_time - start_time) * 1000:.2f} ms")

    # 4. 计算参数量和 FLOPs (自动对齐论文声明)
    if HAS_THOP:
        print("\n[统计] 正在计算模型参数量与 FLOPs...")
        # 注意: profile 需要的 inputs 是一个元组，必须与 forward 的参数对应 (x, warm_flag)
        macs, params = profile(model, inputs=(dummy_input, True), verbose=False)
        macs_str, params_str = clever_format([macs, params], "%.2f")
        print(f"       => 核心参数量 (Params): {params_str}")
        print(f"       => 计算量 (MACs/FLOPs): {macs_str}")
    else:
        print("\n[提示] 未检测到 'thop' 库。如需验证 FLOPs，请运行: pip install thop")

    print("\n✅ 测试成功！网络结构有效，前向传播无维度冲突。")
    print("=" * 60)


if __name__ == "__main__":
    main()