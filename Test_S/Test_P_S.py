"""
@author: zjf
@create time: 2024/11/21 21:16
@desc: 测试模型的参数量and计算时长
"""
import torch
import os
from Model.Network import FFS_T
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# zjf 测试模型
# flying3D
# images = torch.randn(size=(1, 3, 15, 960, 544)).cuda()
# Focus_Dists = torch.randn(size=(1, 15, 960, 544)).cuda()
# 4D 光场
images = torch.randn(size=(1, 3, 34, 512, 512)).cuda()
Focus_Dists = torch.randn(size=(1, 34, 512, 512)).cuda()

# Microscopic
# images = torch.randn(size=(1, 3, 30, 608, 800)).cuda()
# Focus_Dists = torch.randn(size=(1, 30, 608, 800)).cuda()


# smart phone
# images = torch.randn(size=(1, 3, 10, 352, 256)).cuda()
# Focus_Dists = torch.randn(size=(1, 10, 352, 256)).cuda()

# images = torch.randn(size=(1, 3, 34, 512, 512)).cuda()
# Focus_Dists = torch.randn(size=(1, 34, 512, 512)).cuda()

# 初始化模型
model = FFS_T().cuda()
model.eval()  # 设置为评估模式（关闭dropout等）
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} 百万")

# 确保输入数据在GPU上
images = images.cuda()
Focus_Dists = Focus_Dists.cuda()

# 预热运行（不计算在时间内）
with torch.no_grad():
    for _ in range(3):
        _ = model(images, Focus_Dists)
    torch.cuda.synchronize()  # 确保所有GPU操作完成

# 进行100次测试
total_time = 0.0
with torch.no_grad():
    for i in range(100):
        # 同步GPU确保准确计时
        torch.cuda.synchronize()
        start_time = time.time()

        depth = model(images, Focus_Dists)

        torch.cuda.synchronize()  # 再次同步
        end_time = time.time()

        # 计算并累加单次时间
        iter_time = end_time - start_time
        total_time += iter_time

        # 每10次显示一次进度
        if (i + 1) % 10 == 0:
            print(f"完成 {i + 1}/100 次，最近耗时: {iter_time:.4f}s")

# 计算平均时间
avg_time = total_time / 100
print(f"\n平均计算时长: {avg_time:.6f} 秒")
print(f"总计算时长: {total_time:.4f} 秒")
print(f"FPS: {1 / avg_time:.2f}")  # 每秒处理的帧数/样本数

# 显存使用报告
print(f"峰值显存使用: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"当前显存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")