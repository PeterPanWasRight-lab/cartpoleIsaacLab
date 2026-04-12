import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备模型、优化器和数据 (默认在 GPU 上，类型为 FP32)
model = nn.Linear(3, 4).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

data = torch.randn(5, 3).cuda()
target = torch.randn(5, 4).cuda()
# 2. 初始化 GradScaler (关键点 1：防止 FP16 下的梯度下溢)
scaler = torch.amp.GradScaler(enabled=True)

epochs = 5
for epoch in range(epochs):
    optimizer.zero_grad()

    # 3. 开启 autocast 上下文 (关键点 2：前向传播自动选择最优精度)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        # 这里的矩阵乘法会自动转换为 FP16 进行加速
        output = model(data)   # 在autocast内部创建的f16变量，即使出了autocast上下文，也会保持为f16类型
        # Loss 计算等对精度敏感的操作会自动保持在 FP32
        loss = loss_fn(output, target)

    # autocast通常只用于前向传播，不应用于反向传播

    # 出with后，所有原本是FD32转成FD16的变量已经被转回FD32了，但是output等在autocast中生成的变量却被留在FD16了
    # 4. 反向传播：放大 loss，计算梯度
    # 因为 FP16 表示范围小，梯度容易变成 0 (下溢)。
    # scaler.scale 会将 loss 乘以一个缩放因子。
    scaler.scale(loss).backward()

    # 5. 梯度裁剪（必须在 unscale 之后进行）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 这时梯度已经被还原到正常范围，可以安全地进行裁剪了，裁剪针对象是模型参数的梯度，裁剪的方式是按照全局范数进行裁剪，max_norm 是裁剪的阈值，超过这个阈值的梯度会被缩放到 max_norm 的范围内。
    
    # 6. 更新参数
    # scaler.step 会先将梯度除以缩放因子还原，然后再执行 optimizer.step()
    scaler.step(optimizer)   # 如果上面已经调用过unscale_ 内部检测到已反缩放，跳过unscale  可以debug进入查看这个原理

    # 至此梯度的scale已经被还原，神将网络的现场已经恢复（包括float32的数值类型和梯度的scale已经调回）
    
    # 7. 更新缩放因子 (为下一次迭代准备)
    scaler.update()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    print(f"Current Scale: {scaler.get_scale():.1f}")
    print("output dtype:", output.dtype)  # 输出的类型是 torch.float16
    print("outputnew dtype:", model(data).dtype)  # 这里的输出也是 torch.float32，因为模型参数是 float32 的，autocast 只会在 with 块内自动转换数据类型，出了 with 块后，模型参数仍然是 float32，所以输出也是 float32 的。