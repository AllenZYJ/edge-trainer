from torch.optim import Adam
import torch
import torch.nn.functional as F

# 定义损失函数
def compute_loss(noise_pred, noise):
    return F.mse_loss(noise_pred, noise)

# 训练函数
def sd_trainer(model, dataloader, epochs, lr):
    optimizer = Adam(model.parameters(), lr=lr)
    model.train()

    device = torch.device("cuda:0")
    model.to(device)
    
    for epoch in range(epochs):
        for batch in dataloader:
            pixel_values, _ = batch
            pixel_values = pixel_values.to(device)
            
            # 创建一个 batch 的文本输入
            batch_size = pixel_values.shape[0]
            input_ids = model.tokenizer(["A simple shape"] * batch_size, return_tensors="pt", padding=True, truncation=True).input_ids
            input_ids = input_ids.to(device)

            optimizer.zero_grad()

            noise_pred, noise,_,__ = model(input_ids, pixel_values)

            # 打印输出的形状
            
            # 计算损失
            loss = compute_loss(noise_pred, noise).to(device)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Encountered NaN or Inf in loss, skipping backward pass.")
                continue

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")