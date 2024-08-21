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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    
    for epoch in range(epochs):
        for batch in dataloader:
            pixel_values, _ = batch
            pixel_values = pixel_values.to(device)
            
            input_ids = model.tokenizer("A simple shape", return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            
            optimizer.zero_grad()

            noise_pred, noise = model(input_ids, pixel_values)

            # 打印输出的形状
            print(f"noise_pred.shape: {noise_pred.shape}")
            print(f"noise.shape: {noise.shape}")
            
            # 计算损失
            loss = compute_loss(noise_pred, noise).to(device)
            print(f"loss: {loss.item()}")

            if torch.isnan(loss) or torch.isinf(loss):
                print("Encountered NaN or Inf in loss, skipping backward pass.")
                continue

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")