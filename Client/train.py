# ================== train.py ==================
import torch
import time

DEVICE = torch.device("cpu")   # 👉 bạn không có GPU nên fix luôn
MU = 0.001
LR = 0.001
EPOCHS = 2   # 👉 giảm để tăng tốc FL

def train(model, trainloader, global_params, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    # ==============================
    # 🔥 TỐI ƯU FEDPROX (flatten 1 lần)
    # ==============================
    global_vector = torch.cat([p.view(-1) for p in global_params])

    # ==============================
    # TRAINING LOOP
    # ==============================
    for _ in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(trainloader):

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            # ==============================
            # 🔥 FEDPROX (giảm tần suất + vector hóa)
            # ==============================
            if batch_idx % 10 == 0:
                local_vector = torch.cat([p.view(-1) for p in model.parameters()])
                prox_term = torch.norm(local_vector - global_vector) ** 2
                loss = loss + (MU / 2) * prox_term

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred = torch.argmax(output, dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    # ==============================
    # FIX LOSS (đúng theo epoch)
    # ==============================
    avg_loss = total_loss / (len(trainloader) * EPOCHS)

    return {
        "loss": avg_loss,
        "accuracy": correct / total,
        "time": time.time() - start_time
    }