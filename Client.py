# =========================================================
# IMPORT
# =========================================================

import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import os
import json
import hashlib
import time
import sys
import pickle
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split
)

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================================================
# CONFIG
# =========================================================

MU = 0.001
LR = 0.0005
EPOCHS = 2

TOTAL_CLIENTS = 1
LAMBDA = 1

CACHE = {}
LAST_FAULTY = set()

# =========================================================
# MODEL
# =========================================================

class CNN(nn.Module):

    def __init__(self, num_classes=62):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout2d(0.25)

        self.dropout2 = nn.Dropout(0.5)

        dummy = torch.zeros(1, 1, 28, 28)

        dummy = self._forward_conv(dummy)

        flatten_size = dummy.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, 128)

        self.fc2 = nn.Linear(128, num_classes)

    def _forward_conv(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        return x

    def forward(self, x):

        x = self._forward_conv(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.dropout2(x)

        x = self.fc2(x)

        return x
    
# =========================================================
# DATA LOADER
# =========================================================

def load_client_data(client_id, split_type="non_iid", batch_size=512):

    key = f"{split_type}_{client_id}"

    if key in CACHE:
        return CACHE[key]

    path = f"/kaggle/input/YOUR_DATASET/{split_type}/client_{client_id}.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["images"], dtype=np.float32)
    y = data["labels"]

    if X.ndim == 2:
        X = X.reshape(-1, 1, 28, 28)

    elif X.ndim == 3:
        X = X[:, None, :, :]

    X = torch.from_numpy(X).float()

    if isinstance(y[0], str):

        y_processed = []

        for label in y:

            if label.isdigit():
                y_processed.append(int(label))
            else:
                y_processed.append(
                    ord(label.lower()) - ord('a')
                )

        y = torch.tensor(y_processed, dtype=torch.long)

    else:
        y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    CACHE[key] = (trainloader, testloader)

    return trainloader, testloader

# =========================================================
# FAULTY CLIENT
# =========================================================

TOTAL_CLIENTS = 1

# 1. Khi đang HEALTHY:
P_HEALTHY_TO_HEALTHY = 0.85  
P_HEALTHY_TO_FAULTY  = 0.15  

# 2. Khi đang FAULTY:
P_FAULTY_TO_HEALTHY  = 0.80  
P_FAULTY_TO_FAULTY   = 0.20  

current_state = "HEALTHY" 

def get_faulty_clients(round_num):
    global current_state
    
    random_value = np.random.rand()
    
    if current_state == "HEALTHY":
        if random_value < P_HEALTHY_TO_FAULTY:
            current_state = "FAULTY"
            faulty_clients = [1]
            print(f"[Round {round_num}] SỰ CỐ: Hệ thống chuyển từ Chạy tốt -> Gặp lỗi!")
        else:
            current_state = "HEALTHY"
            faulty_clients = []
            print(f"[Round {round_num}] ỔN ĐỊNH: Hệ thống tiếp tục Chạy tốt.")
            
    elif current_state == "FAULTY":
        if random_value < P_FAULTY_TO_HEALTHY:
            current_state = "HEALTHY"
            faulty_clients = []
            print(f"[Round {round_num}] PHỤC HỒI: Hệ thống đã sửa xong lỗi -> Quay lại Chạy tốt.")
        else:
            current_state = "FAULTY"
            faulty_clients = [1]
            print(f"[Round {round_num}] CẢNH BÁO: Lỗi chưa khắc phục xong, tiếp tục dính lỗi!")
            
    return faulty_clients

def corrupt_parameters(params, round_num):
    corrupted = []
    # Độ nhiễu nhỏ và giảm dần theo số round để bảo vệ mô hình 1 client
    decay_factor = max(0.1, 1 - (round_num / 100)) # Giả định bạn chạy tối đa 100 round
    current_std = 0.01 * decay_factor

    for p in params:
        # p đang là numpy array từ self.get_parameters()
        p_corrupted = p.copy() 
        
        # Chỉ làm lỗi ngẫu nhiên 10% số lượng trọng số (tránh hỏng 100% mô hình)
        mask = np.random.rand(*p.shape) < 0.10
        noise = np.random.normal(0, current_std, p.shape)
        
        p_corrupted[mask] += noise[mask]
        corrupted.append(p_corrupted)
        
    return corrupted

# =========================================================
# TRAIN
# =========================================================

def train(model, trainloader, global_params, criterion):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for _ in range(EPOCHS):

        for batch_idx, (data, target) in enumerate(trainloader):

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            if batch_idx % 5 == 0:

                prox_term = 0.0

                for w, w_t in zip(
                    model.parameters(),
                    global_params
                ):
                    prox_term += torch.sum((w - w_t) ** 2)

            else:
                prox_term = 0.0

            loss = loss + (MU / 2) * prox_term

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            pred = torch.argmax(output, dim=1)

            correct += (pred == target).sum().item()

            total += target.size(0)

    avg_loss = total_loss / (len(trainloader) * EPOCHS)

    return {
        "loss": avg_loss,
        "accuracy": correct / total,
        "time": time.time() - start_time
    }

# =========================================================
# EVALUATE
# =========================================================

def evaluate(model, testloader, criterion):

    model.eval()

    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():

        for data, target in testloader:

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)

            loss += criterion(output, target).item()

            pred = torch.argmax(output, dim=1)

            correct += (pred == target).sum().item()

            total += target.size(0)

    return {
        "loss": loss / len(testloader),
        "accuracy": correct / total
    }

# =========================================================
# DUMMY IPFS + ZKP + BLOCKCHAIN
# =========================================================

def upload_to_ipfs(path):

    fake_cid = hashlib.md5(path.encode()).hexdigest()

    return fake_cid

def generate_proof(params):

    return {
        "proof_hash": hashlib.sha256(
            str(params[0].shape).encode()
        ).hexdigest()
    }

def submit_update(
    round_num,
    client_id,
    cid,
    proof_hash,
    acc
):

    tx_hash = hashlib.md5(
        f"{round_num}_{client_id}".encode()
    ).hexdigest()

    return tx_hash

# =========================================================
# FLOWER CLIENT
# =========================================================

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id, split_type):

        self.client_id = client_id

        self.model = CNN(num_classes=62).to(DEVICE)

        self.trainloader, self.testloader = load_client_data(
            client_id,
            split_type
        )

        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):

        return [
            v.detach().cpu().numpy()
            for v in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):

        state_dict = {
            k: torch.tensor(v).to(DEVICE)
            for k, v in zip(
                self.model.state_dict().keys(),
                parameters
            )
        }

        self.model.load_state_dict(
            state_dict,
            strict=False
        )

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        round_num = config.get("server_round", 1)

        faulty_clients = config.get(
            "faulty_clients",
            []
        )

        is_faulty = self.client_id in faulty_clients

        if is_faulty:
            params = corrupt_parameters(params, round_num)
            print(f"💣 [Round {round_num}] Sent corrupted update (Markov State: FAULTY)")
        else:
            print(f"✅ [Round {round_num}] Sent clean update (Markov State: HEALTHY)")

        global_params = [
            torch.from_numpy(p).to(DEVICE)
            for p in parameters
        ]

        result = train(
            self.model,
            self.trainloader,
            global_params,
            self.criterion
        )

        print(
            f"[Client {self.client_id}] "
            f"Acc: {result['accuracy']:.4f}"
        )

        params = self.get_parameters({})

        if is_faulty:
            params = corrupt_parameters(params)
            print("💣 Sent corrupted update")

        cid = upload_to_ipfs("model.pth")

        proof = generate_proof(params)

        proof_str = json.dumps(proof)

        proof_hash = hashlib.sha256(
            (proof_str + cid).encode()
        ).hexdigest()

        tx_hash = submit_update(
            round_num,
            self.client_id,
            cid,
            proof_hash,
            result["accuracy"]
        )

        print("TX:", tx_hash)

        metrics = {
            "client_id": self.client_id,
            "train_time": result["time"],
            "local_accuracy": result["accuracy"],
            "local_loss": result["loss"],
            "cid": cid,
            "proof": proof_str,
        }

        return params, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        result = evaluate(
            self.model,
            self.testloader,
            self.criterion
        )

        print(
            f"[Client {self.client_id}] "
            f"Test Acc: {result['accuracy']:.4f}"
        )

        return (
            result["loss"],
            len(self.testloader.dataset),
            result
        )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage: python federated_client.py "
            "<client_id> <iid|non_iid>"
        )

        sys.exit(1)

    client_id = int(sys.argv[1])

    split_type = sys.argv[2]

    print(
        f"Starting Client {client_id} "
        f"with {split_type}"
    )

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(
            client_id,
            split_type
        ),
    )