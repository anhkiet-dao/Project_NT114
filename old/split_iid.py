# split_iid.py
import numpy as np
import os
from torchvision import datasets, transforms

# ==================== CÀI ĐẶT ====================
NUM_CLIENTS = 5
SAVE_DIR    = './splits/iid'
# =================================================

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train, test

def split_iid(dataset, num_clients):
    num_items = len(dataset) // num_clients
    indices   = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    client_data = []
    for i in range(num_clients):
        client_data.append(indices[i * num_items : (i+1) * num_items])
    return client_data

def save_splits(splits, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, idx in enumerate(splits):
        path = os.path.join(save_dir, f'client_{i+1}.npy')
        np.save(path, np.array(idx))
        print(f"  Đã lưu: {path}")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("Đang tải MNIST...")
    train, test = get_mnist()
    print(f"Train: {len(train)} | Test: {len(test)}")

    splits = split_iid(train, NUM_CLIENTS)
    labels = np.array(train.targets)

    print(f"\n=== IID - {NUM_CLIENTS} Clients ===")
    for i, idx in enumerate(splits):
        counts = np.bincount(labels[idx], minlength=10)
        print(f"Client {i+1} | {len(idx)} samples | Nhãn: {counts}")

    print(f"\nĐang lưu...")
    save_splits(splits, SAVE_DIR)
    print("Hoàn thành!")