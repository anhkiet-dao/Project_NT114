import numpy as np
import os
from torchvision import datasets, transforms

# ==================== CÀI ĐẶT ====================
NUM_CLIENTS    = 5
ALPHA          = 0.5   # Càng nhỏ → nhãn càng lệch (0.1 = rất lệch, 1.0 = gần IID)
QUANTITY_ALPHA = 0.5   # Càng nhỏ → số mẫu mỗi client càng chênh lệch
MIN_SAMPLES    = 100   # Số mẫu tối thiểu mỗi client
SAVE_DIR       = './data/non_iid'
# =================================================

def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train, test

def split_non_iid(dataset, num_clients, alpha=0.5, quantity_alpha=0.5, min_samples=100):
    """
    Non-IID thực sự = Label Skew (Dirichlet) + Quantity Skew.
    
    - Label Skew:    mỗi client có phân phối nhãn khác nhau (alpha nhỏ → lệch mạnh)
    - Quantity Skew: số mẫu mỗi client khác nhau (quantity_alpha nhỏ → chênh lệch nhiều)
    """
    labels      = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    n_total     = len(labels)

    # ---- Bước 1: Quantity Skew - quyết định mỗi client nhận bao nhiêu mẫu ----
    qty_props = np.random.dirichlet(np.repeat(quantity_alpha, num_clients))
    qty_props = np.maximum(qty_props, min_samples / n_total)   # đảm bảo min_samples
    qty_props = qty_props / qty_props.sum()                    # chuẩn hóa lại
    client_sizes = (qty_props * n_total).astype(int)
    client_sizes[-1] = n_total - client_sizes[:-1].sum()       # bù phần dư

    print(f"\n  Số mẫu mỗi client: {client_sizes.tolist()}")

    # ---- Bước 2: Label Skew - phân phối nhãn theo Dirichlet ----
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)

        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))

        cuts = (proportions * len(cls_idx)).astype(int)
        cuts[-1] = len(cls_idx) - cuts[:-1].sum()

        start = 0
        for i, cut in enumerate(cuts):
            client_indices[i].extend(cls_idx[start:start + cut].tolist())
            start += cut

    # ---- Bước 3: Cắt đúng số lượng mẫu theo quantity skew ----
    final_indices = []
    for i in range(num_clients):
        idx = np.array(client_indices[i])
        np.random.shuffle(idx)

        size = min(client_sizes[i], len(idx))   # không cắt quá số có sẵn
        final_indices.append(idx[:size].tolist())

    return final_indices

def save_splits(splits, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, idx in enumerate(splits):
        path = os.path.join(save_dir, f'client_{i+1}.npy')
        np.save(path, np.array(idx))
        print(f"  Đã lưu: {path}  ({len(idx)} mẫu)")

def print_stats(splits, labels, alpha, quantity_alpha):
    print(f"\n=== Non-IID (label_alpha={alpha}, qty_alpha={quantity_alpha}) "
          f"- {len(splits)} Clients ===")
    print(f"{'Client':<10} {'Tổng':>6}  {'Phân phối nhãn (0-9)'}")
    print("-" * 70)
    for i, idx in enumerate(splits):
        counts = np.bincount(labels[idx], minlength=10)
        bar = "  ".join(f"{c:4d}" for c in counts)
        print(f"Client {i+1:<4} {len(idx):>6}  {bar}")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("Đang tải MNIST...")
    train, test = get_mnist()
    print(f"Train: {len(train)} | Test: {len(test)}")

    splits = split_non_iid(
        train,
        NUM_CLIENTS,
        alpha=ALPHA,
        quantity_alpha=QUANTITY_ALPHA,
        min_samples=MIN_SAMPLES
    )

    labels = np.array(train.targets)
    print_stats(splits, labels, ALPHA, QUANTITY_ALPHA)

    print(f"\nĐang lưu...")
    save_splits(splits, SAVE_DIR)
    print("Hoàn thành!")