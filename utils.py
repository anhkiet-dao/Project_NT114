import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import numpy as np

def load_client_data(client_id, split_type="non_iid", batch_size=32):
    path = f"data_new/{split_type}/client_{client_id}.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        X = np.array(data['images'], dtype=np.float32)
        y = data['labels']
    else:
        raise ValueError(f"{path} cần dict với key 'images' và 'labels'")

    if X.ndim == 2: 
        X = X.reshape(-1, 1, 28, 28)
    elif X.ndim == 3:  
        X = X[:, None, :, :]

    X = torch.tensor(X, dtype=torch.float32)

    y_processed = []
    for label in y:
        if isinstance(label, str):
            if label.isdigit():
                y_processed.append(int(label))
            else:
                y_processed.append(ord(label.lower()) - ord('a'))
        else:
            y_processed.append(int(label))
    y = torch.tensor(y_processed, dtype=torch.long)

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader