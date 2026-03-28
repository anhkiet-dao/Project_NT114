import random
import numpy as np

# ===== CONFIG =====
TOTAL_CLIENTS = 5         
FAULTY_PER_ROUND = (1, 2)  

# ===== GLOBAL CACHE (QUAN TRỌNG) =====
FAULTY_MAP = {}


# ===== CHỌN FAULTY CLIENT =====
def get_faulty_clients(round_num):

    if round_num in FAULTY_MAP:
        return FAULTY_MAP[round_num]

    num_faulty = random.randint(*FAULTY_PER_ROUND)

    faulty_clients = random.sample(
        range(1, TOTAL_CLIENTS + 1),  # client_id bắt đầu từ 1
        num_faulty
    )

    FAULTY_MAP[round_num] = faulty_clients

    print(f"[Round {round_num}] Faulty clients: {faulty_clients}")

    return faulty_clients


# ===== CHECK CLIENT CÓ FAULTY KHÔNG =====
def is_faulty_client(client_id, round_num):

    faulty_clients = get_faulty_clients(round_num)

    return client_id in faulty_clients


# ===== CORRUPT PARAMETERS =====
def corrupt_parameters(params):

    corrupted = []

    for p in params:
        noise = np.random.normal(0, 1, p.shape)

        corrupted.append(p + noise)

    return corrupted