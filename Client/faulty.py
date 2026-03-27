# import random
# import numpy as np

# TOTAL_CLIENTS = 10
# FAULTY_PER_ROUND = (1, 2)

# def is_faulty_client(client_id, round_num):
#     random.seed(round_num)

#     num_faulty = random.randint(*FAULTY_PER_ROUND)
#     faulty_clients = random.sample(range(TOTAL_CLIENTS), num_faulty)

#     return client_id in faulty_clients


# def corrupt_parameters(params):
#     corrupted = []
#     for p in params:
#         noise = np.random.normal(0, 20, p.shape)
#         corrupted.append(p + noise)
#     return corrupted
import random
import numpy as np

# ===== CONFIG =====
TOTAL_CLIENTS = 5          # phải match server
FAULTY_PER_ROUND = (1, 2)  # mỗi round có 1-2 client bị lỗi

# ===== GLOBAL CACHE (QUAN TRỌNG) =====
# để đảm bảo tất cả client dùng cùng danh sách faulty
FAULTY_MAP = {}


# ===== CHỌN FAULTY CLIENT =====
def get_faulty_clients(round_num):

    # nếu đã có thì dùng lại (đảm bảo consistency)
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
        # noise vừa phải (tránh bị detect 100%)
        noise = np.random.normal(0, 1, p.shape)

        corrupted.append(p + noise)

    return corrupted