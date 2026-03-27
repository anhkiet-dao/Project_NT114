import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# CONFIG
# ==========================================
IID_FILE = "history/server_history_fedadam_iid.json"
NONIID_FILE = "history/server_history_fedadam_non_iid.json"

SAVE_DIR = "plots/comparison"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================
with open(IID_FILE, "r") as f:
    iid = json.load(f)

with open(NONIID_FILE, "r") as f:
    noniid = json.load(f)

rounds_iid = iid["global"]["round"]
rounds_noniid = noniid["global"]["round"]

# ==========================================
# STYLE
# ==========================================
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.size": 12,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold"
})

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def compute_avg_local_accuracy(data):
    clients = data["clients"]
    rounds = data["global"]["round"]

    avg_acc = []

    for i in range(len(rounds)):
        accs = []

        for cid in clients:
            accs.append(clients[cid]["test_accuracy"][i])

        avg_acc.append(np.mean(accs))

    return avg_acc


def avg_train_time(data):

    times = []

    for cid in data["clients"]:
        times.extend(data["clients"][cid]["train_time"])

    return np.mean(times)


# ======================================================
# 1️⃣ GLOBAL ACCURACY COMPARISON
# ======================================================
plt.figure(figsize=(8,5))

plt.plot(rounds_iid, iid["global"]["accuracy"],
         marker="o", linewidth=2, label="IID")

plt.plot(rounds_noniid, noniid["global"]["accuracy"],
         marker="s", linewidth=2, label="Non-IID")

plt.title("Global Accuracy: IID vs Non-IID")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.ylim(0,1)

plt.legend()
plt.tight_layout()

plt.savefig(f"{SAVE_DIR}/compare_global_accuracy.png", dpi=300)
plt.close()


# ======================================================
# 2️⃣ GLOBAL LOSS COMPARISON
# ======================================================
plt.figure(figsize=(8,5))

plt.plot(rounds_iid, iid["global"]["loss"],
         marker="o", linewidth=2, label="IID")

plt.plot(rounds_noniid, noniid["global"]["loss"],
         marker="s", linewidth=2, label="Non-IID")

plt.title("Global Loss: IID vs Non-IID")
plt.xlabel("Communication Round")
plt.ylabel("Loss")

plt.legend()
plt.tight_layout()

plt.savefig(f"{SAVE_DIR}/compare_global_loss.png", dpi=300)
plt.close()


# ======================================================
# 3️⃣ ROUND TIME COMPARISON (Sửa để tránh KeyError)
# ======================================================
if "round_time" in iid["global"] and "round_time" in noniid["global"]:
    plt.figure(figsize=(8,5))
    plt.plot(rounds_iid, iid["global"]["round_time"], marker="o", linewidth=2, label="IID")
    plt.plot(rounds_noniid, noniid["global"]["round_time"], marker="s", linewidth=2, label="Non-IID")
    plt.title("Round Time Comparison")
    plt.xlabel("Communication Round")
    plt.ylabel("Round Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/compare_round_time.png", dpi=300)
    plt.close()
else:
    print("⚠️ Bỏ qua biểu đồ 3: Một trong các file thiếu dữ liệu 'round_time'")

# ======================================================
# 4️⃣ AVERAGE LOCAL ACCURACY
# ======================================================
iid_avg_local = compute_avg_local_accuracy(iid)
noniid_avg_local = compute_avg_local_accuracy(noniid)

plt.figure(figsize=(8,5))

plt.plot(rounds_iid, iid_avg_local,
         marker="o", linewidth=2, label="IID")

plt.plot(rounds_noniid, noniid_avg_local,
         marker="s", linewidth=2, label="Non-IID")

plt.title("Average Local Accuracy: IID vs Non-IID")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.ylim(0,1)

plt.legend()
plt.tight_layout()

plt.savefig(f"{SAVE_DIR}/compare_avg_local_accuracy.png", dpi=300)
plt.close()


# ======================================================
# 5️⃣ ZKP VERIFICATION TIME (AVERAGE)
# ======================================================
iid_ver = iid["global"].get("verification_time", [])
noniid_ver = noniid["global"].get("verification_time", [])

if len(iid_ver) > 0 and len(noniid_ver) > 0:

    iid_avg = np.mean(iid_ver)
    noniid_avg = np.mean(noniid_ver)

    plt.figure(figsize=(6,5))

    labels = ["IID", "Non-IID"]
    values = [iid_avg, noniid_avg]

    plt.bar(labels, values)

    plt.title("Average ZKP Verification Time")
    plt.ylabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/compare_zkp_verification_time.png", dpi=300)
    plt.close()


# ======================================================
# 6️⃣ TRAINING TIME COMPARISON
# ======================================================
iid_train = avg_train_time(iid)
noniid_train = avg_train_time(noniid)

plt.figure(figsize=(6,5))

labels = ["IID", "Non-IID"]
values = [iid_train, noniid_train]

plt.bar(labels, values)

plt.title("Average Client Training Time")
plt.ylabel("Seconds")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/compare_train_time.png", dpi=300)
plt.close()

# ======================================================
# 7️⃣ AVERAGE REPUTATION PER ROUND
# ======================================================

def compute_avg_reputation(data):

    rounds = data["global"]["round"]
    clients = data["clients"]

    avg_rep = []

    for i in range(len(rounds)):

        reps = []

        for cid in clients:
            reps.append(clients[cid]["reputation"][i]["value"])

        avg_rep.append(np.mean(reps))

    return avg_rep


iid_rep = compute_avg_reputation(iid)
noniid_rep = compute_avg_reputation(noniid)

plt.figure(figsize=(8,5))

plt.plot(rounds_iid, iid_rep,
         marker="o", linewidth=2, label="IID")

plt.plot(rounds_noniid, noniid_rep,
         marker="s", linewidth=2, label="Non-IID")

plt.title("Average Client Reputation: IID vs Non-IID")
plt.xlabel("Communication Round")
plt.ylabel("Reputation Score")

plt.legend()
plt.tight_layout()

plt.savefig(f"{SAVE_DIR}/compare_reputation.png", dpi=300)
plt.close()

# ======================================================
# 8️⃣ REPUTATION DISTRIBUTION
# ======================================================

def collect_reputation(data):

    reps = []

    for cid in data["clients"]:
        for r in data["clients"][cid]["reputation"]:
            reps.append(r["value"])

    return reps


iid_reps = collect_reputation(iid)
noniid_reps = collect_reputation(noniid)

import seaborn as sns

plt.figure(figsize=(6,5))

sns.boxplot(data=[iid_reps, noniid_reps])
plt.xticks([0,1], ["IID", "Non-IID"])

plt.title("Reputation Comparison (Boxplot)")
plt.ylabel("Reputation Score")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/boxplot_reputation.png", dpi=300)
plt.close()


print("✅ All comparison plots saved in:", SAVE_DIR)