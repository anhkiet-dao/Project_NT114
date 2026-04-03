import json
import matplotlib.pyplot as plt
import os
import numpy as np

# ==============================
# CONFIG
# ==============================
MODE = "non_iid"
SAVE_DIR = f"plots/{MODE}"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# Load JSON
# ==============================
with open("history/server_history_fedadam.json", "r") as f:
    history = json.load(f)

global_data = history["global"]
clients_data = history["clients"]

rounds = global_data["round"]

# ==============================
# STYLE
# ==============================
plt.style.use("seaborn-v0_8-whitegrid")

# ==============================
# Helper function (fix mismatch)
# ==============================
def align_xy(x, y):
    n = min(len(x), len(y))
    return x[:n], y[:n]

# ==============================
# 1️⃣ Global Accuracy
# ==============================
plt.figure(figsize=(8,5))

plt.plot(rounds, global_data["accuracy"], marker="o", linewidth=2)

plt.title("Global Model Accuracy")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.ylim(0,1)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/global_accuracy.png", dpi=300)
plt.close()

# ==============================
# 2️⃣ Train Time per Client
# ==============================
plt.figure(figsize=(8,5))

for cid, data in clients_data.items():

    x, y = align_xy(data["round"], data["train_time"])

    plt.plot(x, y, marker="o", label=f"Client {cid}")

plt.title("Training Time per Client")
plt.xlabel("Communication Round")
plt.ylabel("Training Time (seconds)")

plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/train_time.png", dpi=300)
plt.close()

# ==============================
# 3️⃣ Local Test Accuracy
# ==============================
plt.figure(figsize=(8,5))

for cid, data in clients_data.items():

    x, y = align_xy(data["round"], data["test_accuracy"])

    plt.plot(x, y, marker="o", label=f"Client {cid}")

plt.title("Local Test Accuracy per Client")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.ylim(0,1)

plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/local_accuracy.png", dpi=300)
plt.close()

# ==============================
# 4️⃣ Loss Comparison
# ==============================
plt.figure(figsize=(8,5))

plt.plot(rounds, global_data["loss"], linewidth=3, label="Global Loss")

for cid, data in clients_data.items():

    x, y = align_xy(data["round"], data["test_loss"])

    plt.plot(
        x,
        y,
        linestyle="--",
        alpha=0.7,
        label=f"Client {cid}"
    )

plt.title("Loss Comparison (Global vs Clients)")
plt.xlabel("Communication Round")
plt.ylabel("Loss")

plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/loss_comparison.png", dpi=300)
plt.close()

# ==============================
# 5️⃣ Accuracy Comparison
# ==============================
plt.figure(figsize=(8,5))

plt.plot(rounds, global_data["accuracy"], linewidth=3, label="Global Accuracy")

for cid, data in clients_data.items():

    x, y = align_xy(data["round"], data["test_accuracy"])

    plt.plot(
        x,
        y,
        linestyle="--",
        alpha=0.7,
        label=f"Client {cid}"
    )

plt.title("Accuracy Comparison (Global vs Clients)")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")

plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/accuracy_comparison.png", dpi=300)
plt.close()

# =====================================================
# 🔐 ZKP SECURITY METRICS
# =====================================================

# =====================================================
# 6️⃣ ZKP Verification Time (Bản vẽ lại "Nhìn là hiểu")
# =====================================================
v_times = global_data.get("verification_time", [])
rounds = global_data.get("round", [])

if len(v_times) > 0:
    plt.figure(figsize=(8, 5))
    
    # Đồng bộ x (rounds) và y (v_times)
    x_zkp, y_zkp = align_xy(rounds, v_times)
    mean_val = np.mean(y_zkp)

    # Vẽ cột thời gian Verification
    bars = plt.bar(x_zkp, y_zkp, color="skyblue", edgecolor="black", alpha=0.8, label="Verification Time")
    
    # Vẽ đường trung bình (tương tự như cách bạn vẽ Global Loss)
    plt.axhline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.3f}s")

    # Thêm số liệu trực tiếp trên đầu cột cho rõ ràng
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}s', ha='center', va='bottom', fontsize=9)

    plt.title("ZKP Verification Performance per Round")
    plt.xlabel("Communication Round")
    plt.ylabel("Time (seconds)")
    plt.xticks(x_zkp) # Đảm bảo hiện đủ số round trên trục X
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/zkp_performance.png", dpi=300)
    plt.close()

# ==============================
# 7️⃣ Rejected Clients (per round)
# ==============================
rejected = global_data.get("penalty_clients", [])

if len(rejected) > 0:

    x = []
    y = []

    for i, round_clients in enumerate(rejected):
        x.append(i + 1)                 # round
        y.append(len(round_clients))    # số client bị reject

    plt.figure(figsize=(8,5))

    plt.bar(x, y)

    # highlight round có reject
    for i, val in enumerate(y):
        if val > 0:
            plt.text(x[i], val, str(val), ha='center', color='red')

    plt.title("Rejected Clients per Round")
    plt.xlabel("Communication Round")
    plt.ylabel("Number of Rejected Clients")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/rejected_clients.png", dpi=300)
    plt.close()

# ==============================
# 8️⃣ Global Accuracy + Loss
# ==============================
plt.figure(figsize=(8,5))

plt.plot(rounds, global_data["accuracy"], marker="o", label="Accuracy")
plt.plot(rounds, global_data["loss"], marker="s", label="Loss")

plt.title("Global Model Performance")
plt.xlabel("Communication Round")

plt.legend()

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/global_performance.png", dpi=300)
plt.close()

# ==============================
# 9️⃣ Client Reputation
# ==============================
plt.figure(figsize=(8,5))

for cid, data in clients_data.items():

    if "reputation" not in data:
        continue

    r_rounds = []
    r_values = []

    for r in data["reputation"]:
        r_rounds.append(r["round"])
        r_values.append(r["value"])

    plt.plot(r_rounds, r_values, marker="o", label=f"Client {cid}")

plt.title("Client Reputation Evolution")
plt.xlabel("Communication Round")
plt.ylabel("Reputation Score")

plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/client_reputation.png", dpi=300)
plt.close()

print(f"\n✅ All plots saved in folder: {SAVE_DIR}")