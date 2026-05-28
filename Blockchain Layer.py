import numpy as np
import json
import hashlib
import time
import os

BASE_REWARD = 0.5

REP_MIN = 0.1
REP_MAX = 1.5

REWARD_ALPHA = 0.05

BASE_LAMBDA = 0.05
MAX_LAMBDA = 0.2

os.makedirs("blockchain_logs", exist_ok=True)

# =========================================================
# DEFENSE FUNCTIONS
# =========================================================

def flatten_weights(weights):

    return np.concatenate([
        w.flatten() for w in weights
    ])


def compute_delta(global_weights, local_weights):

    gw = flatten_weights(global_weights)
    lw = flatten_weights(local_weights)

    delta = np.linalg.norm(
        lw - gw
    ) / (np.linalg.norm(gw) + 1e-10)

    return delta

# =========================================================
# REPUTATION
# =========================================================

class ReputationManager:

    def __init__(self):

        self.reputation = {}

    def get(self, client_id):

        if client_id not in self.reputation:

            self.reputation[client_id] = BASE_REWARD

        return self.reputation[client_id]

    def reward(self, client_id, gamma):

        r_old = self.get(client_id)

        reward_signal = gamma - 0.3

        r_new = (
            r_old
            + REWARD_ALPHA * reward_signal
        )

        r = 0.9 * r_old + 0.1 * r_new

        r = np.clip(
            r,
            REP_MIN,
            REP_MAX
        )

        self.reputation[client_id] = float(r)

    def penalize(self, client_id):

        r = self.get(client_id)

        r = r * 0.5

        r = np.clip(
            r,
            REP_MIN,
            REP_MAX
        )

        self.reputation[client_id] = float(r)

    def update_reputation(
        self,
        client_id,
        delta
    ):

        r = self.get(client_id)

        r = r + delta

        r = np.clip(
            r,
            REP_MIN,
            REP_MAX
        )

        self.reputation[client_id] = float(r)


reputation_manager = ReputationManager()

# =========================================================
# CLIENT EVALUATION
# =========================================================
def evaluate_clients(global_weights, client_weights_dict, clients_info):

    scores = []

    for info in clients_info:

        cid = info["client_id"]

        delta = compute_delta(
            global_weights,
            info["params"]
        )

        score = np.exp(-delta)

        reputation = reputation_manager.get(cid)

        scores.append(delta)

    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)

    results = {}

    for info in clients_info:

        cid = info["client_id"]

        delta = compute_delta(
            global_weights,
            info["params"]
        )

        score = np.exp(-delta)

        reputation = reputation_manager.get(cid)

        results[cid] = {
            "score": float(score),
            "reputation": float(reputation)
        }

    return results, Q1, Q3

# =========================================================
# BLOCKCHAIN LEDGER
# =========================================================
class BlockchainLedger:

    def __init__(self):

        self.chain = []

    def create_block(
        self,
        round_id,
        client_id,
        reputation,
        reward,
        delta
    ):

        previous_hash = (
            self.chain[-1]["hash"]
            if self.chain else "0"
        )

        block = {
            "timestamp": time.time(),
            "round": round_id,
            "client_id": client_id,
            "reputation": reputation,
            "reward": reward,
            "delta": delta,
            "previous_hash": previous_hash
        }

        block_str = json.dumps(
            block,
            sort_keys=True
        ).encode()

        block_hash = hashlib.sha256(
            block_str
        ).hexdigest()

        block["hash"] = block_hash

        self.chain.append(block)

        self.save_chain()

    def save_chain(self):

        with open(
            "blockchain_logs/ledger.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.chain,
                f,
                indent=4
            )


ledger = BlockchainLedger()