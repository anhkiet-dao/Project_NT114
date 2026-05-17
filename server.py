# =========================================================
# IMPORTS
# =========================================================

import flwr as fl
import numpy as np
import json
import time
import os

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

# =========================================================
# CONFIG
# =========================================================

ROUNDS = 25
NUM_CLIENTS = 5

LR = 0.0007
BETA1 = 0.9
BETA2 = 0.99
EPS = 1e-8

BASE_LAMBDA = 0.05
MAX_LAMBDA = 0.2

LAMBDA_DEFENSE = 0.02

REWARD_ALPHA = 0.05

REP_MIN = 0.1
REP_MAX = 1.5

BASE_REWARD = 0.5

WARMUP_ROUNDS = 5

os.makedirs("history", exist_ok=True)

# =========================================================
# PLACEHOLDER FUNCTIONS
# =========================================================

def verify_update(cid, server_round, status):
    return True


def verify_proof(params, proof):
    return True


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
# DEFENSE
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


def cosine_similarity(global_weights, local_weights):

    gw = flatten_weights(global_weights)
    lw = flatten_weights(local_weights)

    return np.dot(gw, lw) / (
        (np.linalg.norm(gw) * np.linalg.norm(lw))
        + 1e-10
    )


def defense_scaling(
    delta,
    lambda_defense=LAMBDA_DEFENSE
):

    gamma = 1 / (
        1 + np.exp(
            lambda_defense * (delta - 1.0)
        )
    )

    return np.clip(gamma, 0.05, 1.0)

# =========================================================
# FEDADAM
# =========================================================

class FedAdamState:

    def __init__(self):

        self.m = None
        self.v = None
        self.t = 0


fedadam = FedAdamState()


def fedadam_update(global_weights, gradients):

    fedadam.t += 1

    if fedadam.m is None:

        fedadam.m = [
            np.zeros_like(x)
            for x in gradients
        ]

        fedadam.v = [
            np.zeros_like(x)
            for x in gradients
        ]

    new_weights = []

    for i in range(len(gradients)):

        g = gradients[i]

        g = np.clip(g, -1.0, 1.0)

        fedadam.m[i] = (
            BETA1 * fedadam.m[i]
            + (1 - BETA1) * g
        )

        fedadam.v[i] = (
            BETA2 * fedadam.v[i]
            + (1 - BETA2) * (g ** 2)
        )

        m_hat = (
            fedadam.m[i]
            / (1 - BETA1 ** fedadam.t)
        )

        v_hat = (
            fedadam.v[i]
            / (1 - BETA2 ** fedadam.t)
        )

        new_w = (
            global_weights[i]
            + LR * m_hat
            / (np.sqrt(v_hat) + EPS)
        )

        new_weights.append(new_w)

    return new_weights

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
# STRATEGY
# =========================================================

class SecureFLStrategy(
    fl.server.strategy.FedAvg
):

    def __init__(self):

        super().__init__(
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS
        )

        self.start_time = None

        self.global_weights = None

        self.history = {
            "global": {
                "round": [],
                "accuracy": [],
                "loss": [],
                "verification_time": [],
                "penalty_clients": [],
                "round_time": []
            },
            "clients": {}
        }

    # =====================================================
    # AGGREGATE FIT
    # =====================================================

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        self.start_time = time.time()

        if not results:
            return None, {}

        clients_info = []

        round_verify_times = []

        penalty_clients = []

        # =================================================
        # VERIFY
        # =================================================

        for client, fit_res in results:

            metrics = fit_res.metrics

            cid = str(metrics["client_id"])

            params = parameters_to_ndarrays(
                fit_res.parameters
            )

            proof = json.loads(
                metrics.get("proof", "{}")
            )

            print(f"\nClient {cid} update received")

            start = time.time()

            verified = verify_proof(
                params,
                proof
            )

            round_verify_times.append(
                time.time() - start
            )

            if not verified:

                print(
                    f"❌ ZKP FAILED for Client {cid}"
                )

                reputation_manager.update_reputation(
                    cid,
                    -1.0
                )

                continue

            verify_update(
                cid,
                server_round,
                True
            )

            clients_info.append({
                "params": params,
                "client_id": cid,
                "test_acc": metrics.get(
                    "local_accuracy",
                    0
                ),
                "test_loss": metrics.get(
                    "local_loss",
                    0
                ),
                "train_time": metrics.get(
                    "train_time",
                    0
                )
            })

        if not clients_info:
            return None, {}

        # =================================================
        # INIT GLOBAL
        # =================================================

        if self.global_weights is None:

            self.global_weights = (
                clients_info[0]["params"]
            )

        client_weights_dict = {

            info["client_id"]: info["params"]

            for info in clients_info
        }

        eval_results, Q1, Q3 = evaluate_clients(
            self.global_weights,
            client_weights_dict,
            clients_info
        )

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        gradients = []

        final_weights = []

        LAMBDA = min(
            MAX_LAMBDA,
            BASE_LAMBDA + server_round * 0.005
        )

        # =================================================
        # PROCESS CLIENTS
        # =================================================

        for info in clients_info:

            cid = info["client_id"]

            res = eval_results[cid]

            reputation = res["reputation"]

            score = res["score"]

            if score < 0.4 or reputation < 0.2:

                print(
                    f"🚫 Skip client {cid}"
                )

                penalty_clients.append(cid)

                continue

            delta = compute_delta(
                self.global_weights,
                info["params"]
            )

            if delta < lower or delta > upper:

                print(
                    f"⚠ Outlier Client {cid}"
                )

                reputation *= 0.7

                penalty_clients.append(cid)

            gamma = np.exp(
                -LAMBDA * delta
            )

            reward = (
                np.sqrt(reputation)
                * gamma
            )

            reward = np.clip(
                reward,
                0.05,
                1.5
            )

            grad = [

                local_w - global_w

                for local_w, global_w

                in zip(
                    info["params"],
                    self.global_weights
                )
            ]

            gradients.append(grad)

            final_weights.append(reward)

            print(
                f"Client {cid}"
                f" | reputation={reputation:.3f}"
                f" | reward={reward:.3f}"
            )

        if not gradients:

            print(
                "❌ No valid clients"
            )

            return None, {}

        # =================================================
        # AGGREGATE GRADIENT
        # =================================================

        total_weight = (
            sum(final_weights)
            + 1e-8
        )

        agg_grad = []

        for layer_idx in range(
            len(gradients[0])
        ):

            layer_sum = sum(

                grad[layer_idx] * weight

                for grad, weight

                in zip(
                    gradients,
                    final_weights
                )
            )

            layer_avg = (
                layer_sum
                / total_weight
            )

            layer_avg = np.clip(
                layer_avg,
                -1,
                1
            )

            agg_grad.append(layer_avg)

        # =================================================
        # FEDADAM
        # =================================================

        self.global_weights = fedadam_update(
            self.global_weights,
            agg_grad
        )

        # =================================================
        # LOGGING
        # =================================================

        if round_verify_times:

            self.history["global"][
                "verification_time"
            ].append(
                float(np.mean(round_verify_times))
            )

            self.history["global"][
                "penalty_clients"
            ].append(
                penalty_clients
            )

        return (
            ndarrays_to_parameters(
                self.global_weights
            ),
            {}
        )

    # =====================================================
    # EVALUATE
    # =====================================================

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures
    ):

        if not results:
            return None, {}

        accuracies = [

            r.metrics.get(
                "accuracy",
                0
            )

            for _, r in results
        ]

        losses = [
            r.loss
            for _, r in results
        ]

        avg_acc = float(
            np.mean(accuracies)
        )

        avg_loss = float(
            np.mean(losses)
        )

        print(
            f"\n--- Round {server_round} ---"
        )

        print(
            f"Accuracy={avg_acc:.4f}"
            f" | Loss={avg_loss:.4f}"
        )

        self.history["global"][
            "round"
        ].append(server_round)

        self.history["global"][
            "accuracy"
        ].append(avg_acc)

        self.history["global"][
            "loss"
        ].append(avg_loss)

        if self.start_time:

            duration = (
                time.time()
                - self.start_time
            )

            self.history["global"][
                "round_time"
            ].append(float(duration))

        with open(
            "history/server_history.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.history,
                f,
                indent=4
            )

        return avg_loss, {
            "accuracy": avg_acc
        }

# =========================================================
# MAIN
# =========================================================

def main():

    strategy = SecureFLStrategy()

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=ROUNDS
        ),

        strategy=strategy
    )


if __name__ == "__main__":

    print(
        "Starting Federated Learning Server..."
    )

    main()