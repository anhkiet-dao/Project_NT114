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

from Blockchain_Layer import (
    verify_proof,
    verify_update,
    evaluate_clients,
    reputation_manager,
    compute_delta,
    ledger
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

os.makedirs("history", exist_ok=True)

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
# STRATEGY
# =========================================================
class AggregationServer(
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
    
# =========================================================
# MAIN
# =========================================================

def main():

    strategy = AggregationServer()

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=ROUNDS
        ),

        strategy=strategy
    )


if __name__ == "__main__":

    print(
        "Starting Aggregation Server..."
    )

    main()