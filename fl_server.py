import flwr as fl
from server.strategy import SecureFLStrategy
from server.config import ROUNDS

def main():

    strategy = SecureFLStrategy()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )

if __name__ == "__main__":
    print("Starting Federated Learning Server...")
    main()