import torch
from src.model import LogisticModel
from src.client import train_local
from src.server import federated_average


def split_clients(X, y, num_clients=5):
    data_size = len(X)
    shard_size = data_size // num_clients

    clients = []

    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size if i != num_clients - 1 else data_size
        clients.append((X[start:end], y[start:end]))

    return clients


def evaluatee(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test).squeeze()
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()
        accuracy = (preds == y_test).float().mean()
    return accuracy.item()


def federated_training(
    X_train,
    y_train,
    X_test,
    y_test,
    rounds=20,
    local_epochs=5,
    num_clients=5,
    dp=False,
    noise_multiplier=0.0,
    max_grad_norm=1.0
):
    input_dim = X_train.shape[1]
    global_model = LogisticModel(input_dim)

    clients = split_clients(X_train, y_train, num_clients)

    for r in range(rounds):
        client_weights = []

        for client_data, client_targets in clients:
            local_model = LogisticModel(input_dim)
            local_model.load_state_dict(global_model.state_dict())

            updated_weights = train_local(
                local_model,
                client_data,
                client_targets,
                epochs=local_epochs,
                dp=dp,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm
            )

            client_weights.append(updated_weights)

        global_model = federated_average(global_model, client_weights)

        acc = evaluate(global_model, X_test, y_test)
        print(f"Round {r+1} - Global Accuracy: {acc:.4f}")

    return global_model
