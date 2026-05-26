import os
import pandas as pd
import matplotlib.pyplot as plt # type: ignore

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.centralized_train import centralized_training
from src.federated_train import federated_training, evaluate
from src.privacy import approximate_epsilon


DATA_PATH = "data/diabetes.csv"
RESULTS_DIR = "results"

ROUNDS = 20
LOCAL_EPOCHS = 5
NUM_CLIENTS = 5
SEED = 42

NOISE_LEVELS = [0.0, 0.5, 1.0, 1.5]


def run_experiments():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df, seed=SEED)

    results = []

    # -------- Centralized --------
    print("\nRunning Centralized Training")
    acc, loss = centralized_training(
        X_train, y_train, X_test, y_test,
        epochs=ROUNDS * LOCAL_EPOCHS
    )

    results.append({
        "Method": "Centralized",
        "Noise": 0.0,
        "Epsilon": float("inf"),
        "Accuracy": acc,
        "Loss": loss
    })

    # -------- Federated Variants --------
    for noise in NOISE_LEVELS:
        print(f"\nRunning Federated Training (noise={noise})")

        model = federated_training(
            X_train,
            y_train,
            X_test,
            y_test,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            num_clients=NUM_CLIENTS,
            dp=(noise > 0),
            noise_multiplier=noise,
            max_grad_norm=1.0
        )

        acc = evaluate(model, X_test, y_test)

        epsilon = approximate_epsilon(
            noise_multiplier=noise,
            steps=ROUNDS * LOCAL_EPOCHS,
            sample_rate=1 / NUM_CLIENTS
        )

        results.append({
            "Method": "Federated-DP" if noise > 0 else "Federated",
            "Noise": noise,
            "Epsilon": epsilon,
            "Accuracy": acc,
            "Loss": None
        })

    # -------- Save CSV --------
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n=== Final Results ===")
    print(df_results)

    # -------- Plot --------
    dp_results = df_results[df_results["Method"] != "Centralized"]

    plt.figure()
    plt.plot(dp_results["Noise"], dp_results["Accuracy"])
    plt.xlabel("Noise Multiplier")
    plt.ylabel("Accuracy")
    plt.title("Privacy vs Accuracy")
    plt.savefig(os.path.join(RESULTS_DIR, "privacy_vs_accuracy.png"))
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_experiments()