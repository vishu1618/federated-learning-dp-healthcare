import torch
import torch.nn as nn
import torch.optim as optim
from src.model import LogisticModel


def centralized_training(X_train, y_train, X_test, y_test,
                         epochs=20, lr=0.05):

    model = LogisticModel(X_train.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test).squeeze()
        loss = criterion(logits, y_test).item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y_test).float().mean().item()

    return acc, loss