import torch
import torch.nn as nn
import torch.optim as optim


def train_local(
    model,
    data,
    targets,
    epochs=1,
    lr=0.05,
    dp=False,
    noise_multiplier=0.0,
    max_grad_norm=1.0
):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()

        outputs = model(data).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()

        if dp:
            # ---- Gradient Clipping ----
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm
            )

            # ---- Add Gaussian Noise ----
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=noise_multiplier * max_grad_norm,
                        size=param.grad.shape,
                        device=param.grad.device
                    )
                    param.grad += noise

        optimizer.step()

    return model.state_dict()