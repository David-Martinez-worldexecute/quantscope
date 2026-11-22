"""A two-layer Forward-Forward network, trained one layer at a time.

Each layer has its own objective and its own optimiser; earlier layers are frozen
and run under ``no_grad`` while the current layer learns. There is no global
backward pass, no shared loss, and no label head — the label is *input*, appended
to the image as ten one-hot columns.

The classifier is the same trick read backwards: for each of the ten candidate
labels, append it and sum the goodness of every layer's activations; predict the
label with the highest total.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import TernaryLinear

PIXELS = 28 * 28
LABELS = 10
IN_DIM = PIXELS + LABELS


def one_hot(y: torch.Tensor, n_classes: int = LABELS) -> torch.Tensor:
    return F.one_hot(y, n_classes).float()


def goodness(h: torch.Tensor) -> torch.Tensor:
    """Per-sample mean of squared activations — Hinton's goodness."""
    return h.pow(2).mean(dim=1)


def ff_loss(g_pos: torch.Tensor, g_neg: torch.Tensor, theta: float) -> torch.Tensor:
    """Two-sided logistic loss: push positives above theta, negatives below."""
    return F.softplus(-(g_pos - theta)).mean() + F.softplus(g_neg - theta).mean()


class FFNet(nn.Module):
    """Layer-wise trained classifier; input is ``[pixels | one-hot label]``."""

    def __init__(self, hidden: int = 500, n_layers: int = 2, spans=None) -> None:
        super().__init__()
        dims = [IN_DIM] + [hidden] * n_layers
        # Only the first layer sees the heterogeneous input ([pixels | labels]); from
        # layer 1 onwards the input is a hidden layer, so a single scale per row is
        # the right grouping. Passing the label spans to later layers would be a bug
        # that silently "works" only when the widths happen to match.
        self.layers = nn.ModuleList(
            TernaryLinear(dims[i], dims[i + 1], spans=spans if i == 0 else None)
            for i in range(n_layers)
        )
        self.n_layers = n_layers
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = []
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h))
            activations.append(h)
        return activations

    def features(self, x: torch.Tensor, upto: int) -> torch.Tensor:
        """Activations of layer ``upto``; layers before it run without grad."""
        with torch.no_grad():
            h = x
            for index in range(upto):
                h = torch.relu(self.layers[index](h))
        return torch.relu(self.layers[upto](h))


def train_layer(
    model: FFNet,
    layer_index: int,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1,
    lr: float = 1e-3,
    theta: float = 1.0,
    batch_size: int = 128,
    generator: torch.Generator | None = None,
    log=print,
) -> list[float]:
    """Train one layer; returns the mean loss per epoch."""
    optimiser = torch.optim.Adam(model.layers[layer_index].parameters(), lr=lr)
    losses: list[float] = []
    n = x.size(0)
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        total, seen = 0.0, 0
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            xb, yb = x[index], y[index]
            y_neg = (yb + torch.randint(1, LABELS, (yb.size(0),), generator=generator)) % LABELS
            positive = torch.cat([xb, one_hot(yb)], dim=1)
            negative = torch.cat([xb, one_hot(y_neg)], dim=1)

            h_pos = model.features(positive, layer_index)
            h_neg = model.features(negative, layer_index)
            loss = ff_loss(goodness(h_pos), goodness(h_neg), theta)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * yb.size(0)
            seen += yb.size(0)
        mean = total / max(seen, 1)
        losses.append(mean)
        log(f"[ff] layer {layer_index + 1}/{model.n_layers} epoch {epoch + 1}/{epochs} loss={mean:.4f}")
    return losses


def train_ff(
    model: FFNet,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1,
    lr: float = 1e-3,
    theta: float = 1.0,
    batch_size: int = 128,
    seed: int = 0,
    log=print,
) -> list[list[float]]:
    """Train every layer in order, each with its own optimiser."""
    generator = torch.Generator().manual_seed(seed)
    per_layer = []
    for layer_index in range(model.n_layers):
        per_layer.append(
            train_layer(
                model,
                layer_index,
                x,
                y,
                epochs=epochs,
                lr=lr,
                theta=theta,
                batch_size=batch_size,
                generator=generator,
                log=log,
            )
        )
    return per_layer


@torch.no_grad()
def predict(model: FFNet, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Classify by summing goodness over layers for each candidate label."""
    model.eval()
    predictions = []
    for start in range(0, x.size(0), batch_size):
        xb = x[start : start + batch_size]
        b = xb.size(0)
        scores = torch.zeros(b, LABELS)
        for candidate in range(LABELS):
            y = torch.full((b,), candidate, dtype=torch.long)
            activations = model(torch.cat([xb, one_hot(y)], dim=1))
            scores[:, candidate] = sum(goodness(a) for a in activations)
        predictions.append(scores.argmax(dim=1))
    return torch.cat(predictions)


def evaluate(model: FFNet, x: torch.Tensor, y: torch.Tensor) -> tuple[int, list[int]]:
    predictions = predict(model, x)
    correct = int((predictions == y).sum().item())
    per_class = [0] * LABELS
    for label in range(LABELS):
        mask = y == label
        per_class[label] = int((predictions[mask] == y[mask]).sum().item())
    return correct, per_class


def normalise(images: torch.Tensor) -> torch.Tensor:
    """Same normalisation torchvision's MNIST example uses."""
    return (images.float() / 255.0 - 0.1307) / 0.3081
