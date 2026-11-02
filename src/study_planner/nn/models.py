"""Small CPU-compatible PyTorch regression network definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class MLPConfig:
    """Hyperparameters controlling one MLP configuration."""

    name: str
    hidden_sizes: tuple[int, ...]
    dropout: float
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    loss_name: str = "huber"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe hyperparameter values."""
        payload = asdict(self)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload


DEFAULT_SEARCH_CONFIGS: tuple[MLPConfig, ...] = (
    MLPConfig(name="mlp_32", hidden_sizes=(32,), dropout=0.0),
    MLPConfig(name="mlp_64_32_dropout", hidden_sizes=(64, 32), dropout=0.1),
)


class MLPRegressor(nn.Module):
    """Feed-forward MLP with an unconstrained scalar regression output."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_size < 1 or not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("MLP input and hidden layer sizes must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one unbounded score estimate per input row."""
        return self.layers(inputs).squeeze(-1)
