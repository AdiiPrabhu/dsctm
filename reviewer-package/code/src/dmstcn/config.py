"""Typed configuration for the D-MSTCN architecture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DMSTCNConfig:
    input_dim: int
    num_classes: int
    num_subjects: int
    hidden_dim: int = 128
    subject_embedding_dim: int = 8
    kernel_size: int = 3
    dropout: float = 0.1
    branch_dilations: tuple[tuple[int, ...], ...] = (
        (1, 2, 4, 8),
        (8, 16, 32, 64),
        (32, 64, 128, 256),
    )

    def __post_init__(self) -> None:
        positive = {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "num_subjects": self.num_subjects,
            "hidden_dim": self.hidden_dim,
            "subject_embedding_dim": self.subject_embedding_dim,
            "kernel_size": self.kernel_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if len(self.branch_dilations) < 2:
            raise ValueError("at least two temporal branches are required")
        if any(not schedule or any(d <= 0 for d in schedule) for schedule in self.branch_dilations):
            raise ValueError("branch dilation schedules must be non-empty and positive")

