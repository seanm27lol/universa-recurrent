"""Optional PyTorch implementation of learned structured recurrence."""
from .data import StructuredFlowDataset, candidate_bases
from .model import NeuralConfig, StructuredRecurrentNet, training_loss
from .verification import NeuralCheckResult, verify_neural_record

__all__ = [
    "StructuredFlowDataset",
    "candidate_bases",
    "NeuralConfig",
    "StructuredRecurrentNet",
    "training_loss",
    "NeuralCheckResult",
    "verify_neural_record",
]
