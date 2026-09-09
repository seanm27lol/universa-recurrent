"""Optional PyTorch implementations of learned structured recurrence."""
from .data import StructuredFlowDataset, candidate_bases
from .model import (
    NeuralConfig,
    StructuredRecurrentNet,
    future_regret_targets,
    training_loss,
)
from .verification import NeuralCheckResult, verify_neural_record
from .v2 import (
    AmbientRecurrentControl,
    CalibrationConfig,
    DecisionPolicy,
    DirectMultiHypothesisNet,
    MultiHypothesisRecurrentNet,
    UntiedMultiHypothesisNet,
    V2Config,
    ambient_training_loss,
    apply_policy_to_trajectory,
    parameter_count,
    v2_training_loss,
)
from .v2_verification import V2CheckResult, verify_v2_record

__all__ = [
    "StructuredFlowDataset",
    "candidate_bases",
    "NeuralConfig",
    "StructuredRecurrentNet",
    "future_regret_targets",
    "training_loss",
    "NeuralCheckResult",
    "verify_neural_record",
    "AmbientRecurrentControl",
    "CalibrationConfig",
    "DecisionPolicy",
    "DirectMultiHypothesisNet",
    "MultiHypothesisRecurrentNet",
    "UntiedMultiHypothesisNet",
    "V2Config",
    "ambient_training_loss",
    "apply_policy_to_trajectory",
    "parameter_count",
    "v2_training_loss",
    "V2CheckResult",
    "verify_v2_record",
]
