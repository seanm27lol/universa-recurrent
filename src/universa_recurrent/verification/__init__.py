"""Independent numerical checkers: no calls into the producer's solver."""
from .checks import CheckResult, check_projection, check_optimality, verify_record
__all__ = ["CheckResult", "check_projection", "check_optimality", "verify_record"]
