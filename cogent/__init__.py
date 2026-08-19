"""Reference implementation of the COGENT manuscript pipeline."""

from .config import ExperimentConfig, manuscript_defaults
from .model import COGENTModel

__all__ = ["ExperimentConfig", "manuscript_defaults", "COGENTModel"]
