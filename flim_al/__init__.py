"""
flim_al — Active Learning module for FLIM decoders.

Pipeline:
  1. UncertaintyDecoder: wraps any FLIM decoder + uncertainty head
  2. AcquisitionFunction: scores unlabeled images (entropy / BALD)
  3. ALLoop: iterative label selection + retraining
"""

from .uncertainty_decoder import UncertaintyDecoder
from .acquisition import entropy_map, bald_score, least_confidence
from .al_loop import ALLoop

__all__ = ["UncertaintyDecoder", "entropy_map", "bald_score", "least_confidence", "ALLoop"]
