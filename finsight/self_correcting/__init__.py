"""Self-correcting RAG module for FinSight-RAG"""
from .engine import SelfCorrectingEngine, CorrectionAttempt

__all__ = ["SelfCorrectingEngine", "CorrectionAttempt"]
