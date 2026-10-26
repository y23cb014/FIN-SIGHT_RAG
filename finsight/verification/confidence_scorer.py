"""
Confidence Scorer & Calibrated Trust Engine for FinSight-RAG
Evaluates answer grounding, numerical validity, source authority, and temporal alignment.
Directly addresses trust limitations highlighted in MimirRAG.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from finsight.config import CONFIDENCE_WEIGHTS

@dataclass
class ConfidenceBreakdown:
    overall_score: float
    confidence_level: str  # "High", "Moderate", "Low"
    source_agreement: bool
    reporting_period_aligned: bool
    numerical_grounded: bool
    calculation_verified: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "confidence_level": self.confidence_level,
            "source_agreement": self.source_agreement,
            "reporting_period_aligned": self.reporting_period_aligned,
            "numerical_grounded": self.numerical_grounded,
            "calculation_verified": self.calculation_verified,
            "explanation": self.explanation
        }

class ConfidenceScorer:
    """Calculates a multi-factor calibrated trust metric for financial answers."""

    @classmethod
    def evaluate_confidence(cls, 
                            has_citations: bool, 
                            period_aligned: bool, 
                            calculation_verified: bool, 
                            has_conflict: bool = False,
                            retrieval_relevance: float = 0.85) -> ConfidenceBreakdown:
        
        # Factor weights
        score = 0.0
        
        # 1. Source agreement (no unresolvable conflict)
        source_agree = not has_conflict
        if source_agree:
            score += CONFIDENCE_WEIGHTS["source_agreement"] * 100
        else:
            score += CONFIDENCE_WEIGHTS["source_agreement"] * 50  # Partial credit if resolved

        # 2. Period Alignment
        if period_aligned:
            score += CONFIDENCE_WEIGHTS["period_alignment"] * 100
        else:
            score += CONFIDENCE_WEIGHTS["period_alignment"] * 40

        # 3. Numerical Grounding
        if has_citations:
            score += CONFIDENCE_WEIGHTS["numerical_grounding"] * 100
        else:
            score += CONFIDENCE_WEIGHTS["numerical_grounding"] * 30

        # 4. Calculation Verification
        if calculation_verified:
            score += CONFIDENCE_WEIGHTS["calculation_verification"] * 100
        else:
            score += CONFIDENCE_WEIGHTS["calculation_verification"] * 70

        # Boost or dampen by retrieval relevance
        score = round(min(100.0, max(10.0, score * (0.8 + 0.2 * retrieval_relevance))), 1)

        level = "High" if score >= 85.0 else ("Moderate" if score >= 65.0 else "Low")
        
        expl = (
            f"Confidence calculated at {score}% ({level}). "
            f"Source agreement: {'✓' if source_agree else '⚠ Discrepancy checked'}; "
            f"Period alignment: {'✓' if period_aligned else '⚠ Check year'}; "
            f"Evidence citations: {'✓' if has_citations else '✗ Missing citations'}; "
            f"Calculation verified: {'✓' if calculation_verified else '— N/A'}."
        )

        return ConfidenceBreakdown(
            overall_score=score,
            confidence_level=level,
            source_agreement=source_agree,
            reporting_period_aligned=period_aligned,
            numerical_grounded=has_citations,
            calculation_verified=calculation_verified,
            explanation=expl
        )
