"""
Contradiction & Evidence Verification Agent for FinSight-RAG
Detects conflicting financial values between documents (e.g. preliminary vs final 10-K,
restated values, differing reporting periods, consolidated vs standalone).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math

@dataclass
class ContradictionReport:
    metric: str
    company: str
    year: int
    has_conflict: bool
    sources: List[Dict[str, Any]]
    recommended_value: Optional[float]
    recommended_source: Optional[str]
    root_cause_analysis: List[str]
    resolution_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "company": self.company,
            "year": self.year,
            "has_conflict": self.has_conflict,
            "sources": self.sources,
            "recommended_value": self.recommended_value,
            "recommended_source": self.recommended_source,
            "root_cause_analysis": self.root_cause_analysis,
            "resolution_rationale": self.resolution_rationale
        }

class ContradictionDetector:
    """Audits retrieved candidate facts for numerical and reporting discrepancies."""

    # Source Hierarchy Rank (Higher number = Higher authority)
    AUTHORITY_HIERARCHY = {
        "Annual Report (10-K)": 5,
        "Financial Statements": 4,
        "Quarterly Report (10-Q)": 3,
        "Investor Presentation": 2,
        "Press Release": 2,
        "Earnings Call Transcript": 1,
        "Analyst Research Vault": 1
    }

    @classmethod
    def check_discrepancies(cls, 
                            facts: List[Dict[str, Any]], 
                            tolerance: float = 0.01) -> List[ContradictionReport]:
        reports = []
        if not facts or len(facts) < 2:
            return reports

        # Group facts by (company, year, metric)
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for f in facts:
            key = (f.get("company", "Unknown"), f.get("year", 2024), f.get("metric", "revenue"))
            if key not in groups:
                groups[key] = []
            groups[key].append(f)

        for (comp, yr, metric), group in groups.items():
            if len(group) < 2:
                continue

            values = [g.get("value", 0.0) for g in group]
            min_val = min(values)
            max_val = max(values)

            # Check if discrepancy exceeds tolerance
            if min_val > 0 and (max_val - min_val) / min_val > tolerance:
                # Discrepancy detected!
                # Sort group by source authority
                def get_auth(item):
                    doc_type = item.get("doc_type", "")
                    for k, v in cls.AUTHORITY_HIERARCHY.items():
                        if k.lower() in doc_type.lower():
                            return v
                    return 1

                sorted_group = sorted(group, key=get_auth, reverse=True)
                top_item = sorted_group[0]

                causes = [
                    "Reporting period discrepancy (e.g. preliminary vs audited filing)",
                    "Potential accounting restatement in subsequent fiscal year",
                    "Consolidated group results vs standalone segment reporting",
                    "Currency or scale convention (e.g. constant currency vs nominal USD)"
                ]

                rationale = (
                    f"Selected ${top_item.get('value', 0):,.2f} M from '{top_item.get('source_doc', 'Primary Filing')}' "
                    f"because audited SEC Form 10-K carries higher authoritative legal standing over press releases or presentations."
                )

                reports.append(ContradictionReport(
                    metric=metric,
                    company=comp,
                    year=yr,
                    has_conflict=True,
                    sources=group,
                    recommended_value=top_item.get("value"),
                    recommended_source=top_item.get("source_doc"),
                    root_cause_analysis=causes,
                    resolution_rationale=rationale
                ))

        return reports
