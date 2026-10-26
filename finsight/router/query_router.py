"""
Adaptive Query Router & Query Intelligence Agent for FinSight-RAG
Classifies financial queries and selects specialized retrieval and reasoning strategies.
Main architectural novelty over static RAG pipelines (e.g. MimirRAG).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import re
from finsight.config import COMPANIES, YEARS, METRIC_ALIASES

class QueryType(str, Enum):
    FACT = "Fact Retrieval"
    NUMERICAL = "Numerical & Financial Calculation"
    MULTI_DOC = "Multi-Document Cross-Company"
    TEMPORAL = "Temporal Multi-Year Trend"
    TABLE_QA = "Table Intelligence & Statements"
    REASONING = "Qualitative & Causal Reasoning"
    CONTRADICTION = "Contradiction & Evidence Verification"
    VISUALIZATION = "Financial Visualization"
    CROSS_SOURCE = "Cross-Source (Report vs Presentation/News)"
    DECISION_ANALYSIS = "Decision & Health Analysis"

@dataclass
class RoutingDecision:
    query: str
    query_type: QueryType
    confidence: float
    target_companies: List[str]
    target_years: List[int]
    target_metrics: List[str]
    retrieval_strategy: str
    requires_calculator: bool
    requires_graph: bool
    requires_contradiction_check: bool
    requires_visualization: bool
    recommended_top_k: int
    cost_tier: str  # "FAST" or "REASONING"
    reasoning_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "target_companies": self.target_companies,
            "target_years": self.target_years,
            "target_metrics": self.target_metrics,
            "retrieval_strategy": self.retrieval_strategy,
            "requires_calculator": self.requires_calculator,
            "requires_graph": self.requires_graph,
            "requires_contradiction_check": self.requires_contradiction_check,
            "requires_visualization": self.requires_visualization,
            "recommended_top_k": self.recommended_top_k,
            "cost_tier": self.cost_tier,
            "reasoning_explanation": self.reasoning_explanation
        }

class AdaptiveQueryRouter:
    """Classifies user queries and selects optimal retrieval, computation, and verification strategies."""

    @classmethod
    def analyze_query(cls, query: str) -> RoutingDecision:
        q_lower = query.lower()

        # 1. Identify Target Companies
        detected_companies = []
        for comp in COMPANIES:
            # Check company name and common tickers
            ticker_map = {"Apple": "aapl", "Microsoft": "msft", "Amazon": "amzn", "NVIDIA": "nvda"}
            ticker = ticker_map.get(comp, "")
            if comp.lower() in q_lower or (ticker and re.search(r'\b' + ticker + r'\b', q_lower)):
                detected_companies.append(comp)

        # 2. Identify Target Years
        detected_years = []
        for yr in YEARS:
            if str(yr) in q_lower:
                detected_years.append(yr)

        # 3. Identify Target Metrics
        detected_metrics = []
        for canonical, aliases in METRIC_ALIASES.items():
            for alias in aliases:
                if alias in q_lower:
                    if canonical not in detected_metrics:
                        detected_metrics.append(canonical)
                    break

        # Additional financial ratio keywords
        ratio_keywords = ["margin", "cagr", "ratio", "growth", "roe", "roa", "ebitda", "turnover", "debt-to-equity"]
        for rk in ratio_keywords:
            if rk in q_lower and rk not in detected_metrics:
                detected_metrics.append(rk)

        # 4. Strategy & Query Type Classification
        # Visualization
        if any(w in q_lower for w in ["plot", "chart", "visualize", "graph", "show trend", "display chart"]):
            q_type = QueryType.VISUALIZATION
            strategy = "Table Extraction + Plotly Visualization Agent"
            calc = False
            graph = False
            contra = False
            viz = True
            top_k = 8
            cost_tier = "FAST"
            explanation = "Visual request detected. Triggering data extraction and Plotly chart generation."

        # Contradiction / Conflict
        elif any(w in q_lower for w in ["conflict", "contradiction", "discrepan", "two different", "conflicting", "differ", "reconcile"]):
            q_type = QueryType.CONTRADICTION
            strategy = "Evidence Verification Agent + Source Hierarchy Reconciliation"
            calc = False
            graph = True
            contra = True
            viz = False
            top_k = 10
            cost_tier = "REASONING"
            explanation = "Conflict detection inquiry. Triggering discrepancy analysis across source reports."

        # Numerical Calculation
        elif any(w in q_lower for w in ["calculate", "formula", "cagr", "margin", "ratio", "debt-to-equity", "roe", "roa", "percentage of"]):
            q_type = QueryType.NUMERICAL
            strategy = "Table Intelligence + Deterministic Financial Calculator"
            calc = True
            graph = False
            contra = False
            viz = False
            top_k = 6
            cost_tier = "FAST"
            explanation = "Mathematical financial metric inquiry. Forwarding to deterministic financial calculation engine."

        # Multi-Document / Cross-Company
        elif len(detected_companies) > 1 or any(w in q_lower for w in ["compare", "versus", "vs", "which company", "rank", "highest revenue", "largest increase"]):
            q_type = QueryType.MULTI_DOC
            strategy = "Cross-Document Fusion Retrieval + Multi-Company Synthesis"
            calc = any(w in q_lower for w in ["growth", "margin", "rank", "highest"])
            graph = False
            contra = False
            viz = False
            top_k = 10
            cost_tier = "REASONING"
            explanation = "Multi-document comparative inquiry across multiple entities or documents."

        # Temporal / Multi-Year Trend
        elif len(detected_years) > 1 or any(w in q_lower for w in ["trend", "between 2023 and 2025", "over the last three years", "accelerating or slowing", "consistent revenue"]):
            q_type = QueryType.TEMPORAL
            strategy = "Time-Aware Chronological Retrieval + Multi-Year Trend Alignment"
            calc = any(w in q_lower for w in ["growth", "change", "accelerat"])
            graph = False
            contra = False
            viz = False
            top_k = 8
            cost_tier = "REASONING"
            explanation = "Time-dependent historical inquiry. Chronologically aligning 2023-2025 financial facts."

        # Table Extraction
        elif any(w in q_lower for w in ["extract", "statements of operations", "balance sheet", "segment", "table", "breakdown"]):
            q_type = QueryType.TABLE_QA
            strategy = "Financial Table Intelligence + Structured DataFrame View"
            calc = False
            graph = False
            contra = False
            viz = False
            top_k = 6
            cost_tier = "FAST"
            explanation = "Tabular financial statement extraction request."

        # Causal Reasoning
        elif any(w in q_lower for w in ["why", "how did", "affect", "impact", "cause", "drivers", "factors", "risk"]):
            q_type = QueryType.REASONING
            strategy = "Knowledge Graph (Graph RAG) + Vector Semantic Retrieval"
            calc = False
            graph = True
            contra = False
            viz = False
            top_k = 8
            cost_tier = "REASONING"
            explanation = "Causal/qualitative reasoning inquiry. Triggering Graph RAG to traverse accounting links."

        # Default: Fact Retrieval
        else:
            q_type = QueryType.FACT
            strategy = "Hybrid BM25 + Vector Search with Metadata Filtering"
            calc = False
            graph = False
            contra = False
            viz = False
            top_k = 5
            cost_tier = "FAST"
            explanation = "Direct factual lookup. Applying BM25 and vector semantic search."

        return RoutingDecision(
            query=query,
            query_type=q_type,
            confidence=0.94,
            target_companies=detected_companies,
            target_years=detected_years,
            target_metrics=detected_metrics,
            retrieval_strategy=strategy,
            requires_calculator=calc,
            requires_graph=graph,
            requires_contradiction_check=contra,
            requires_visualization=viz,
            recommended_top_k=top_k,
            cost_tier=cost_tier,
            reasoning_explanation=explanation
        )
