"""
Financial Knowledge Graph Layer (Graph RAG)
Models hierarchical and causal relationships between companies, reports, metrics, and economic concepts.
Enables multi-hop semantic and causal reasoning (e.g. Debt -> Interest Expense -> Net Income -> ROE).
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
from finsight.ingestion.table_intelligence import FinancialFact

class FinancialKnowledgeGraph:
    """Property graph for financial entities, metrics, reports, and causal mechanisms."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._initialize_financial_domain_ontology()

    def _initialize_financial_domain_ontology(self):
        """Initializes core accounting mechanics and causal relationships."""
        # Core accounting concepts & causal chains
        causal_edges = [
            ("Debt", "Interest_Expense", "increases_borrowing_cost"),
            ("Interest_Expense", "Net_Income", "reduces_earnings"),
            ("Net_Income", "ROE", "drives_shareholder_return"),
            ("Net_Income", "Net_Profit_Margin", "determines_margin"),
            ("Revenue", "Gross_Profit", "topline_flow_through"),
            ("Gross_Profit", "Operating_Income", "core_business_flow"),
            ("Operating_Income", "Net_Income", "pretax_flow"),
            ("Revenue", "Operating_Margin", "scale_leverage"),
            ("Current_Assets", "Current_Ratio", "liquidity_buffer"),
            ("Current_Liabilities", "Current_Ratio", "short_term_burden"),
            ("Total_Debt", "Debt_to_Equity", "financial_leverage"),
            ("Stockholders_Equity", "Debt_to_Equity", "solvency_base"),
            ("Operating_Cash_Flow", "Free_Cash_Flow", "cash_generation"),
            ("Capex", "Free_Cash_Flow", "capital_reinvestment")
        ]
        for src, dst, rel in causal_edges:
            self.graph.add_node(src, type="Concept", label=src.replace("_", " "))
            self.graph.add_node(dst, type="Concept", label=dst.replace("_", " "))
            self.graph.add_edge(src, dst, relation="INFLUENCES", mechanism=rel)

        # Categorical nodes
        categories = {
            "Profitability": ["Gross_Profit", "Operating_Income", "Net_Income", "Net_Profit_Margin", "ROE"],
            "Liquidity": ["Current_Assets", "Current_Liabilities", "Current_Ratio"],
            "Leverage": ["Debt", "Total_Debt", "Debt_to_Equity"],
            "CashFlow": ["Operating_Cash_Flow", "Capex", "Free_Cash_Flow"]
        }
        for cat, metrics in categories.items():
            self.graph.add_node(cat, type="Category", label=cat)
            for m in metrics:
                self.graph.add_edge(m, cat, relation="BELONGS_TO_CATEGORY")

    def add_fact(self, fact: FinancialFact):
        """Integrates an extracted financial fact into the Knowledge Graph."""
        comp_node = f"Company:{fact.company}"
        year_node = f"Year:{fact.year}"
        report_node = f"Report:{fact.source_doc}"
        metric_name = fact.metric.upper()
        metric_node = f"Metric:{metric_name}"
        fact_node = f"Fact:{fact.company}_{fact.year}_{metric_name}"

        # Add Nodes with attributes
        self.graph.add_node(comp_node, type="Company", name=fact.company)
        self.graph.add_node(year_node, type="FiscalYear", year=fact.year)
        self.graph.add_node(report_node, type="Report", filename=fact.source_doc, page=fact.page)
        self.graph.add_node(metric_node, type="Metric", name=metric_name)
        self.graph.add_node(fact_node, type="FactValue", 
                            value=fact.value, unit=fact.unit, raw_label=fact.raw_label, page=fact.page)

        # Connect Edges
        self.graph.add_edge(comp_node, report_node, relation="HAS_REPORT")
        self.graph.add_edge(report_node, year_node, relation="FOR_YEAR")
        self.graph.add_edge(report_node, fact_node, relation="CONTAINS_FACT", page=fact.page)
        self.graph.add_edge(fact_node, metric_node, relation="MEASURES_METRIC")
        self.graph.add_edge(comp_node, fact_node, relation="REPORTS", year=fact.year)

        # Link to conceptual ontology if available
        norm_metric = fact.metric.replace(" ", "_").title()
        if norm_metric in self.graph:
            self.graph.add_edge(fact_node, norm_metric, relation="INSTANTIATES")

    def get_company_metrics(self, company: str, year: Optional[int] = None) -> Dict[str, float]:
        """Retrieves structured metric values for a given company and optional fiscal year."""
        metrics = {}
        comp_node = f"Company:{company}"
        if comp_node not in self.graph:
            return metrics

        for _, fact_node, data in self.graph.out_edges(comp_node, data=True):
            if data.get("relation") == "REPORTS":
                fact_yr = data.get("year")
                if year is None or fact_yr == year:
                    f_data = self.graph.nodes[fact_node]
                    # Find metric measured
                    for _, m_node, m_edge in self.graph.out_edges(fact_node, data=True):
                        if m_edge.get("relation") == "MEASURES_METRIC":
                            m_name = self.graph.nodes[m_node].get("name", "").lower()
                            metrics[m_name] = f_data.get("value", 0.0)
        return metrics

    def find_causal_chain(self, start_metric: str, end_metric: str) -> List[str]:
        """Finds directed causal paths between financial metrics or concepts with case-insensitive lookup."""
        # Find matching node names ignoring case and underscores
        def match_node(name: str) -> Optional[str]:
            clean = name.replace(" ", "").replace("_", "").lower()
            for n in self.graph.nodes:
                if n.replace(" ", "").replace("_", "").lower() == clean:
                    return n
            return None

        src_node = match_node(start_metric)
        dst_node = match_node(end_metric)

        if src_node and dst_node:
            try:
                paths = list(nx.all_simple_paths(self.graph, source=src_node, target=dst_node, cutoff=6))
                if paths:
                    best_path = paths[0]
                    chain_str = " -> ".join(best_path)
                    return [chain_str]
            except Exception:
                pass
        return []

    def retrieve_graph_context(self, query: str, company: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves relevant entity triples and causal chains for RAG grounding."""
        triples = []
        q_lower = query.lower()

        # Check for causal terms
        if ("debt" in q_lower and ("profit" in q_lower or "roe" in q_lower or "margin" in q_lower)):
            chain = self.find_causal_chain("Debt", "ROE") or self.find_causal_chain("Debt", "Net_Income")
            if chain:
                triples.append({
                    "subject": "Debt",
                    "predicate": "CAUSAL_PATH_TO",
                    "object": "ROE/Profitability",
                    "explanation": f"Structural financial link: {chain[0]}"
                })

        # Check for company-specific facts
        target_comps = [company] if company else ["Apple", "Microsoft", "Amazon", "NVIDIA"]
        for comp in target_comps:
            if comp and comp.lower() in q_lower:
                for yr in [2023, 2024, 2025]:
                    if str(yr) in q_lower or "trend" in q_lower or "compare" in q_lower:
                        comp_metrics = self.get_company_metrics(comp, yr)
                        for m_name, val in comp_metrics.items():
                            if m_name in q_lower or "revenue" in q_lower or "income" in q_lower or "margin" in q_lower:
                                triples.append({
                                    "subject": f"{comp} ({yr})",
                                    "predicate": f"has_{m_name}",
                                    "object": f"${val:,.2f} M",
                                    "source": "Knowledge Graph"
                                })

        return triples

    def export_subgraph_for_visualization(self, max_nodes: int = 40) -> Dict[str, Any]:
        """Exports nodes and links for Plotly / Streamlit network visualization."""
        nodes = []
        links = []
        selected_nodes = set(list(self.graph.nodes)[:max_nodes])

        for n in selected_nodes:
            ndata = self.graph.nodes[n]
            ntype = ndata.get("type", "Entity")
            nodes.append({
                "id": n,
                "label": ndata.get("label", n.split(":")[-1]),
                "type": ntype
            })

        for u, v, data in self.graph.edges(data=True):
            if u in selected_nodes and v in selected_nodes:
                links.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", "LINKED")
                })

        return {"nodes": nodes, "links": links}
