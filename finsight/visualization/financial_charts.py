"""
Financial Visualization Agent for FinSight-RAG
Generates publication-quality interactive Plotly charts for financial intelligence.
"""

from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class FinancialVisualizationAgent:
    """Creates interactive Plotly figures for trends, comparisons, margins, and segments."""

    # Professional Financial Color Palette
    THEME_COLORS = {
        "Apple": "#007AFF",
        "Microsoft": "#00A4EF",
        "Amazon": "#FF9900",
        "NVIDIA": "#76B900",
        "Revenue": "#2E7D32",
        "Net Income": "#1565C0",
        "Operating Income": "#00838F",
        "Debt": "#C62828",
        "Equity": "#2E7D32"
    }

    @classmethod
    def plot_multi_company_trend(cls, 
                                 data: List[Dict[str, Any]], 
                                 metric_name: str = "Revenue",
                                 unit: str = "$ Billions") -> go.Figure:
        """Plots multi-year trend lines for one or more companies."""
        df = pd.DataFrame(data)
        fig = go.Figure()

        if df.empty:
            return fig

        companies = df["company"].unique()
        for comp in companies:
            cdf = df[df["company"] == comp].sort_values("year")
            color = cls.THEME_COLORS.get(comp, None)
            # Display in billions if in millions
            y_vals = cdf["value"] / 1000.0 if "million" in cdf.get("unit", pd.Series(["million"])).iloc[0].lower() else cdf["value"]

            fig.add_trace(go.Scatter(
                x=cdf["year"],
                y=y_vals,
                mode="lines+markers+text",
                name=comp,
                text=[f"${v:,.1f}B" for v in y_vals],
                textposition="top center",
                line=dict(width=3, color=color),
                marker=dict(size=9)
            ))

        fig.update_layout(
            title=dict(text=f"<b>{metric_name} Trajectory (2023 - 2025)</b>", font=dict(size=18)),
            xaxis=dict(title="Fiscal Year", tickmode="linear", tick0=2023, dtick=1, gridcolor="#E0E0E0"),
            yaxis=dict(title=f"{metric_name} ({unit})", gridcolor="#E0E0E0"),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @classmethod
    def plot_margin_comparison(cls, data: List[Dict[str, Any]]) -> go.Figure:
        """Plots Gross, Operating, and Net Margins for companies."""
        df = pd.DataFrame(data)
        fig = go.Figure()
        if df.empty:
            return fig

        metrics = ["Gross Margin", "Operating Margin", "Net Margin"]
        colors = ["#4CAF50", "#2196F3", "#9C27B0"]

        for m_idx, m_name in enumerate(metrics):
            if m_name in df.columns:
                fig.add_trace(go.Bar(
                    name=m_name,
                    x=df["company"],
                    y=df[m_name],
                    text=[f"{v:.1f}%" for v in df[m_name]],
                    textposition="auto",
                    marker_color=colors[m_idx % len(colors)]
                ))

        fig.update_layout(
            barmode="group",
            title=dict(text="<b>Profitability Margins Comparison (%)</b>", font=dict(size=18)),
            xaxis=dict(title="Company"),
            yaxis=dict(title="Margin Percentage (%)", gridcolor="#E0E0E0"),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @classmethod
    def plot_segment_breakdown(cls, company: str, segments: Dict[str, float], year: int = 2024) -> go.Figure:
        """Plots revenue contribution by business segment as an interactive donut chart."""
        labels = list(segments.keys())
        values = list(segments.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.45,
            textinfo="label+percent",
            insidetextorientation="radial"
        )])

        fig.update_layout(
            title=dict(text=f"<b>{company} Business Segment Revenue Breakdown ({year})</b>", font=dict(size=18)),
            template="plotly_white",
            showlegend=True
        )
        return fig

    @classmethod
    def plot_capital_structure(cls, data: List[Dict[str, Any]]) -> go.Figure:
        """Plots Total Debt vs Stockholders Equity."""
        df = pd.DataFrame(data)
        fig = go.Figure()
        if df.empty:
            return fig

        fig.add_trace(go.Bar(
            name="Total Debt",
            x=df["company"],
            y=df["total_debt"] / 1000.0,
            marker_color="#E53935",
            text=[f"${v/1000.0:.1f}B" for v in df["total_debt"]],
            textposition="auto"
        ))
        fig.add_trace(go.Bar(
            name="Stockholders' Equity",
            x=df["company"],
            y=df["equity"] / 1000.0,
            marker_color="#43A047",
            text=[f"${v/1000.0:.1f}B" for v in df["equity"]],
            textposition="auto"
        ))

        fig.update_layout(
            barmode="group",
            title=dict(text="<b>Capital Structure & Solvency: Debt vs Equity ($B)</b>", font=dict(size=18)),
            xaxis=dict(title="Company"),
            yaxis=dict(title="Amount ($ Billions)", gridcolor="#E0E0E0"),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
