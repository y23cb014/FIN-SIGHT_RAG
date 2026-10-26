"""
Financial Calculation Engine for FinSight-RAG
Executes verified, deterministic calculations of financial ratios and growth metrics.
Prevents LLM numerical hallucinations by evaluating mathematical operations deterministically.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import math

@dataclass
class CalculationResult:
    metric_name: str
    formula: str
    inputs: Dict[str, Any]
    value: float
    formatted_value: str
    unit: str
    verified: bool
    proof_steps: List[str]
    citations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "formula": self.formula,
            "inputs": self.inputs,
            "value": self.value,
            "formatted_value": self.formatted_value,
            "unit": self.unit,
            "verified": self.verified,
            "proof_steps": self.proof_steps,
            "citations": self.citations
        }

class FinancialCalculator:
    """Deterministic, zero-hallucination financial calculation engine."""

    @staticmethod
    def net_profit_margin(net_income: float, revenue: float, citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if revenue == 0:
            raise ValueError("Revenue cannot be zero when calculating Net Profit Margin.")
        val = (net_income / revenue) * 100.0
        formula = "Net Margin = (Net Income / Revenue) * 100"
        proof = [
            f"Net Income: ${net_income:,.2f} M",
            f"Revenue: ${revenue:,.2f} M",
            f"Calculation: ({net_income:,.2f} / {revenue:,.2f}) * 100 = {val:.2f}%"
        ]
        return CalculationResult(
            metric_name="Net Profit Margin",
            formula=formula,
            inputs={"net_income": net_income, "revenue": revenue},
            value=round(val, 4),
            formatted_value=f"{val:.2f}%",
            unit="%",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )

    @staticmethod
    def operating_margin(operating_income: float, revenue: float, citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if revenue == 0:
            raise ValueError("Revenue cannot be zero when calculating Operating Margin.")
        val = (operating_income / revenue) * 100.0
        formula = "Operating Margin = (Operating Income / Revenue) * 100"
        proof = [
            f"Operating Income: ${operating_income:,.2f} M",
            f"Revenue: ${revenue:,.2f} M",
            f"Calculation: ({operating_income:,.2f} / {revenue:,.2f}) * 100 = {val:.2f}%"
        ]
        return CalculationResult(
            metric_name="Operating Margin",
            formula=formula,
            inputs={"operating_income": operating_income, "revenue": revenue},
            value=round(val, 4),
            formatted_value=f"{val:.2f}%",
            unit="%",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )

    @staticmethod
    def gross_margin(gross_profit: float, revenue: float, citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if revenue == 0:
            raise ValueError("Revenue cannot be zero when calculating Gross Margin.")
        val = (gross_profit / revenue) * 100.0
        formula = "Gross Margin = (Gross Profit / Revenue) * 100"
        proof = [
            f"Gross Profit: ${gross_profit:,.2f} M",
            f"Revenue: ${revenue:,.2f} M",
            f"Calculation: ({gross_profit:,.2f} / {revenue:,.2f}) * 100 = {val:.2f}%"
        ]
        return CalculationResult(
            metric_name="Gross Margin",
            formula=formula,
            inputs={"gross_profit": gross_profit, "revenue": revenue},
            value=round(val, 4),
            formatted_value=f"{val:.2f}%",
            unit="%",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )

    @staticmethod
    def current_ratio(current_assets: float, current_liabilities: float, citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if current_liabilities == 0:
            raise ValueError("Current liabilities cannot be zero.")
        val = current_assets / current_liabilities
        formula = "Current Ratio = Current Assets / Current Liabilities"
        proof = [
            f"Current Assets: ${current_assets:,.2f} M",
            f"Current Liabilities: ${current_liabilities:,.2f} M",
            f"Calculation: {current_assets:,.2f} / {current_liabilities:,.2f} = {val:.2f}x"
        ]
        return CalculationResult(
            metric_name="Current Ratio",
            formula=formula,
            inputs={"current_assets": current_assets, "current_liabilities": current_liabilities},
            value=round(val, 4),
            formatted_value=f"{val:.2f}x",
            unit="ratio",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )

    @staticmethod
    def quick_ratio(current_assets: float, inventory: float, current_liabilities: float, citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if current_liabilities == 0:
            raise ValueError("Current liabilities cannot be zero.")
        val = (current_assets - inventory) / current_liabilities
        formula = "Quick Ratio = (Current Assets - Inventory) / Current Liabilities"
        proof = [
            f"Current Assets: ${current_assets:,.2f} M",
            f"Inventory: ${inventory:,.2f} M",
            f"Current Liabilities: ${current_liabilities:,.2f} M",
            f"Calculation: ({current_assets:,.2f} - {inventory:,.2f}) / {current_liabilities:,.2f} = {val:.2f}x"
        ]
        return CalculationResult(
            metric_name="Quick Ratio",
            formula=formula,
            inputs={"current_assets": current_assets, "inventory": inventory, "current_liabilities": current_liabilities},
            value=round(val, 4),
            formatted_value=f"{val:.2f}x",
            unit="ratio",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )

    @staticmethod
    def debt_to_equity(total_debt: float, equity: float, citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if equity == 0:
            raise ValueError("Stockholders' equity cannot be zero.")
        val = total_debt / equity
        formula = "Debt-to-Equity = Total Debt / Stockholders' Equity"
        proof = [
            f"Total Debt: ${total_debt:,.2f} M",
            f"Stockholders' Equity: ${equity:,.2f} M",
            f"Calculation: {total_debt:,.2f} / {equity:,.2f} = {val:.2f}x"
        ]
        return CalculationResult(
            metric_name="Debt-to-Equity Ratio",
            formula=formula,
            inputs={"total_debt": total_debt, "equity": equity},
            value=round(val, 4),
            formatted_value=f"{val:.2f}x",
            unit="ratio",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )

    @staticmethod
    def roe(net_income: float, equity: float, citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if equity == 0:
            raise ValueError("Equity cannot be zero.")
        val = (net_income / equity) * 100.0
        formula = "ROE = (Net Income / Stockholders' Equity) * 100"
        proof = [
            f"Net Income: ${net_income:,.2f} M",
            f"Stockholders' Equity: ${equity:,.2f} M",
            f"Calculation: ({net_income:,.2f} / {equity:,.2f}) * 100 = {val:.2f}%"
        ]
        return CalculationResult(
            metric_name="Return on Equity (ROE)",
            formula=formula,
            inputs={"net_income": net_income, "equity": equity},
            value=round(val, 4),
            formatted_value=f"{val:.2f}%",
            unit="%",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )

    @staticmethod
    def yoy_growth(initial_val: float, final_val: float, metric_label: str = "Metric", citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if initial_val == 0:
            raise ValueError(f"Base year value for {metric_label} cannot be zero.")
        growth = ((final_val - initial_val) / initial_val) * 100.0
        sign = "+" if growth > 0 else ""
        formula = f"YoY Growth = ((Value_t - Value_t-1) / Value_t-1) * 100"
        proof = [
            f"Base Year {metric_label}: ${initial_val:,.2f} M",
            f"Target Year {metric_label}: ${final_val:,.2f} M",
            f"Absolute Change: ${final_val - initial_val:,.2f} M",
            f"Calculation: (({final_val:,.2f} - {initial_val:,.2f}) / {initial_val:,.2f}) * 100 = {sign}{growth:.2f}%"
        ]
        return CalculationResult(
            metric_name=f"{metric_label} YoY Growth",
            formula=formula,
            inputs={"initial_val": initial_val, "final_val": final_val},
            value=round(growth, 4),
            formatted_value=f"{sign}{growth:.2f}%",
            unit="%",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )

    @staticmethod
    def cagr(start_val: float, end_val: float, periods: int, metric_label: str = "Metric", citations: Optional[List[Dict[str, Any]]] = None) -> CalculationResult:
        if start_val <= 0:
            raise ValueError("Starting value must be greater than zero for CAGR.")
        if periods <= 0:
            raise ValueError("Number of periods must be positive.")
        cagr_val = ((end_val / start_val) ** (1.0 / periods) - 1.0) * 100.0
        sign = "+" if cagr_val > 0 else ""
        formula = "CAGR = ((Ending Value / Beginning Value) ^ (1 / n)) - 1"
        proof = [
            f"Beginning Value: ${start_val:,.2f} M",
            f"Ending Value: ${end_val:,.2f} M",
            f"Number of Periods (n): {periods}",
            f"Calculation: (({end_val:,.2f} / {start_val:,.2f}) ^ (1 / {periods}) - 1) * 100 = {sign}{cagr_val:.2f}%"
        ]
        return CalculationResult(
            metric_name=f"{metric_label} CAGR ({periods}Y)",
            formula=formula,
            inputs={"start_val": start_val, "end_val": end_val, "periods": periods},
            value=round(cagr_val, 4),
            formatted_value=f"{sign}{cagr_val:.2f}%",
            unit="%",
            verified=True,
            proof_steps=proof,
            citations=citations or []
        )
