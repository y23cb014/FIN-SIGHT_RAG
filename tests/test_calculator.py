import pytest
from finsight.calculator import FinancialCalculator

def test_net_profit_margin():
    res = FinancialCalculator.net_profit_margin(93736.0, 391035.0)
    assert res.verified
    assert abs(res.value - 23.9713) < 0.01
    assert "23.97%" in res.formatted_value
    assert len(res.proof_steps) >= 3

def test_operating_margin():
    res = FinancialCalculator.operating_margin(109433.0, 245122.0)
    assert res.verified
    assert abs(res.value - 44.6443) < 0.01
    assert "44.64%" in res.formatted_value

def test_gross_margin():
    res = FinancialCalculator.gross_margin(180683.0, 391035.0)
    assert res.verified
    assert abs(res.value - 46.2063) < 0.01

def test_current_ratio():
    res = FinancialCalculator.current_ratio(152987.0, 176392.0)
    assert res.verified
    assert abs(res.value - 0.8673) < 0.01

def test_debt_to_equity():
    res = FinancialCalculator.debt_to_equity(49850.0, 268542.0)
    assert res.verified
    assert abs(res.value - 0.1856) < 0.01

def test_yoy_growth():
    res = FinancialCalculator.yoy_growth(574785.0, 637960.0, "Amazon Revenue")
    assert res.verified
    assert abs(res.value - 10.991) < 0.01
    assert "+10.99%" in res.formatted_value

def test_cagr():
    res = FinancialCalculator.cagr(26974.0, 130500.0, periods=2, metric_label="NVIDIA Revenue")
    assert res.verified
    assert abs(res.value - 120.0) < 1.0

def test_zero_division():
    with pytest.raises(ValueError):
        FinancialCalculator.net_profit_margin(100.0, 0.0)
