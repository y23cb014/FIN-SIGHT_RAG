import pytest
from finsight.router import AdaptiveQueryRouter, QueryType

def test_fact_routing():
    d = AdaptiveQueryRouter.analyze_query("What was Apple total net sales in 2024?")
    assert d.query_type == QueryType.FACT
    assert "Apple" in d.target_companies
    assert 2024 in d.target_years

def test_numerical_routing():
    d = AdaptiveQueryRouter.analyze_query("Calculate Apple net profit margin for 2024")
    assert d.query_type == QueryType.NUMERICAL
    assert d.requires_calculator

def test_multi_doc_routing():
    d = AdaptiveQueryRouter.analyze_query("Compare the total revenue of Apple, Microsoft, Amazon, and NVIDIA in 2024")
    assert d.query_type == QueryType.MULTI_DOC
    assert len(d.target_companies) == 4

def test_temporal_routing():
    d = AdaptiveQueryRouter.analyze_query("How did Apple revenue change between 2023, 2024, and 2025?")
    assert d.query_type == QueryType.TEMPORAL
    assert len(d.target_years) == 3

def test_contradiction_routing():
    d = AdaptiveQueryRouter.analyze_query("I found conflicting values for revenue, identify conflicting figures")
    assert d.query_type == QueryType.CONTRADICTION
    assert d.requires_contradiction_check

def test_visualization_routing():
    d = AdaptiveQueryRouter.analyze_query("Plot the revenue trend of Apple and Microsoft from 2023 to 2025")
    assert d.query_type == QueryType.VISUALIZATION
    assert d.requires_visualization
