import pytest
from finsight.pipeline import FinSightPipeline

@pytest.fixture(scope="module")
def pipeline():
    return FinSightPipeline()

def test_pipeline_basic_query(pipeline):
    res = pipeline.run("What was Apple total net sales in 2024?")
    assert res["query"] is not None
    assert len(res["citations"]) > 0
    assert res["confidence"]["overall_score"] > 60.0
    assert len(res["audit_trail"]["steps"]) == 6

def test_pipeline_calculation(pipeline):
    res = pipeline.run("Calculate Apple net profit margin for 2024")
    assert res["calculation"] is not None
    assert res["calculation"]["verified"]
    assert "23.97%" in res["calculation"]["formatted_value"]

def test_pipeline_multi_company(pipeline):
    res = pipeline.run("Compare the total revenue of Apple, Microsoft, Amazon, and NVIDIA in 2024")
    assert res["routing"]["query_type"] == "Multi-Document Cross-Company"
    assert "Amazon" in res["answer"]
    assert "NVIDIA" in res["answer"]
