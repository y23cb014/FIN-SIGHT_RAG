import pytest
from finsight.evaluation.metrics import FinancialMetrics
from finsight.evaluation.ablation_engine import AblationEngine

def test_recall_and_hits():
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    ground_truth = ["doc_B", "doc_F"]
    
    assert FinancialMetrics.hit_at_k(retrieved, ground_truth, k=1) == 0.0
    assert FinancialMetrics.hit_at_k(retrieved, ground_truth, k=5) == 1.0
    assert FinancialMetrics.recall_at_k(retrieved, ground_truth, k=5) == 0.5
    assert FinancialMetrics.mrr(retrieved, ground_truth) == 0.5

def test_numerical_accuracy():
    assert FinancialMetrics.numerical_accuracy(100.5, 100.0) == 1.0
    assert FinancialMetrics.numerical_accuracy(120.0, 100.0) == 0.0
    assert FinancialMetrics.numerical_accuracy(None, 100.0) == 0.0

def test_ablation_benchmarks():
    df = AblationEngine.get_ablation_benchmarks()
    assert len(df) == 8
    assert "Accuracy (%)" in df.columns
    # Check monotonic accuracy progression
    accs = df["Accuracy (%)"].tolist()
    assert accs[-1] > accs[0]
