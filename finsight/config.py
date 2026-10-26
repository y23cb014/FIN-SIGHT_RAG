import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SAMPLE_DOCS_DIR = DATA_DIR / 'sample_documents'
VAULT_DIR = DATA_DIR / 'vault'
EVAL_DATASET_PATH = DATA_DIR / 'evaluation_dataset.json'

# Financial Ontology / Canonical Metric Mappings
METRIC_ALIASES = {
    'services_revenue': ['services revenue', 'services net sales', 'services category'],
    'data_center': ['data center revenue', 'data center segment', 'data center'],
    'aws': ['aws segment sales', 'aws net sales', 'aws revenue'],
    'revenue': ['total net sales', 'total revenue', 'net sales', 'operating revenue', 'revenue', 'sales'],
    'gross_profit': ['gross profit', 'gross margin (dollars)', 'total gross profit'],
    'operating_income': ['operating income', 'operating profit', 'income from operations', 'ebit'],
    'net_income': ['net income', 'net earnings', 'net profit', 'net income attributable to shareholders'],
    'total_debt': ['total debt', 'long-term debt', 'short-term debt and long-term debt', 'total liabilities debt', 'debt'],
    'equity': ["stockholders' equity", "shareholders' equity", 'total equity', 'total stockholders equity'],
    'cash': ['cash and cash equivalents', 'cash', 'cash, cash equivalents and marketable securities', 'total cash'],
    'current_assets': ['total current assets', 'current assets'],
    'current_liabilities': ['total current liabilities', 'current liabilities'],
    'operating_cash_flow': ['cash provided by operating activities', 'cash generated from operations', 'operating cash flow', 'net cash provided by operating activities'],
    'capex': ['capital expenditures', 'payments for property, plant and equipment', 'purchase of property and equipment'],
    'ebitda': ['ebitda', 'operating income before depreciation', 'adjusted ebitda']
}

# Supported Benchmark Companies
COMPANIES = ['Apple', 'Microsoft', 'Amazon', 'NVIDIA']
YEARS = [2023, 2024, 2025]

# Confidence Scoring Weights
CONFIDENCE_WEIGHTS = {
    'source_agreement': 0.25,
    'period_alignment': 0.25,
    'numerical_grounding': 0.25,
    'calculation_verification': 0.25
}

# Retrieval Defaults
DEFAULT_TOP_K = 5
RRF_K = 60

# Model Routing Configuration
ROUTER_CONFIG = {
    'fast_model': os.getenv('FINSIGHT_FAST_MODEL', 'gemini-2.5-flash'),
    'reasoning_model': os.getenv('FINSIGHT_REASONING_MODEL', 'gemini-2.5-pro'),
    'gemini_api_key': os.getenv('GEMINI_API_KEY', '')
}
