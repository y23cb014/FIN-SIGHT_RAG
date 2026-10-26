"""
Financial Table Intelligence Layer for FinSight-RAG
Performs table detection, header normalization, financial schema mapping,
and structured fact extraction from financial tables and filings.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
from finsight.config import METRIC_ALIASES

@dataclass
class FinancialFact:
    company: str
    year: int
    quarter: str
    metric: str
    raw_label: str
    value: float
    unit: str
    currency: str
    source_doc: str
    page: int
    statement_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class TableIntelligence:
    """Detects, normalizes, and extracts structured financial facts from text and tables."""

    @staticmethod
    def clean_financial_number(raw_str: str) -> Optional[float]:
        """Parses financial numbers including parentheses for negatives, commas, and dollar signs."""
        if not raw_str:
            return None
        text = str(raw_str).strip()
        if text in ["-", "—", "N/A", "n/a", "–", ""]:
            return None
        
        # Check for parentheses: (1,234) -> -1234
        is_negative = False
        if (text.startswith("(") and text.endswith(")")) or (text.startswith("（") and text.endswith("）")):
            is_negative = True
            text = text[1:-1].strip()
        elif text.startswith("-") or text.startswith("–"):
            is_negative = True
            text = text[1:].strip()

        # Remove currency symbols and formatting
        text = re.sub(r'[\$,€£¥%]', '', text).strip()
        text = text.replace(',', '')

        try:
            val = float(text)
            return -val if is_negative else val
        except ValueError:
            return None

    @classmethod
    def match_canonical_metric(cls, line_item: str) -> Optional[str]:
        """Maps a textual line item description to a canonical financial metric."""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', line_item.lower()).strip()
        for canonical, aliases in METRIC_ALIASES.items():
            for alias in aliases:
                # exact or regex boundary match
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, cleaned):
                    return canonical
        return None

    @classmethod
    def extract_facts_from_table(cls, 
                                 df_or_lines: Any, 
                                 company: str, 
                                 source_doc: str, 
                                 page: int = 1,
                                 default_year: Optional[int] = None,
                                 statement_type: str = "Financial Statement") -> List[FinancialFact]:
        """Extracts structured financial facts from a parsed table DataFrame or Markdown/text table."""
        facts = []
        if isinstance(df_or_lines, list):
            # Parse text lines formatted as markdown or tab/pipe-delimited table
            rows = []
            for line in df_or_lines:
                if "|" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if parts and not all(set(p) <= set("-: ") for p in parts):
                        rows.append(parts)
            if len(rows) >= 2:
                headers = rows[0]
                df = pd.DataFrame(rows[1:], columns=headers[:len(rows[1])])
            else:
                return []
        elif isinstance(df_or_lines, pd.DataFrame):
            df = df_or_lines
        else:
            return []

        if df.empty:
            return []

        # Detect year columns in headers
        col_years = {}
        for col in df.columns:
            # Look for 4-digit years: 2023, 2024, 2025
            year_match = re.search(r'\b(20[2-3][0-9])\b', str(col))
            if year_match:
                col_years[col] = int(year_match.group(1))

        # First column is usually line item label
        label_col = df.columns[0]

        for _, row in df.iterrows():
            raw_label = str(row[label_col]).strip()
            canonical = cls.match_canonical_metric(raw_label)
            if not canonical:
                continue

            # Iterate over columns with recognized years
            if col_years:
                for col, yr in col_years.items():
                    val = cls.clean_financial_number(str(row[col]))
                    if val is not None:
                        facts.append(FinancialFact(
                            company=company,
                            year=yr,
                            quarter="FY",
                            metric=canonical,
                            raw_label=raw_label,
                            value=val,
                            unit="million USD",
                            currency="USD",
                            source_doc=source_doc,
                            page=page,
                            statement_type=statement_type
                        ))
            elif default_year is not None:
                # If single year or fallback
                for col in df.columns[1:]:
                    val = cls.clean_financial_number(str(row[col]))
                    if val is not None:
                        facts.append(FinancialFact(
                            company=company,
                            year=default_year,
                            quarter="FY",
                            metric=canonical,
                            raw_label=raw_label,
                            value=val,
                            unit="million USD",
                            currency="USD",
                            source_doc=source_doc,
                            page=page,
                            statement_type=statement_type
                        ))
                        break

        return facts

    @classmethod
    def extract_facts_from_text(cls, text: str, company: str, year: int, source_doc: str, page: int = 1) -> List[FinancialFact]:
        """Extracts financial facts from prose sentences (e.g. 'net sales were $383,285 million')."""
        facts = []
        # Pattern for: metric ... [was/were/of/at] $XXX million
        pattern = r'(?i)\b(revenue|net sales|net income|operating income|gross profit|total debt|gross margin|services revenue)\b[^\$\n\d]{1,40}\$\s*([0-9,]+(?:\.[0-9]+)?)\s*(million|billion)?'
        for m in re.finditer(pattern, text):
            raw_metric = m.group(1).strip()
            num_str = m.group(2).strip()
            scale = m.group(3) or "million"
            canonical = cls.match_canonical_metric(raw_metric)
            if not canonical:
                continue
            val = cls.clean_financial_number(num_str)
            if val is not None:
                if scale.lower() == "billion":
                    val *= 1000.0
                facts.append(FinancialFact(
                    company=company,
                    year=year,
                    quarter="FY",
                    metric=canonical,
                    raw_label=raw_metric,
                    value=val,
                    unit="million USD",
                    currency="USD",
                    source_doc=source_doc,
                    page=page,
                    statement_type="Text Statement / Press Release"
                ))
        return facts

