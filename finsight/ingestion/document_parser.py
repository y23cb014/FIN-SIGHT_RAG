"""
Document Parser for FinSight-RAG
Ingests PDFs, CSVs, Excel, and Markdown/TXT documents with financial metadata extraction.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import pandas as pd
from PyPDF2 import PdfReader
from finsight.config import COMPANIES, YEARS

class ParsedDocument:
    def __init__(self, 
                 filename: str, 
                 filepath: str, 
                 company: str, 
                 year: int, 
                 quarter: str, 
                 doc_type: str,
                 pages: List[Dict[str, Any]], 
                 raw_tables: Optional[List[Dict[str, Any]]] = None):
        self.filename = filename
        self.filepath = filepath
        self.company = company
        self.year = year
        self.quarter = quarter
        self.doc_type = doc_type
        self.pages = pages  # list of {"page_num": int, "text": str}
        self.raw_tables = raw_tables or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "company": self.company,
            "year": self.year,
            "quarter": self.quarter,
            "doc_type": self.doc_type,
            "total_pages": len(self.pages),
            "total_tables": len(self.raw_tables)
        }

class DocumentParser:
    """Parses heterogeneous financial files into structured document representations."""

    @classmethod
    def infer_metadata(cls, filename: str, sample_text: str = "") -> Dict[str, Any]:
        """Infers company, fiscal year, fiscal quarter, and document type from filename or header."""
        name_lower = filename.lower()
        combined = (name_lower + " " + sample_text[:1000].lower())

        # Infer Company
        company = "Unknown"
        for comp in COMPANIES:
            if comp.lower() in combined:
                company = comp
                break
        if company == "Unknown":
            if "aapl" in combined:
                company = "Apple"
            elif "msft" in combined:
                company = "Microsoft"
            elif "amzn" in combined:
                company = "Amazon"
            elif "nvda" in combined:
                company = "NVIDIA"

        # Infer Year
        year = 2024
        yr_matches = re.findall(r'\b(202[0-9])\b', filename)
        if yr_matches:
            year = int(yr_matches[0])
        else:
            text_yr = re.findall(r'\b(202[0-9])\b', sample_text[:1500])
            if text_yr:
                year = int(text_yr[0])

        # Infer Quarter
        quarter = "FY"
        for q in ["q1", "q2", "q3", "q4"]:
            if q in name_lower:
                quarter = q.upper()
                break

        # Infer Document Type
        doc_type = "Annual Report (10-K)"
        if "10-q" in name_lower or "quarterly" in name_lower or quarter != "FY":
            doc_type = "Quarterly Report (10-Q)"
        elif "presentation" in name_lower or "deck" in name_lower:
            doc_type = "Investor Presentation"
        elif "earnings" in name_lower or "transcript" in name_lower:
            doc_type = "Earnings Call Transcript"
        elif "press" in name_lower or "release" in name_lower:
            doc_type = "Press Release"
        elif "vault" in name_lower or "note" in name_lower or "thesis" in name_lower:
            doc_type = "Analyst Research Vault"
        elif "table" in name_lower or "statement" in name_lower:
            doc_type = "Financial Statements"

        return {
            "company": company,
            "year": year,
            "quarter": quarter,
            "doc_type": doc_type
        }

    @classmethod
    def parse_pdf(cls, file_path: Path) -> ParsedDocument:
        """Extracts text page-by-page from a PDF document."""
        reader = PdfReader(str(file_path))
        pages = []
        full_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page_num": i + 1, "text": text})
            if i < 3:
                full_text += text + "\n"

        meta = cls.infer_metadata(file_path.name, full_text)
        return ParsedDocument(
            filename=file_path.name,
            filepath=str(file_path),
            company=meta["company"],
            year=meta["year"],
            quarter=meta["quarter"],
            doc_type=meta["doc_type"],
            pages=pages
        )

    @classmethod
    def parse_csv_or_excel(cls, file_path: Path) -> ParsedDocument:
        """Parses CSV or Excel sheets containing financial data."""
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        meta = cls.infer_metadata(file_path.name)
        text_repr = df.to_string()
        markdown_table = df.to_markdown(index=False)

        return ParsedDocument(
            filename=file_path.name,
            filepath=str(file_path),
            company=meta["company"],
            year=meta["year"],
            quarter=meta["quarter"],
            doc_type="Financial Statements (Spreadsheet)",
            pages=[{"page_num": 1, "text": text_repr}],
            raw_tables=[{"table_df": df, "markdown": markdown_table, "page": 1}]
        )

    @classmethod
    def parse_text_or_markdown(cls, file_path: Path) -> ParsedDocument:
        """Parses text or markdown documents, splitting by headings or form feeds if available."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        meta = cls.infer_metadata(file_path.name, content[:1500])
        # Detect page markers like --- PAGE X --- or split into logical page-sized chunks
        page_splits = re.split(r'(?i)---+\s*page\s*(\d+)\s*---+', content)
        pages = []
        if len(page_splits) > 1:
            for i in range(1, len(page_splits), 2):
                p_num = int(page_splits[i])
                p_text = page_splits[i+1].strip()
                pages.append({"page_num": p_num, "text": p_text})
        else:
            pages = [{"page_num": 1, "text": content}]

        return ParsedDocument(
            filename=file_path.name,
            filepath=str(file_path),
            company=meta["company"],
            year=meta["year"],
            quarter=meta["quarter"],
            doc_type=meta["doc_type"],
            pages=pages
        )

    @classmethod
    def parse_file(cls, file_path: Path) -> ParsedDocument:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext in [".csv", ".xlsx", ".xls"]:
            return cls.parse_csv_or_excel(file_path)
        else:
            return cls.parse_text_or_markdown(file_path)
