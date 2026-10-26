"""
Financial Chunking Engine for FinSight-RAG
Creates semantically enriched text chunks and table-aware chunks with deep financial metadata.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
from finsight.ingestion.document_parser import ParsedDocument
from finsight.ingestion.table_intelligence import TableIntelligence, FinancialFact

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    company: str
    year: int
    quarter: str
    doc_type: str
    page: int
    source_file: str
    is_table: bool
    structured_facts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "company": self.company,
            "year": self.year,
            "quarter": self.quarter,
            "doc_type": self.doc_type,
            "page": self.page,
            "source_file": self.source_file,
            "is_table": self.is_table,
            "structured_facts": self.structured_facts
        }

class FinancialChunker:
    """Splits documents into coherent financial chunks, preserving tables and section context."""

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        chunks = []
        chunk_idx = 0

        # 1. Process explicit tables if available
        for t_idx, table_info in enumerate(doc.raw_tables):
            df = table_info.get("table_df")
            page_num = table_info.get("page", 1)
            md_text = table_info.get("markdown", "")
            facts = TableIntelligence.extract_facts_from_table(
                df, company=doc.company, source_doc=doc.filename, page=page_num, default_year=doc.year
            )
            fact_dicts = [f.to_dict() for f in facts]
            c_id = f"{doc.company}_{doc.year}_p{page_num}_tbl{t_idx}"
            table_chunk_text = f"[Financial Table - {doc.company} {doc.year} (Page {page_num})]\n{md_text}"
            chunks.append(DocumentChunk(
                chunk_id=c_id,
                text=table_chunk_text,
                company=doc.company,
                year=doc.year,
                quarter=doc.quarter,
                doc_type=doc.doc_type,
                page=page_num,
                source_file=doc.filename,
                is_table=True,
                structured_facts=fact_dicts
            ))

        # 2. Process regular pages
        for p in doc.pages:
            page_num = p.get("page_num", 1)
            text = p.get("text", "").strip()
            if not text:
                continue

            # Detect embedded markdown tables or pipe-delimited structures in page text
            lines = text.split("\n")
            table_buffer = []
            prose_buffer = []

            for line in lines:
                if "|" in line:
                    table_buffer.append(line)
                else:
                    if table_buffer:
                        # Process buffered table
                        if len(table_buffer) >= 2:
                            tbl_facts = TableIntelligence.extract_facts_from_table(
                                table_buffer, company=doc.company, source_doc=doc.filename, page=page_num, default_year=doc.year
                            )
                            c_id = f"{doc.company}_{doc.year}_p{page_num}_c{chunk_idx}"
                            chunk_idx += 1
                            chunks.append(DocumentChunk(
                                chunk_id=c_id,
                                text="\n".join(table_buffer),
                                company=doc.company,
                                year=doc.year,
                                quarter=doc.quarter,
                                doc_type=doc.doc_type,
                                page=page_num,
                                source_file=doc.filename,
                                is_table=True,
                                structured_facts=[f.to_dict() for f in tbl_facts]
                            ))
                        else:
                            prose_buffer.extend(table_buffer)
                        table_buffer = []
                    prose_buffer.append(line)

            if table_buffer and len(table_buffer) >= 2:
                tbl_facts = TableIntelligence.extract_facts_from_table(
                    table_buffer, company=doc.company, source_doc=doc.filename, page=page_num, default_year=doc.year
                )
                c_id = f"{doc.company}_{doc.year}_p{page_num}_c{chunk_idx}"
                chunk_idx += 1
                chunks.append(DocumentChunk(
                    chunk_id=c_id,
                    text="\n".join(table_buffer),
                    company=doc.company,
                    year=doc.year,
                    quarter=doc.quarter,
                    doc_type=doc.doc_type,
                    page=page_num,
                    source_file=doc.filename,
                    is_table=True,
                    structured_facts=[f.to_dict() for f in tbl_facts]
                ))

            # Process prose text with sliding window
            prose_text = "\n".join(prose_buffer).strip()
            if prose_text:
                prose_chunks = self._sliding_window_chunks(prose_text)
                for pc in prose_chunks:
                    c_id = f"{doc.company}_{doc.year}_p{page_num}_c{chunk_idx}"
                    chunk_idx += 1
                    p_facts = TableIntelligence.extract_facts_from_text(
                        text=pc, company=doc.company, year=doc.year, source_doc=doc.filename, page=page_num
                    )
                    chunks.append(DocumentChunk(
                        chunk_id=c_id,
                        text=pc,
                        company=doc.company,
                        year=doc.year,
                        quarter=doc.quarter,
                        doc_type=doc.doc_type,
                        page=page_num,
                        source_file=doc.filename,
                        is_table=False,
                        structured_facts=[f.to_dict() for f in p_facts]
                    ))

        return chunks

    def _sliding_window_chunks(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= self.chunk_size:
                current = f"{current}\n\n{para}".strip()
            else:
                if current:
                    chunks.append(current)
                if len(para) > self.chunk_size:
                    # Break long paragraph
                    words = para.split()
                    sub_chunk = ""
                    for w in words:
                        if len(sub_chunk) + len(w) + 1 <= self.chunk_size:
                            sub_chunk = f"{sub_chunk} {w}".strip()
                        else:
                            chunks.append(sub_chunk)
                            sub_chunk = w
                    if sub_chunk:
                        current = sub_chunk
                else:
                    current = para
        if current:
            chunks.append(current)
        return chunks
