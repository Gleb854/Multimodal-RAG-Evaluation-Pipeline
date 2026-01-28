from typing import List, Optional
from dataclasses import dataclass, field
import re


@dataclass
class TableChunk:
    content: str
    html_content: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    chunk_type: str = "table"


class TableProcessor:
    def __init__(self):
        pass
    
    def html_to_markdown(self, html: str) -> str:
        if not html:
            return ""
        
        html = re.sub(r'<thead[^>]*>', '', html)
        html = re.sub(r'</thead>', '', html)
        html = re.sub(r'<tbody[^>]*>', '', html)
        html = re.sub(r'</tbody>', '', html)
        
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
        if not rows:
            return self._strip_html_tags(html)
        
        markdown_rows = []
        header_processed = False
        
        for row in rows:
            cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row, re.DOTALL)
            cells = [self._strip_html_tags(cell).strip() for cell in cells]
            
            if not cells:
                continue
            
            row_str = "| " + " | ".join(cells) + " |"
            markdown_rows.append(row_str)
            
            if not header_processed:
                separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                markdown_rows.append(separator)
                header_processed = True
        
        return "\n".join(markdown_rows)
    
    def _strip_html_tags(self, text: str) -> str:
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()
    
    def process(
        self, 
        table_html: str, 
        metadata: dict = None
    ) -> TableChunk:
        metadata = metadata or {}
        
        markdown_content = self.html_to_markdown(table_html)
        
        if not markdown_content:
            markdown_content = self._strip_html_tags(table_html)
        
        table_prefix = "Table:"
        if metadata.get("table_number"):
            table_prefix = f"Table {metadata['table_number']}:"
        
        content_with_prefix = f"{table_prefix}\n{markdown_content}"
        
        return TableChunk(
            content=content_with_prefix,
            html_content=table_html,
            metadata={**metadata, "element_type": "table"},
            chunk_type="table"
        )
    
    def process_multiple(
        self, 
        tables: List[dict], 
        base_metadata: dict = None
    ) -> List[TableChunk]:
        base_metadata = base_metadata or {}
        result = []
        
        for i, table in enumerate(tables):
            html = table.get("html", table.get("content", ""))
            table_metadata = {
                **base_metadata,
                **table.get("metadata", {}),
                "table_number": i + 1,
                "table_index": i
            }
            result.append(self.process(html, table_metadata))
        
        return result
