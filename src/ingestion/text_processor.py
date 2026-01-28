from typing import List
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class TextChunk:
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_type: str = "text"


class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process(self, text: str, metadata: dict = None) -> List[TextChunk]:
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = self.splitter.split_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            result.append(TextChunk(
                content=chunk.strip(),
                metadata=chunk_metadata,
                chunk_type="text"
            ))
        
        return result
    
    def merge_title_with_content(
        self, 
        title: str, 
        content: str, 
        metadata: dict = None
    ) -> List[TextChunk]:
        if title:
            combined = f"{title}\n\n{content}"
        else:
            combined = content
        return self.process(combined, metadata)
