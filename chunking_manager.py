from typing import List
from text_processor import TextProcessor
from semantic_analyzer import SemanticAnalyzer
from llama_index.core.embeddings import BaseEmbedding

class ChunkingManager:
    """Manages the advanced chunking process with semantic optimization"""
    
    def __init__(self, base_chunk_size: int, overlap: int, 
                 min_chunk_size: int, max_chunk_size: int,
                 embed_model: BaseEmbedding | None = None):
        
        if not embed_model:
            raise ValueError("ChunkingManager requires an embedding model.")
            
        self.base_chunk_size = base_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.text_processor = TextProcessor()
        self.semantic_analyzer = SemanticAnalyzer(embed_model=embed_model)

    def optimize_chunk_size(self, group_texts: List[str]) -> List[str]:
        """Optimize chunk size based on content type and semantic coherence"""
        optimized_chunks = []
        
        for text in group_texts:
            text_len = len(text)
            
            if self.min_chunk_size <= text_len <= self.max_chunk_size:
                optimized_chunks.append(text)
            
            elif text_len > self.max_chunk_size:
                sub_chunks = self.text_processor.intelligent_split(
                    text, self.min_chunk_size, self.max_chunk_size
                )
                optimized_chunks.extend(sub_chunks)
            
            else:
                optimized_chunks.append(text)
        
        return optimized_chunks

    def merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks with semantically similar ones"""
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            if len(current_chunk) < self.min_chunk_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                combined = f"{current_chunk}\n\n{next_chunk}"
                
                if len(combined) <= self.max_chunk_size:
                    merged_chunks.append(combined)
                    i += 2
                    continue
            
            merged_chunks.append(current_chunk)
            i += 1
        
        return merged_chunks

    def chunk_text_agentic(self, text: str) -> List[str]:
        """Main agentic chunking method with optimised grouping"""
        text = self.text_processor.clean_text(text)
        
        if len(text) <= self.base_chunk_size:
            return [text] if len(text) > self.min_chunk_size else []
        
        semantic_units = self.text_processor.extract_semantic_units(text)
        
        if not semantic_units:
            return self._fallback_chunking(text)

        similarity_matrix = self.semantic_analyzer.calculate_semantic_similarity(semantic_units)
        
        groups = self.semantic_analyzer.group_related_units(semantic_units, similarity_matrix)
        
        raw_chunks = []
        for group in groups:
            group_texts = [semantic_units[i] for i in group]
            combined_text = "\n\n".join(group_texts)
            raw_chunks.append(combined_text)
        
        optimized_chunks = self.optimize_chunk_size(raw_chunks)

        final_chunks = self.merge_small_chunks(optimized_chunks)

        final_chunks = [chunk for chunk in final_chunks if len(chunk) >= self.min_chunk_size]
        
        return final_chunks if final_chunks else self._fallback_chunking(text)

    def _fallback_chunking(self, text: str) -> List[str]:
        """Fallback to simple chunking if agentic method fails"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.base_chunk_size, len(text))
            
            if end < len(text):
                last_sentence = text.rfind('.', start, end)
                if last_sentence > start + self.min_chunk_size:
                    end = last_sentence + 1
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            start = end - self.overlap if end > start + self.overlap else end
        
        return chunks