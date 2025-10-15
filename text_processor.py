import re
from typing import List

class TextProcessor:
    """Handles text cleaning and preprocessing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.:,;?!()-]', ' ', text)
        return text.strip()

    @staticmethod
    def extract_semantic_units(text: str) -> List[str]:
        """Extract semantic units from text (sentences, paragraphs, Q&A pairs)"""
        units = []
        
        qa_pattern = r'T:\s*[^\n]+\s*J:\s*[^T]*(?=T:|$)'
        qa_matches = re.findall(qa_pattern, text, re.MULTILINE | re.DOTALL)
        
        if qa_matches and len(qa_matches) > 2:
            for qa in qa_matches:
                qa_clean = qa.strip()
                if len(qa_clean) > 50:
                    units.append(qa_clean)
        
        section_pattern = r'((?:Kategori:|BAGIAN \d+:|[0-9]+\.|#{1,3}\s+)[^\n]*(?:\n[^\n]*)*?)(?=(?:Kategori:|BAGIAN \d+:|[0-9]+\.|#{1,3}\s+)|$)'
        sections = re.findall(section_pattern, text, re.MULTILINE)
        
        for section in sections:
            section_clean = section.strip()
            if len(section_clean) > 100:
                units.append(section_clean)
        
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            para_clean = para.strip()
            if len(para_clean) > 100 and para_clean not in units:
                units.append(para_clean)
        
        if not units:
            sentences = re.split(r'[.!?]+\s+', text)
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if len(sentence_clean) > 30:
                    units.append(sentence_clean)
        
        return units

    @staticmethod
    def split_by_sentences(text: str, min_size: int, max_size: int) -> List[str]:
        """Split text by sentences while respecting size constraints"""
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = f"{current_chunk} {sentence}".strip()
            
            if len(test_chunk) <= max_size:
                current_chunk = test_chunk
            else:
                if current_chunk and len(current_chunk) >= min_size:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk and len(current_chunk) >= min_size:
            chunks.append(current_chunk)
        
        return chunks

    @staticmethod
    def intelligent_split(text: str, min_size: int, max_size: int) -> List[str]:
        """Split long text at intelligent boundaries"""
        chunks = []
        
        section_splits = re.split(r'\n(?=(?:Kategori:|BAGIAN \d+:|[0-9]+\.|#{1,3}\s+))', text)
        
        if len(section_splits) > 1:
            for section in section_splits:
                section = section.strip()
                if len(section) > max_size:
                    chunks.extend(TextProcessor.split_by_sentences(section, min_size, max_size))
                elif len(section) >= min_size:
                    chunks.append(section)
        else:
            chunks.extend(TextProcessor.split_by_sentences(text, min_size, max_size))
        
        return chunks