import re
from typing import List

class TextProcessor:
    """Handles text cleaning and preprocessing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.:,;?!()\n-]', ' ', text)
        return text.strip()

    @staticmethod
    def extract_semantic_units(text: str) -> List[str]:
        """Extract semantic units from text (sentences, paragraphs, Q&A pairs)"""
        units = []

        text_cleaned_for_units = re.sub(r'[^\w\s\.:,;?!()\n-]', ' ', text)
        text_cleaned_for_units = re.sub(r' +', ' ', text_cleaned_for_units).strip()

        qa_pattern = r'T:\s*[^\n]+\s*J:\s*[^T]*(?=T:|$)'
        qa_matches = re.findall(qa_pattern, text_cleaned_for_units, re.MULTILINE | re.DOTALL)
        
        if qa_matches and len(qa_matches) > 2:
            for qa in qa_matches:
                qa_clean = qa.strip()
                if len(qa_clean) > 50:
                    units.append(qa_clean)
        
        section_pattern = r'((?:Kategori:|BAGIAN \d+:|[0-9]+\.|#{1,3}\s+)[^\n]*(?:\n[^\n]*)*?)(?=(?:Kategori:|BAGIAN \d+:|[0-9]+\.|#{1,3}\s+)|$)'
        sections = re.findall(section_pattern, text_cleaned_for_units, re.MULTILINE)
        
        for section in sections:
            section_clean = section[0].strip()
            if len(section_clean) > 100:
                units.append(section_clean)
        
        paragraphs = re.split(r'\n\s*\n', text_cleaned_for_units)
        for para in paragraphs:
            para_clean = para.strip()
            if len(para_clean) > 100 and para_clean not in units:
                units.append(para_clean)
        
        if not units:
            small_units = re.split(r'[\n.!?]+\s+', text_cleaned_for_units)
            for unit in small_units:
                unit_clean = unit.strip()
                if len(unit_clean) > 30:
                    units.append(unit_clean)

        seen = set()
        final_units = []
        for unit in units:
            if unit not in seen:
                seen.add(unit)
                final_units.append(unit)
        
        return final_units

    @staticmethod
    def split_by_sentences(text: str, min_size: int, max_size: int) -> List[str]:
        """Split text by sentences OR newlines while respecting size constraints"""

        sentences = re.split(r'[.!?\n]+\s+', text)
        
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

                if len(sentence) > max_size:
                    for i in range(0, len(sentence), max_size):
                        sub_chunk = sentence[i:i + max_size].strip()
                        if len(sub_chunk) >= min_size:
                            chunks.append(sub_chunk)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk and len(current_chunk) >= min_size:
            chunks.append(current_chunk)
        
        return chunks

    @staticmethod
    def intelligent_split(text: str, min_size: int, max_size: int) -> List[str]:
        """Split long text at intelligent boundaries"""
        chunks = []
        
        section_splits = re.split(r'\n(?=(?:Kategori:|BAGIAN \d+:|[0-9]+\.|#{1,3}\s+))', text)
        
        for section in section_splits:
            section = section.strip()
            if not section:
                continue
                
            if len(section) > max_size:
                chunks.extend(TextProcessor.split_by_sentences(section, min_size, max_size))
            elif len(section) >= min_size:
                chunks.append(section)
        
        if not chunks:
            chunks.extend(TextProcessor.split_by_sentences(text, min_size, max_size))

        return chunks