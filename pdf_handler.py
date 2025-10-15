import os
from typing import List
from llama_index.readers.file import PDFReader

class PDFHandler:
    """Handles PDF reading and text extraction"""
    
    def __init__(self):
        self.reader = PDFReader()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        documents = self.reader.load_data(file=pdf_path)

        if not documents:
            raise ValueError("No content found in PDF")

        full_text = ""
        for doc in documents:
            if doc.text and doc.text.strip():
                full_text += " " + doc.text.strip()
        
        if not full_text.strip():
            raise ValueError("No readable text found in PDF")

        return full_text.strip()

    @staticmethod
    def validate_pdf_path(pdf_path: str) -> bool:
        """Validate if PDF path exists and is accessible"""
        return os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf')