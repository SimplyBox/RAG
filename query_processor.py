import time
from typing import List
from llama_index.core import Settings

class QueryProcessor:
    """Handles multi-tenant query processing and response generation"""
    
    def __init__(self):
        pass

    def generate_response(self, question: str, relevant_chunks: List[str], tenant_id: str) -> str:
        """Generate response based on question and relevant chunks for specific tenant"""
        if not question.strip():
            return "Silakan ajukan pertanyaan yang valid."

        if not relevant_chunks:
            return f"Tidak ada dokumen yang ditemukan untuk tenant '{tenant_id}'. Silakan upload dokumen PDF terlebih dahulu."
        
        try:
            context_text = "\n\n---\n\n".join(relevant_chunks)
            
            prompt = f"""Anda adalah asisten AI customer service yang sangat membantu untuk perusahaan dengan ID '{tenant_id}'. 
Anda memiliki akses ke knowledge base perusahaan yang telah diproses dengan teknologi agentic chunking.

INSTRUKSI PENTING:
- HANYA gunakan informasi dari konteks knowledge base di bawah ini
- Berikan jawaban yang lengkap, terstruktur, dan mudah dipahami
- Jika ada informasi terkait di beberapa bagian konteks, gabungkan dengan logis
- Jika tidak ada informasi yang sesuai, katakan dengan jelas dan sarankan menghubungi customer service
- Jawab dalam bahasa Indonesia yang profesional dan ramah sesuai brand perusahaan
- Berikan contoh konkret jika tersedia dalam konteks
- Prioritaskan memberikan solusi yang actionable

KNOWLEDGE BASE PERUSAHAAN ({tenant_id}):
{context_text}

PERTANYAAN CUSTOMER: {question}

JAWABAN CUSTOMER SERVICE:"""
            
            time.sleep(1)
            response = Settings.llm.complete(prompt)
            response_text = str(response).strip()
            
            if len(response_text) < 20:
                return "Maaf, tidak dapat menghasilkan jawaban yang memadai. Silakan coba lagi dalam beberapa detik atau hubungi customer service langsung."
            
            return response_text
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                return "Terlalu banyak permintaan. Silakan tunggu beberapa detik dan coba lagi."
            return f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}"

    @staticmethod
    def validate_question(question: str) -> bool:
        """Validate if question is valid"""
        return bool(question and question.strip())