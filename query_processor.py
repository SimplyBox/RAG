import time
from typing import List
from llama_index.llms.groq import Groq
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from groq_vision_wrapper import GroqVisionWrapper
from config import AgenticRAGConfig

class QueryProcessor:
    """Handles multi-tenant query processing and response generation"""
    
    def __init__(self, config: AgenticRAGConfig):
        """Inisialisasi kedua agen LLM dari Groq"""
        try:
            self.text_llm = Groq(
                model=config.model_name,
                api_key=config.GROQ_API_KEY,
                request_timeout=config.request_timeout
            )
            self.vision_llm = Groq(
                model=config.MAVERICK_MODEL_NAME,
                api_key=config.GROQ_API_KEY,
                request_timeout=config.request_timeout
            )
            self.vision_llm = GroqVisionWrapper(
                api_key=config.GROQ_API_KEY,
                model=config.MAVERICK_MODEL_NAME
            )
            print(f"Text agent initialized ({config.model_name})")
            print(f"Vision 'Maverick' agent initialized ({config.MAVERICK_MODEL_NAME})")
        except Exception as e:
            print(f"Error initializing LLMs: {e}")
            raise

    def analyze_image(self, image_url: str, caption: str | None) -> str:
        """Gunakan 'Maverick' (Vision LLM on Groq) untuk menganalisis gambar."""
        if not self.vision_llm:
            return "[Error: Vision agent (Maverick) not initialized.]"
            
        try:
            image_documents = load_image_urls([image_url])
            
            user_prompt = "Analyze this image concisely."
            if caption:
                user_prompt = f"User uploaded this image with the caption: '{caption}'. Analyze the image based on this context."
            
            system_prompt = """You are an AI assistant. Your task is to analyze the provided image and generate a concise, objective description. 
This description will be stored in a database as 'internal_analysis' to provide context for future text-based chats. 
Focus on key objects, themes, text (if any), and overall style.
Example: 'Analysis: A photo of a BNI bank transfer receipt for Rp 500,000.'
Example: 'Analysis: A minimalist UI/UX design portfolio for a mobile app, dark mode.'

Generate only the analysis text."""
            
            response = self.vision_llm.complete(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                image_documents=image_documents
            )
            
            return str(response).strip()
        
        except Exception as e:
            print(f"Error during image analysis: {e}")
            return f"[Error analyzing image: {e}]"

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
            response = self.text_llm.complete(prompt)
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