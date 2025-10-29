import time
from typing import List, Optional
from llama_index.llms.groq import Groq
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
            self.vision_llm = GroqVisionWrapper(
                api_key=config.GROQ_API_KEY,
                model=config.MAVERICK_MODEL_NAME
            )
            print(f"Text agent initialized ({config.model_name})")
            print(f"Vision 'Maverick' agent initialized ({config.MAVERICK_MODEL_NAME})")
        except Exception as e:
            print(f"Error initializing LLMs: {e}")
            raise

    def analyze_image(self, image_url: str | None, caption: str | None, local_image_path: str | None = None) -> str:
        """Gunakan 'Maverick' (Vision LLM on Groq) untuk menganalisis gambar."""
        if not self.vision_llm:
            return "[Error: Vision agent (Maverick) not initialized.]"
        
        if not image_url and not local_image_path:
            return "[Error: No image URL or local path provided for analysis.]"
            
        try:
            user_prompt = "Transcribe all text from this image. If it is a menu, list all items and prices."
            if caption:
                user_prompt = f"User uploaded this image with the caption: '{caption}'. Prioritize transcribing all text (like menu items, prices) from the image."
            
            system_prompt = """You are a powerful AI assistant with Optical Character Recognition (OCR) capabilities. 
Your PRIMARY task is to meticulously transcribe ALL text from the provided image, verbatim.

INSTRUCTIONS:
1.  If the image is a menu, document, screenshot, or receipt, transcribe all text content. Preserve the structure, categories, and prices exactly as seen.
2.  If the image has NO text (e.g., a photograph of a landscape), THEN and ONLY THEN, provide a concise visual description.
3.  Present the transcribed text clearly.

Example (for a menu):
Analysis: 
[TRANSCRIPTION START]
Makanan
Mie Goreng 15K
Nasi Goreng 17K
Mie Seafood 16K
...
Minuman
Es Jeruk 15K
...
[TRANSCRIPTION END]

Example (for a photo with no text):
Analysis: A photo of a golden retriever playing in a park with a red ball.
"""

            response = self.vision_llm.complete(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                image_url=image_url,
                local_image_path=local_image_path
            )
            
            return str(response).strip()
        
        except Exception as e:
            print(f"Error during image analysis: {e}")
            return f"[Error analyzing image: {e}]"

    def generate_response(self, question: str, relevant_chunks: List[str], tenant_id: str, persona_prompt: Optional[str] = None) -> str:
        """Generate response based on question and relevant chunks for specific tenant"""
        if not question.strip():
            return "Silakan ajukan pertanyaan yang valid."

        if not relevant_chunks:
            return f"Tidak ada dokumen yang ditemukan untuk tenant '{tenant_id}'. Silakan upload dokumen PDF terlebih dahulu."
        
        try:
            context_text = "\n\n---\n\n".join(relevant_chunks)

            custom_rules = persona_prompt
            if not custom_rules or not custom_rules.strip():
                custom_rules = "Tidak ada aturan persona khusus yang ditetapkan oleh perusahaan. Jawab dengan profesional."
            
            prompt = f"""Anda adalah asisten AI customer service yang sangat membantu untuk perusahaan dengan ID '{tenant_id}'. 
Anda memiliki akses ke knowledge base perusahaan yang telah diproses dengan teknologi agentic chunking.

INSTRUKSI PENTING (TERKUNCI):
- HANYA gunakan informasi dari konteks knowledge base di bawah ini
- Berikan jawaban yang lengkap, terstruktur, dan mudah dipahami
- Jika ada informasi terkait di beberapa bagian konteks, gabungkan dengan logis
- Jika tidak ada informasi yang sesuai, katakan dengan jelas dan sarankan menghubungi customer service
- Jawab dalam bahasa Indonesia yang profesional dan ramah sesuai brand perusahaan
- Berikan contoh konkret jika tersedia dalam konteks
- Prioritaskan memberikan solusi yang actionable

INSTRUKSI KHUSUS UNTUK GAMBAR (TERKUNCI):
- Riwayat chat mungkin berisi analisis gambar dalam format '(Analisis AI: ...)'.
- KNOWLEDGE BASE Anda juga mungkin berisi analisis gambar dari file yang disimpan, dalam format '(Konteks dari Analisis Gambar ...: ...)'.
- Teks ini adalah deskripsi dari gambar. ANDA HARUS MENGGUNAKAN KEDUA SUMBER ANALISIS GAMBAR INI sebagai konteks.
- JANGAN PERNAH mengatakan 'Saya tidak bisa melihat gambar'. Anggap teks analisis ini adalah mata Anda.
- Jika user bertanya tentang gambar (baik yang baru di-upload atau yang ada di knowledge base), gunakan teks analisis itu untuk menjawab.

---
ATURAN PERSONA DARI PERUSAHAAN (DARI DATABASE):
{custom_rules}
---

KNOWLEDGE BASE PERUSAHAAN ({tenant_id}):
{context_text}

RIWAYAT PERCAKAPAN DAN PERTANYAAN TERBARU:
{question}

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