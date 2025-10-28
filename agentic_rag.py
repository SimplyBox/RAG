import os
from typing import List, Optional
from config import AgenticRAGConfig
from pdf_handler import PDFHandler
from chunking_manager import ChunkingManager
from vector_store_manager import VectorStoreManager
from query_processor import QueryProcessor
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.clip import ClipEmbedding

class AgenticRAG:
    """Main Multi-tenant Agentic RAG class that orchestrates all components"""
    
    def __init__(self, tenant_id: str = "company_A", pinecone_api_key: str = None, 
                 groq_api_key: str = None, base_chunk_size: int = 600, 
                 overlap: int = 50, min_chunk_size: int = 100, max_chunk_size: int = 1200):
        """Initialize Multi-tenant Agentic RAG with Optimised Grouping"""
        
        self.config = AgenticRAGConfig(
            PINECONE_API_KEY=pinecone_api_key or AgenticRAGConfig.PINECONE_API_KEY,
            GROQ_API_KEY=groq_api_key or AgenticRAGConfig.GROQ_API_KEY,
            tenant_id=tenant_id,
            base_chunk_size=base_chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
        
        Settings.llm = Groq(
            model=self.config.model_name,
            api_key=self.config.GROQ_API_KEY,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            request_timeout=self.config.request_timeout
        )
        print(f"Initializing Multi-Modal Embedding Model: {self.config.embedding_model}")
        Settings.embed_model = ClipEmbedding(
            model_name=self.config.embedding_model
        )
        print(f"Embedding Model Dimension: {self.config.dimension}")

        self.pdf_handler = PDFHandler()

        self.chunking_manager = ChunkingManager(
            base_chunk_size, overlap, min_chunk_size, max_chunk_size,
            embed_model=Settings.embed_model  # Berikan modelnya
        )

        self.vector_store_manager = VectorStoreManager(self.config)

        self.query_processor = QueryProcessor(self.config)

    def upload_pdf(self, pdf_path: str, category: str = "General", original_filename: str = None) -> str:
        """Upload PDF to vector store with agentic chunking and multi-tenant support"""        
        if not self.pdf_handler.validate_pdf_path(pdf_path):
            raise FileNotFoundError(f"PDF file not found or invalid: {pdf_path}")
        
        if category not in self.config.CATEGORIES:
            print(f"Warning: Category '{category}' not in predefined categories. Using 'General'.")
            category = "General"
        
        source_filename = original_filename if original_filename else os.path.basename(pdf_path)
        
        try:
            full_text = self.pdf_handler.extract_text_from_pdf(pdf_path)
            
            print(f"Processing document '{source_filename}' for tenant '{self.config.tenant_id}' with agentic chunking...")
            
            chunks = self.chunking_manager.chunk_text_agentic(full_text)
            
            if not chunks:
                raise ValueError("No meaningful chunks created from PDF")

            print(f"Created {len(chunks)} optimized chunks for category '{category}'")
            
            successful_inserts, total_chunks = self.vector_store_manager.insert_chunks(
                chunks, source_filename, self.config.tenant_id, category
            )

            if successful_inserts == 0:
                raise ValueError("Failed to insert any documents")

            return f"Successfully uploaded {successful_inserts}/{total_chunks} optimized chunks for tenant '{self.config.tenant_id}' in category '{category}'"
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def upload_image(self, image_path: str, category: str = "General", original_filename: str = None) -> str:
        """Analyze image, create vector, and upload to vector store"""
        if category not in self.config.CATEGORIES:
            print(f"Warning: Category '{category}' not in predefined categories. Using 'General'.")
            category = "General"
        
        source_filename = original_filename if original_filename else os.path.basename(image_path)

        try:
            print(f"Analyzing image '{source_filename}' for tenant '{self.config.tenant_id}'...")
            text_analysis = self.query_processor.analyze_image(
                image_url=None,
                caption=f"Internal analysis for knowledge base file: {source_filename}",
                local_image_path=image_path
            )

            if "[Error:" in text_analysis:
                raise ValueError(f"Failed to analyze image with Maverick: {text_analysis}")

            print(f"Image analysis complete. Indexing image vector...")

            successful_insert = self.vector_store_manager.insert_image(
                image_path=image_path,
                source_filename=source_filename,
                tenant_id=self.config.tenant_id,
                category=category,
                text_analysis=text_analysis
            )

            if successful_insert == 0:
                raise ValueError("Failed to insert image vector into Pinecone")

            return f"Successfully uploaded 1/1 image vector for tenant '{self.config.tenant_id}' in category '{category}'"
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")


    def ask(self, question: str, persona_prompt: Optional[str] = None) -> str:
        """Ask question and get answer with enhanced retrieval for specific tenant"""
        if not self.query_processor.validate_question(question):
            return "Silakan ajukan pertanyaan yang valid."

        try:
            relevant_chunks = self.vector_store_manager.retrieve_relevant_chunks(
                question, self.config.tenant_id
            )
            
            return self.query_processor.generate_response(
                question, relevant_chunks, self.config.tenant_id, persona_prompt
            )
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                return "Terlalu banyak permintaan. Silakan tunggu beberapa detik dan coba lagi."
            return f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}"
        
    def analyze_image_with_maverick(self, image_url: str, caption: str | None) -> str:
        """Panggil vision processor ('Maverick') untuk menganalisis gambar."""
        if not image_url:
            raise ValueError("Image URL is required for analysis")

        return self.query_processor.analyze_image(image_url, caption, local_image_path=None)
        
    def delete_file(self, file_name: str) -> str:
        """Delete all chunks associated with a specific file for the current tenant"""
        return self.vector_store_manager.delete_file_data(self.config.tenant_id, file_name)
    
    def switch_tenant(self, new_tenant_id: str):
        """Switch to different tenant"""
        self.config.tenant_id = new_tenant_id
        print(f"Switched to tenant: {new_tenant_id}")
    
    def list_categories(self) -> str:
        """List available categories"""
        categories_text = "Available categories:\n"
        for code, description in self.config.CATEGORIES.items():
            categories_text += f"- {code}: {description}\n"
        return categories_text