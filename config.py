import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AgenticRAGConfig:
    """Configuration class for Multi-tenant Agentic RAG system"""
    # API Keys
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Multi-tenant settings
    tenant_id: str = "SimplyBox"
    
    # Chunking parameters
    base_chunk_size: int = 600
    overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1200
    
    # Pinecone settings
    index_name: str = "simplybox-multimodal"
    dimension: int = 512
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    
    # LLM settings
    model_name: str = "llama-3.3-70b-versatile"
    MAVERICK_MODEL_NAME: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    temperature: float = 0.1
    max_tokens: int = 2048
    request_timeout: float = 120.0
    
    # Embedding model
    embedding_model: str = "ViT-B/32"
    
    # Retrieval settings
    similarity_top_k: int = 7
    final_chunks_count: int = 5
    similarity_threshold: float = 0.3
    
    # Document categories
    CATEGORIES = {
        "FAQ": "Frequently Asked Questions",
        "Payment": "Payment and Billing Information", 
        "Complaint": "Customer Complaints and Issues",
        "Product": "Product Information and Features",
        "Technical": "Technical Support and Troubleshooting",
        "Partnership": "Partnership and Collaboration Inquiries",
        "Legal": "Legal and Compliance Information",
        "General": "General Information"
    }

    @classmethod
    def from_env(cls, tenant_id: str = "SimplyBox"):
        """Create config from environment variables with tenant ID"""
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required but not set in environment variables.")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required but not set in environment variables.")
        
        return cls(
            PINECONE_API_KEY=pinecone_api_key,
            GROQ_API_KEY=groq_api_key,
            tenant_id=tenant_id
        )