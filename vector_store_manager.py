import os
import uuid
from typing import List, Tuple
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import pinecone
from config import AgenticRAGConfig

class VectorStoreManager:
    """Manages multi-tenant vector store operations and indexing"""
    
    def __init__(self, config: AgenticRAGConfig):
        self.config = config
        self._setup_llm_and_embeddings()
        self._setup_vector_store()

    def _setup_llm_and_embeddings(self):
        """Initialize LLM and embedding model"""
        Settings.llm = Groq(
            model=self.config.model_name,
            api_key=self.config.GROQ_API_KEY,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            request_timeout=self.config.request_timeout
        )
        
        print(f"Initializing Text Embedding Model: {self.config.embedding_model}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding_model
        )
        print(f"Embedding Model Dimension: {self.config.dimension}")

    def _setup_vector_store(self):
        """Setup Pinecone vector store and global index"""
        pc = pinecone.Pinecone(api_key=self.config.PINECONE_API_KEY)

        existing_indexes = [index.name for index in pc.list_indexes()]
        if self.config.index_name not in existing_indexes:
            pc.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec={"serverless": {
                    "cloud": self.config.cloud, 
                    "region": self.config.region
                }}
            )

        self.pinecone_index = pc.Index(self.config.index_name)
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.index = VectorStoreIndex.from_vector_store(vector_store)

    def insert_chunks(self, chunks: List[str], source_filename: str, 
                     tenant_id: str, category: str) -> Tuple[int, int]:
        """Insert chunks into vector store with namespace and metadata"""
        documents_to_insert = []
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{tenant_id}_{category}_{uuid.uuid4().hex[:8]}"

            metadata = {
                "tenant_id": tenant_id,
                "category": category,
                "source": source_filename,
                "chunk_length": len(chunk),
                "type": "text",
                "processing_method": "agentic_optimized",
                "chunk_index": i
            }
            
            doc = Document(
                text=chunk,
                metadata=metadata,
                doc_id=doc_id,
                excluded_embed_metadata_keys=list(metadata.keys())
            )
            documents_to_insert.append(doc)

        successful_inserts = 0
        
        for doc in documents_to_insert:
            try:
                vector_store_with_namespace = PineconeVectorStore(
                    pinecone_index=self.pinecone_index,
                    namespace=tenant_id
                )
                temp_index = VectorStoreIndex.from_vector_store(vector_store_with_namespace)
                temp_index.insert(doc)
                
                successful_inserts += 1
                print(f"Inserted chunk {successful_inserts}/{len(documents_to_insert)} for tenant {tenant_id}", end='\r')
                
            except Exception as e:
                print(f"Failed to insert chunk: {str(e)}")
                continue

        print()
        return successful_inserts, len(documents_to_insert)
    
    def insert_image(self, text_analysis: str, source_filename: str, tenant_id: str, 
                     category: str) -> int:
        """Insert a single image *analysis text* into vector store"""
        doc_id = f"{tenant_id}_{category}_{uuid.uuid4().hex[:8]}"

        formatted_text = f"(Konteks dari Analisis Gambar '{source_filename}': {text_analysis})"

        metadata = {
            "tenant_id": tenant_id,
            "category": category,
            "source": source_filename,
            "type": "image_analysis",
            "chunk_length": len(formatted_text)
        }
        
        doc = Document(
            text=formatted_text,
            doc_id=doc_id,
            metadata=metadata,
            excluded_embed_metadata_keys=list(metadata.keys())
        )

        try:
            vector_store_with_namespace = PineconeVectorStore(
                pinecone_index=self.pinecone_index,
                namespace=tenant_id
            )
            temp_index = VectorStoreIndex.from_vector_store(vector_store_with_namespace)
            temp_index.insert(doc)
            
            print(f"Successfully inserted image analysis text {source_filename} for tenant {tenant_id}")
            return 1
            
        except Exception as e:
            print(f"Failed to insert image analysis text {source_filename}: {str(e)}")
            return 0

    def retrieve_relevant_chunks(self, question: str, tenant_id: str) -> List[str]:
        """Retrieve relevant text chunks and image analysis text"""
        try:
            vector_store_with_namespace = PineconeVectorStore(
                pinecone_index=self.pinecone_index,
                namespace=tenant_id
            )
            tenant_index = VectorStoreIndex.from_vector_store(vector_store_with_namespace)
            
            retriever = tenant_index.as_retriever(similarity_top_k=self.config.similarity_top_k)
            nodes = retriever.retrieve(question)
            
            if not nodes:
                return []

            nodes = sorted(nodes, key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
            
            relevant_chunks = []
            for node in nodes[:self.config.final_chunks_count]:
                relevant_chunks.append(node.node.text)
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Error retrieving chunks for tenant {tenant_id}: {str(e)}")
            return []

    def delete_tenant_data(self, tenant_id: str) -> str:
        """Delete all data for a specific tenant"""
        try:
            self.pinecone_index.delete(delete_all=True, namespace=tenant_id)
            return f"All data for tenant '{tenant_id}' has been deleted"
        except Exception as e:
            return f"Error deleting tenant data: {str(e)}"
        
    def delete_file_data(self, tenant_id: str, file_name: str) -> str:
        """Delete all chunks associated with a specific file for a tenant"""
        try:
            import time
            
            print(f"DEBUG: Starting delete for file '{file_name}' in tenant '{tenant_id}'")
            
            dummy_vector = [0.0] * self.config.dimension

            all_records_response = self.pinecone_index.query(
                namespace=tenant_id,
                vector=dummy_vector,
                top_k=10000,
                include_metadata=True
            )
            
            print(f"DEBUG: Total records in namespace '{tenant_id}': {len(all_records_response.matches)}")
            
            source_files = {}
            matches_to_delete = []

            for match in all_records_response.matches:
                if not match.metadata or 'source' not in match.metadata:
                    continue
                    
                source_value = match.metadata['source']
                
                if source_value not in source_files:
                    source_files[source_value] = 0
                source_files[source_value] += 1

                source_base_name = os.path.splitext(source_value)[0]

                if source_value == file_name or source_base_name == file_name:
                    matches_to_delete.append(match)
            
            print(f"DEBUG: Available source files: {list(source_files.keys())}")
            print(f"DEBUG: Looking for file: '{file_name}'")

            print(f"DEBUG: Filtering logic found: {len(matches_to_delete)} records")
            
            if not matches_to_delete:
                return f"No documents found for file '{file_name}' in tenant '{tenant_id}'. Available files: {list(source_files.keys())}"

            ids_to_delete = [match.id for match in matches_to_delete]
            print(f"DEBUG: Proceeding to delete {len(ids_to_delete)} records")

            batch_size = 1000
            deleted_count = 0
            
            for i in range(0, len(ids_to_delete), batch_size):
                batch_ids = ids_to_delete[i:i + batch_size]
                delete_response = self.pinecone_index.delete(
                    ids=batch_ids,
                    namespace=tenant_id
                )
                deleted_count += len(batch_ids)
                print(f"DEBUG: Deleted batch {i//batch_size + 1}, {len(batch_ids)} records")
            
            time.sleep(3)

            if ids_to_delete:
                verify_response = self.pinecone_index.fetch(ids=ids_to_delete[:100], namespace=tenant_id)
                remaining_count = len(verify_response.get('vectors', {}))
                print(f"DEBUG: Records remaining after deletion (checked {len(ids_to_delete[:100])} IDs): {remaining_count}")
            else:
                remaining_count = 0
                print("DEBUG: No records to delete, skipping verification.")

            
            if remaining_count > 0:
                return f"Partially deleted. {deleted_count} deleted, {remaining_count} remaining for file '{file_name}'"
            
            return f"Successfully deleted {deleted_count} chunks for file '{file_name}' in tenant '{tenant_id}'"
            
        except Exception as e:
            error_msg = f"Error deleting file data for tenant {tenant_id}: {str(e)}"
            print(error_msg)
            return error_msg