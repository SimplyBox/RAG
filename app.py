import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import tempfile
import os
import logging
from datetime import datetime

from agentic_rag import AgenticRAG
from config import AgenticRAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-tenant Agentic RAG API",
    description="REST API for Multi-tenant RAG system with agentic chunking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeImageRequest(BaseModel):
    company_id: str
    image_url: str
    caption: Optional[str] = None

class IngestRequest(BaseModel):
    company_id: str
    category: Optional[str] = "General"

class QueryRequest(BaseModel):
    company_id: str
    question: str
    persona_prompt: Optional[str] = None

class IngestResponse(BaseModel):
    success: bool
    message: str
    company_id: str
    category: str
    chunks_processed: Optional[int] = None

class QueryResponse(BaseModel):
    success: bool
    answer: str
    company_id: str
    timestamp: str

class DeleteFileRequest(BaseModel):
    company_id: str
    file_name: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

rag_instances: Dict[str, AgenticRAG] = {}

def get_rag_instance(company_id: str) -> AgenticRAG:
    """Get or create RAG instance for specific company"""
    if company_id not in rag_instances:
        try:
            rag_instances[company_id] = AgenticRAG(tenant_id=company_id)
            logger.info(f"Created new RAG instance for company: {company_id}")
        except Exception as e:
            logger.error(f"Failed to create RAG instance for {company_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG for company {company_id}")
    
    return rag_instances[company_id]

def process_document_background(
    temp_file_path: str,
    file_ext: str,
    company_id: str,
    category: str,
    source_filename_with_ext: str
):
    """Fungsi ini akan berjalan di background"""
    try:
        rag = get_rag_instance(company_id)
        
        logger.info(f"[BG Task] Processing upload for company {company_id}, file {source_filename_with_ext}")
        
        result = ""
        if file_ext == '.pdf':
            result = rag.upload_pdf(temp_file_path, category, source_filename_with_ext)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.webp']:
            result = rag.upload_image(temp_file_path, category, source_filename_with_ext)
        
        logger.info(f"[BG Task] Successfully processed upload for company {company_id}: {result}")
        
    except Exception as e:
        logger.error(f"[BG Task] Error processing upload for {company_id}: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"[BG Task] Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.error(f"[BG Task] Error cleaning up temp file {temp_file_path}: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company_id: str = Form(...),
    category: str = Form("General"),
    file_name: str = Form(None)
):
    """
    Ingest PDF or Image document for specific company
    
    - **file**: PDF, PNG, JPG, JPEG, WEBP file to upload
    - **company_id**: Company identifier (will be used as tenant_id)
    - **category**: Document category (FAQ, Payment, Complaint, Product, Technical, General)
    - **file_name**: Custom filename from frontend (optional, will use original if not provided)
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no filename")

    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_pdf = ['.pdf']
    allowed_images = ['.png', '.jpg', '.jpeg', '.webp']

    if file_ext not in allowed_pdf and file_ext not in allowed_images:
        raise HTTPException(status_code=400, detail=f"Only PDF, PNG, JPG, JPEG, WEBP files are allowed. Got {file_ext}")
    
    if not company_id or not company_id.strip():
        raise HTTPException(status_code=400, detail="company_id is required")

    if file_name and file_name.strip():
        original_filename = file_name.strip()
        logger.info(f"Using custom filename from frontend: '{original_filename}'")
    else:
        original_filename = file.filename
        if original_filename:
            original_filename = os.path.splitext(original_filename)[0]
        logger.info(f"Using original filename: '{original_filename}'")

    if original_filename:
        original_filename = os.path.splitext(original_filename)[0]

    source_filename_with_ext = f"{original_filename}{file_ext}"
    logger.info(f"Final source filename to be stored: '{source_filename_with_ext}'")
    
    try:
        get_rag_instance(company_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        logger.info(f"Queueing background task for company {company_id}, file {source_filename_with_ext}")

        background_tasks.add_task(
            process_document_background,
            temp_file_path=temp_file_path,
            file_ext=file_ext,
            company_id=company_id,
            category=category,
            source_filename_with_ext=source_filename_with_ext
        )

        return {
            "success": True,
            "message": "Upload accepted and is being processed in the background.",
            "company_id": company_id,
            "file_name": source_filename_with_ext
        }
        
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        logger.error(f"Error queuing upload task for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error queuing upload task: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents for specific company
    
    - **company_id**: Company identifier
    - **question**: Question to ask
    - **persona_prompt**: (Optional) Custom persona string
    """
    
    if not request.company_id or not request.company_id.strip():
        raise HTTPException(status_code=400, detail="company_id is required")
    
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    
    try:
        rag = get_rag_instance(request.company_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        logger.info(f"Processing query for company {request.company_id}: {request.question}")
        
        answer = rag.ask(request.question, request.persona_prompt)
        
        logger.info(f"Successfully generated answer for company {request.company_id}")
        
        return QueryResponse(
            success=True,
            answer=answer,
            company_id=request.company_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing query for company {request.company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
@app.post("/analyze_image", response_model=Dict[str, Any])
async def analyze_image_endpoint(request: AnalyzeImageRequest):
    """
    Analyze an image using the 'Maverick' vision model
    """
    if not request.company_id:
        raise HTTPException(status_code=400, detail="company_id is required")
    if not request.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")
    
    try:
        rag = get_rag_instance(request.company_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        logger.info(f"Processing image analysis for company {request.company_id}...")
        
        analysis_text = rag.analyze_image_with_maverick(request.image_url, request.caption)
        
        logger.info(f"Successfully generated image analysis for company {request.company_id}")
        
        return {
            "success": True,
            "analysis": analysis_text,
            "company_id": request.company_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing image analysis for {request.company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get available document categories"""
    try:
        config = AgenticRAGConfig()
        return {
            "success": True,
            "categories": config.CATEGORIES
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting categories: {str(e)}")

@app.delete("/tenant/{company_id}")
async def delete_company_data(company_id: str):
    """
    Delete all data for specific company
    
    - **company_id**: Company identifier
    """
    
    if not company_id or not company_id.strip():
        raise HTTPException(status_code=400, detail="company_id is required")
    
    try:
        rag = get_rag_instance(company_id)
        result = rag.vector_store_manager.delete_tenant_data(company_id)
        
        if company_id in rag_instances:
            del rag_instances[company_id]
        
        return {
            "success": True,
            "message": result,
            "company_id": company_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting data for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting company data: {str(e)}")
    
@app.delete("/file", response_model=Dict[str, Any])
async def delete_file(request: DeleteFileRequest):
    """
    Delete specific file data for a company from vector store
    
    - **company_id**: Company identifier
    - **file_name**: Name of the file to delete
    """
    
    if not request.company_id or not request.company_id.strip():
        raise HTTPException(status_code=400, detail="company_id is required")
    
    if not request.file_name or not request.file_name.strip():
        raise HTTPException(status_code=400, detail="file_name is required")
    
    try:
        rag = get_rag_instance(request.company_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        logger.info(f"Deleting file {request.file_name} for company {request.company_id}")
        
        result = rag.delete_file(request.file_name)
        
        logger.info(f"Successfully deleted file {request.file_name} for company {request.company_id}")
        
        return {
            "success": True,
            "message": result,
            "company_id": request.company_id,
            "file_name": request.file_name
        }
        
    except Exception as e:
        logger.error(f"Error deleting file {request.file_name} for company {request.company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-tenant Agentic RAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ingest": "Upload and process PDF documents",
            "POST /query": "Query documents and get answers",
            "GET /categories": "Get available categories",
            "DELETE /tenant/{company_id}": "Delete company data",
            "DELETE /file": "Delete specific file data",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )