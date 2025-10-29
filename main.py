import time
from agentic_rag import AgenticRAG
from config import AgenticRAGConfig

def print_help():
    """Print help information"""
    print("\n🤖 Multi-tenant Agentic RAG CS AI System")
    print("="*60)
    print("Commands:")
    print("- 'tenant <tenant_id>'              : Switch to different tenant")
    print("- 'upload <pdf_path> [category]'    : Upload and process PDF with category")
    print("- 'ask <question> [category]'       : Ask question with optional category filter")
    print("- 'categories'                      : List available categories") 
    print("- 'status'                          : Check current tenant index status")
    print("- 'stats'                           : Check all tenants statistics")
    print("- 'delete-tenant <tenant_id>'       : Delete all data for a tenant")
    print("- 'help'                            : Show this help")
    print("- 'exit'                            : Exit the program")
    print("="*60)
    print(f"Current tenant: {config.tenant_id}")
    print("Available categories:", ", ".join(config.CATEGORIES.keys()))
    print()

def main():
    """Main application entry point for multi-tenant system"""
    
    global config
    config = AgenticRAGConfig.from_env()
    
    if config.GROQ_API_KEY == "your-groq-api-key-here":
        print("Please set your GROQ_API_KEY!")
        return

    try:
        print("Initializing Multi-tenant Agentic RAG CS AI System...")
        rag = AgenticRAG(
            tenant_id=config.tenant_id,
            pinecone_api_key=config.PINECONE_API_KEY,
            groq_api_key=config.GROQ_API_KEY
        )

        print_help()

        while True:
            user_input = input(f"[{config.tenant_id}] You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Terima kasih telah menggunakan Multi-tenant Agentic RAG CS AI System!")
                break
            
            elif user_input.lower() == "help":
                print_help()
                
            elif user_input.lower().startswith("tenant "):
                new_tenant_id = user_input[7:].strip()
                if new_tenant_id:
                    rag.switch_tenant(new_tenant_id)
                    config.tenant_id = new_tenant_id
                    print(f"✅ Switched to tenant: {new_tenant_id}")
                else:
                    print("❌ Please specify tenant ID: tenant <tenant_id>")
                    
            elif user_input.lower().startswith("upload "):
                try:
                    parts = user_input[7:].strip().split()
                    if len(parts) < 1:
                        print("❌ Usage: upload <pdf_path> [category]")
                        continue
                        
                    pdf_path = parts[0].strip('"\'')
                    category = parts[1] if len(parts) > 1 else "General"
                    
                    print(f"Memulai proses upload untuk tenant '{config.tenant_id}' dengan kategori '{category}'...")
                    result = rag.upload_pdf(pdf_path, category)
                    print(f"\n✅ {result}")
                    
                except Exception as e:
                    print(f"❌ Error uploading PDF: {e}")
            
            elif user_input.lower().startswith("ask "):
                try:
                    question_part = user_input[4:].strip()
                    
                    words = question_part.split()
                    category_filter = None
                    
                    if len(words) > 1 and words[-1] in config.CATEGORIES:
                        category_filter = words[-1]
                        question = " ".join(words[:-1])
                    else:
                        question = question_part
                    
                    if category_filter:
                        print(f"🔍 Mencari jawaban dalam kategori '{category_filter}' untuk tenant '{config.tenant_id}'...")
                    else:
                        print(f"🔍 Mencari jawaban untuk tenant '{config.tenant_id}'...")
                        
                    answer = rag.ask(question, category_filter)
                    print(f"\n🤖 CS Bot [{config.tenant_id}]: {answer}\n")
                    print("-" * 60)
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            elif user_input.lower() == "categories":
                print(rag.list_categories())
                    
            elif user_input.lower() == "status":
                status = rag.get_index_stats()
                print(f"📊 Index Status: {status}")
            
            elif user_input.lower() == "stats":
                stats = rag.vector_store_manager.get_all_tenants_stats()
                print(f"📊 {stats}")
                
            elif user_input.lower().startswith("delete-tenant "):
                tenant_to_delete = user_input[14:].strip()
                if tenant_to_delete:
                    confirm = input(f"⚠️  Are you sure you want to delete all data for tenant '{tenant_to_delete}'? (yes/no): ")
                    if confirm.lower() == 'yes':
                        result = rag.vector_store_manager.delete_tenant_data(tenant_to_delete)
                        print(f"🗑️  {result}")
                    else:
                        print("Deletion cancelled.")
                else:
                    print("❌ Please specify tenant ID: delete-tenant <tenant_id>")
                
            elif user_input:
                try:
                    print(f"🔍 Mencari jawaban untuk tenant '{config.tenant_id}'...")
                    answer = rag.ask(user_input)
                    print(f"\n🤖 CS Bot [{config.tenant_id}]: {answer}\n")
                    print("-" * 60)
                    time.sleep(1)
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("⚠️  Please enter a valid command or question. Type 'help' for available commands.")

    except Exception as e:
        print(f"❌ Setup Error: {e}")


if __name__ == "__main__":
    main()