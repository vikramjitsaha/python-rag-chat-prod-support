"""
Script to clear ChromaDB collection
"""

import chromadb
import os

def clear_chromadb():
    """Clear all data from ChromaDB"""
    
    # Adjust path if running from source directory
    from pathlib import Path
    
    if Path(__file__).parent.name == 'source':
        project_root = Path(__file__).parent.parent
        db_path = str(project_root / "chroma_db")
    else:
        db_path = "./chroma_db"
    
    collection_name = "prod_support"
    
    if not os.path.exists(db_path):
        print(f"✓ ChromaDB directory '{db_path}' does not exist. Nothing to clear.")
        return
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        
        # Try to get the collection
        try:
            collection = client.get_collection(name=collection_name)
            count = collection.count()
            print(f"Found collection '{collection_name}' with {count} documents")
            
            # Delete the collection
            client.delete_collection(name=collection_name)
            print(f"✓ Successfully deleted collection '{collection_name}'")
            
        except Exception as e:
            print(f"Collection '{collection_name}' does not exist or already deleted")
            print(f"  Error: {e}")
        
        print("\n✓ ChromaDB cleared successfully!")
        print("\nNext steps:")
        print("1. Run: python data_loader.py")
        print("   OR")
        print("2. Use Streamlit app to upload and load data")
        
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")


if __name__ == "__main__":
    print("="*80)
    print("ChromaDB Cleanup Script")
    print("="*80)
    print()
    
    clear_chromadb()
