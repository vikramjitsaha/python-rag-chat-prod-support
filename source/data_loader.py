"""
Data Loader Module for Production Support RAG System
Loads Excel data into ChromaDB vector database
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict


class DataLoader:
    def __init__(self, excel_path: str = "Input.xlsx", db_path: str = "./chroma_db"):
        """
        Initialize the DataLoader
        
        Args:
            excel_path: Path to the Excel file
            db_path: Path to store ChromaDB database
        """
        # Adjust paths if running from source directory
        from pathlib import Path
        
        # Get the project root (parent of source directory)
        if Path(__file__).parent.name == 'source':
            project_root = Path(__file__).parent.parent
            if not Path(excel_path).is_absolute():
                excel_path = str(project_root / excel_path)
            if not Path(db_path).is_absolute() and db_path.startswith('./'):
                db_path = str(project_root / db_path.lstrip('./'))
        
        self.excel_path = excel_path
        self.db_path = db_path
        
        # Load embedding model from environment variable or use default
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
    def load_excel_data(self) -> pd.DataFrame:
        """
        Load data from Excel file
        
        Returns:
            DataFrame containing the Excel data
        """
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        
        # Read all sheets from Excel file
        excel_file = pd.ExcelFile(self.excel_path)
        
        # Combine all sheets into one DataFrame
        all_data = []
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            
            # Clean up column names - replace "Unnamed: X" with actual header or generic name
            new_columns = []
            for col in df.columns:
                if str(col).startswith('Unnamed:'):
                    # Check if first row has a meaningful value
                    if len(df) > 0 and pd.notna(df.iloc[0][col]) and str(df.iloc[0][col]).strip():
                        new_columns.append(f"Column_{col.split(':')[1].strip()}")
                    else:
                        new_columns.append(col)
                else:
                    new_columns.append(col)
            df.columns = new_columns
            
            df['sheet_name'] = sheet_name
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    
    def prepare_documents(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare documents from DataFrame for vector storage
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        for idx, row in df.iterrows():
            # Combine all column values into a single text document
            text_parts = []
            metadata = {}
            
            for col in df.columns:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")
                    metadata[col] = str(row[col])
            
            document_text = " | ".join(text_parts)
            
            documents.append({
                'id': f"doc_{idx}",
                'text': document_text,
                'metadata': metadata
            })
        
        return documents
    
    def create_or_update_collection(self, documents: List[Dict], collection_name: str = "prod_support"):
        """
        Create or update ChromaDB collection with documents
        
        Args:
            documents: List of document dictionaries
            collection_name: Name of the collection
        """
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Production Support Knowledge Base"}
        )
        
        # Prepare data for ChromaDB
        ids = [doc['id'] for doc in documents]
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            print(f"Added batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
        
        print(f"Successfully loaded {len(documents)} documents into ChromaDB")
    
    def load_data_to_vectordb(self):
        """
        Main method to load Excel data into vector database
        """
        print("Loading Excel data...")
        df = self.load_excel_data()
        print(f"Loaded {len(df)} rows from Excel file")
        
        print("Preparing documents...")
        documents = self.prepare_documents(df)
        
        print("Creating vector database collection...")
        self.create_or_update_collection(documents)
        
        print("Data loading complete!")
        return len(documents)


def main():
    """
    Main function to run the data loader
    """
    loader = DataLoader()
    num_docs = loader.load_data_to_vectordb()
    print(f"\n✓ Successfully loaded {num_docs} documents into the vector database")


if __name__ == "__main__":
    main()
