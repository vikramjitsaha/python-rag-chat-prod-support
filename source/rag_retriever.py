"""
RAG Retrieval Module for Production Support System
Handles similarity search and context retrieval from ChromaDB
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import google.generativeai as genai
import os
import re
import pandas as pd
from datetime import datetime
import json


class RAGRetriever:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "prod_support", 
                 config_path: str = "./config/patterns_config.json"):
        """
        Initialize the RAG Retriever

        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection to query
            config_path: Path to patterns configuration JSON file
        """
        # Adjust paths if running from source directory
        import sys
        from pathlib import Path
        
        # Get the project root (parent of source directory)
        if Path(__file__).parent.name == 'source':
            project_root = Path(__file__).parent.parent
            db_path = str(project_root / db_path.lstrip('./'))
            config_path = str(project_root / config_path.lstrip('./'))
        
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Load embedding model from environment variable or use default
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Load patterns configuration
        self.config = self._load_config(config_path)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)

        # Get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            raise ValueError(
                f"Collection '{collection_name}' not found. Please run data_loader.py first.")

        # Configure Google Generative AI (optional, for enhanced responses)
        self.use_gemini = False
        if os.getenv("GOOGLE_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                # Use model from environment variable or default to gemini-1.5-flash
                model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
                self.model = genai.GenerativeModel(model_name)
                self.use_gemini = True
            except:
                pass

    def _load_config(self, config_path: str) -> Dict:
        """
        Load patterns configuration from JSON file
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file '{config_path}' not found. Using default configuration.")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Warning: Error parsing config file: {e}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration if config file is not available
        
        Returns:
            Default configuration dictionary
        """
        return {
            "patterns": [
                {"name": "meter to bill", "keywords": ["meter to bill", "meter-to-bill"]},
                {"name": "data dump", "keywords": ["data dump"]},
                {"name": "remediation hold", "keywords": ["remediation hold", "remediation-hold"]},
                {"name": "account status", "keywords": ["account status"]},
                {"name": "missing consumption", "keywords": ["missing consumption"]}
            ],
            "analytical_keywords": ["how many", "count", "total", "list all", "show me all", "find all"],
            "recency_keywords": ["most recent", "latest", "newest", "last", "recent"],
            "type_filters": {
                "jira": {"keywords": ["jira", "ticket"], "match_terms": ["jira", "ticket"]},
                "script": {"keywords": ["db script", "database script", "sql"], "match_terms": ["script", "sql", "delete"]}
            },
            "keyword_filters": [{"name": "call note", "keywords": ["call note", "callnote"]}],
            "action_filters": [{"name": "delete", "keywords": ["delete"]}],
            "months": {
                "january": "01", "february": "02", "march": "03", "april": "04",
                "may": "05", "june": "06", "july": "07", "august": "08",
                "september": "09", "october": "10", "november": "11", "december": "12"
            }
        }

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant context from vector database

        Args:
            query: User query
            top_k: Number of top results to retrieve

        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        retrieved_docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

        return retrieved_docs

    def format_document_text(self, doc_text: str, metadata: Dict) -> str:
        """
        Format document text into a more readable format
        
        Args:
            doc_text: Raw document text with pipe-separated key-value pairs
            metadata: Document metadata
            
        Returns:
            Formatted readable text
        """
        formatted_parts = []
        
        # Define field mapping for generic column names (when Excel has no headers)
        # Based on typical WORK-Old sheet structure
        column_name_mapping = {
            'Column_0': 'Row Number',
            'Column_1': 'JIRA Ticket',
            'Column_2': 'Parent Category',
            'Column_3': 'Category',
            'Column_4': 'Child Category',
            'Column_5': 'Description/Query',
            'Column_6': 'Date',
            'Column_7': 'Assignee',
            'Column_8': 'Team Members',
            'Column_17': 'Status',
            'Column_18': 'Comments/Solution',
            'Column_23': 'Additional Notes'
        }
        
        # Priority fields to display first (in order)
        priority_column_order = [
            'Column_1',  # JIRA Ticket
            'Column_2',  # Parent Category
            'Column_3',  # Category
            'Column_4',  # Child Category
            'Column_5',  # Description
            'Column_6',  # Date
            'Column_7',  # Assignee
            'Column_8',  # Team
            'Column_17', # Status
            'Column_18', # Comments/Solution
        ]
        
        # Track which keys we've already added
        added_keys = set()
        
        # First, add priority columns in order
        for col_key in priority_column_order:
            if col_key in metadata:
                value = metadata[col_key]
                if pd.notna(value) and str(value).strip():
                    value_str = str(value).strip()
                    if value_str and value_str.lower() not in ['nan', 'none', '', 'nat', 'no']:
                        # Use friendly name if available
                        display_name = column_name_mapping.get(col_key, col_key)
                        
                        # Limit very long values but keep more context
                        if len(value_str) > 1000:
                            value_str = value_str[:1000] + "..."
                        
                        formatted_parts.append(f"**{display_name}**: {value_str}")
                        added_keys.add(col_key)
        
        # Then add any remaining non-empty fields (including other Column_X fields)
        for key, value in sorted(metadata.items()):
            if key not in added_keys and key != 'sheet_name':
                if pd.notna(value) and str(value).strip():
                    value_str = str(value).strip()
                    if value_str and value_str.lower() not in ['nan', 'none', '', 'nat', 'no']:
                        # Use friendly name for known columns
                        display_name = column_name_mapping.get(key, key)
                        
                        # Limit very long values
                        if len(value_str) > 1000:
                            value_str = value_str[:1000] + "..."
                        
                        formatted_parts.append(f"**{display_name}**: {value_str}")
                        added_keys.add(key)
        
        # Always add sheet_name at the end if present
        if 'sheet_name' in metadata:
            formatted_parts.append(f"**Source Sheet**: {metadata['sheet_name']}")
        
        return "\n".join(formatted_parts) if formatted_parts else doc_text
    
    def is_recency_query(self, query: str) -> bool:
        """
        Determine if query is asking for most recent/latest items
        """
        recency_keywords = self.config.get('recency_keywords', ['most recent', 'latest', 'newest', 'last', 'recent'])
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in recency_keywords)

    def is_analytical_query(self, query: str) -> bool:
        """
        Determine if the query requires analytical processing (counting, filtering, aggregating)

        Args:
            query: User query

        Returns:
            True if analytical query, False otherwise
        """
        analytical_keywords = self.config.get('analytical_keywords', [
            'how many', 'count', 'total', 'sum', 'average', 'list all',
            'show me all', 'find all', 'get all', 'filter'
        ])

        query_lower = query.lower()
        
        # Check for analytical keywords
        has_analytical = any(keyword in query_lower for keyword in analytical_keywords)
        
        # Also check for recency queries (most recent, latest, etc.)
        has_recency = self.is_recency_query(query)
        
        return has_analytical or has_recency

    def extract_filters_from_query(self, query: str) -> Dict:
        """
        Extract filters from analytical queries using configuration

        Args:
            query: User query

        Returns:
            Dictionary of filters (month, pattern, type, etc.)
        """
        filters = {}
        query_lower = query.lower()

        # Extract month from configuration
        months = self.config.get('months', {})
        for month_name, month_num in months.items():
            if month_name in query_lower:
                filters['month'] = month_name
                filters['month_num'] = month_num
                break

        # Extract type filters from configuration
        type_filters = self.config.get('type_filters', {})
        for filter_type, filter_config in type_filters.items():
            keywords = filter_config.get('keywords', [])
            if any(keyword in query_lower for keyword in keywords):
                filters['type'] = filter_type
                break

        # Extract patterns from configuration
        patterns = self.config.get('patterns', [])
        for pattern_config in patterns:
            pattern_name = pattern_config.get('name')
            keywords = pattern_config.get('keywords', [])
            if any(keyword in query_lower for keyword in keywords):
                filters['pattern'] = pattern_name
                break

        # Extract keyword filters from configuration
        keyword_filters = self.config.get('keyword_filters', [])
        for kw_filter in keyword_filters:
            keywords = kw_filter.get('keywords', [])
            if any(keyword in query_lower for keyword in keywords):
                filters['keyword'] = kw_filter.get('name')
                break

        # Extract action filters from configuration
        action_filters = self.config.get('action_filters', [])
        for act_filter in action_filters:
            keywords = act_filter.get('keywords', [])
            if any(keyword in query_lower for keyword in keywords):
                filters['action'] = act_filter.get('name')
                break

        return filters

    def perform_analytical_query(self, query: str, top_k: int = 100) -> Tuple[str, List[Dict]]:
        """
        Perform analytical queries (counting, filtering, aggregating)

        Args:
            query: User query
            top_k: Number of documents to retrieve for analysis

        Returns:
            Tuple of (response, context_documents)
        """
        # Extract filters from query
        filters = self.extract_filters_from_query(query)

        # Retrieve more documents for analysis
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 500)  # Limit to avoid performance issues
        )

        # Convert to list of documents
        all_docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                all_docs.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

        # Apply filters
        filtered_docs = []
        for doc in all_docs:
            text_lower = doc['text'].lower()
            metadata_str = str(doc['metadata']).lower()
            combined = text_lower + ' ' + metadata_str

            match = True

            # Filter by month
            if 'month' in filters:
                month_pattern = filters['month']
                if month_pattern not in combined:
                    match = False

            # Filter by type
            if 'type' in filters:
                if filters['type'] == 'jira':
                    if not ('jira' in combined or 'ticket' in combined):
                        match = False
                elif filters['type'] == 'script':
                    if not ('script' in combined or 'sql' in combined or 'delete' in combined):
                        match = False

            # Filter by pattern
            if 'pattern' in filters:
                if filters['pattern'] not in combined:
                    match = False

            # Filter by keyword
            if 'keyword' in filters:
                if filters['keyword'] not in combined:
                    match = False

            # Filter by action
            if 'action' in filters:
                if filters['action'] not in combined:
                    match = False

            if match:
                filtered_docs.append(doc)

        # Sort by recency if query asks for "most recent" or "latest"
        if self.is_recency_query(query):
            # Only sort if we have filtered docs - don't fallback to all_docs
            # This prevents showing wrong results when pattern doesn't match
            if filtered_docs:
                docs_to_sort = filtered_docs
            else:
                # If no filtered docs and we have a pattern filter, return empty
                # This means the pattern wasn't found in the data
                if 'pattern' in filters or 'keyword' in filters:
                    # Pattern specified but no matches - don't fallback
                    docs_to_sort = []
                else:
                    # No pattern filter, can use all_docs
                    docs_to_sort = all_docs
            
            # Try to extract dates from metadata and sort
            dated_docs = []
            for doc in docs_to_sort:
                max_date = None
                for key, value in doc['metadata'].items():
                    if 'date' in key.lower() and pd.notna(value):
                        try:
                            # Try to parse date
                            date_val = pd.to_datetime(str(value), errors='coerce')
                            if pd.notna(date_val):
                                if max_date is None or date_val > max_date:
                                    max_date = date_val
                        except:
                            pass
                
                if max_date is not None:
                    dated_docs.append((doc, max_date))
            
            # Sort by date descending (most recent first)
            if dated_docs:
                dated_docs.sort(key=lambda x: x[1], reverse=True)
                filtered_docs = [doc for doc, _ in dated_docs]
                count = len(filtered_docs)  # Update count after sorting

        # Generate analytical response
        count = len(filtered_docs)

        # Build response based on query type
        query_lower = query.lower()

        if self.is_recency_query(query):
            # Recency query
            if count == 0:
                # No results found for the pattern
                pattern_text = ""
                if 'pattern' in filters:
                    pattern_text = f" for **{filters['pattern'].title()}**"
                response = f"I could not find any tickets{pattern_text} in the knowledge base. This might mean:\n"
                response += f"1. There are no {filters.get('pattern', 'such')} tickets in the data\n"
                response += f"2. The category name might be different in the database\n"
                response += f"3. Please check the exact category name in Input.xlsx"
                return response, []
            
            # Show the most recent item
            most_recent = filtered_docs[0]
            formatted_text = self.format_document_text(most_recent['text'], most_recent['metadata'])
            
            # Try to extract ticket number from any field
            ticket_number = None
            child_category = None
            
            for key, value in most_recent['metadata'].items():
                # Look for ticket/JIRA number
                if ticket_number is None:
                    if ('jira' in key.lower() or 'ticket' in key.lower() or 'no' in key.lower()) and pd.notna(value):
                        value_str = str(value).strip()
                        if value_str and 'cb-' in value_str.lower():
                            ticket_number = value_str
                
                # Look for child category
                if child_category is None:
                    if 'child' in key.lower() and 'category' in key.lower() and pd.notna(value):
                        child_category = str(value).strip()
            
            if ticket_number:
                response = f"I have found JIRA ticket **{ticket_number}** as the latest"
                if child_category:
                    response += f" for **\"{child_category}\"** category"
                elif 'pattern' in filters:
                    response += f" for **{filters['pattern'].title()}** category"
                response += ".\n\n" + formatted_text
            else:
                response = f"Here is the most recent record"
                if 'pattern' in filters:
                    response += f" for **{filters['pattern'].title()}**"
                response += ":\n\n" + formatted_text
                
        elif 'how many' in query_lower or 'count' in query_lower or 'total' in query_lower:
            # Counting query
            filter_desc = []
            if 'month' in filters:
                filter_desc.append(f"in {filters['month']}")
            if 'pattern' in filters:
                filter_desc.append(f"related to '{filters['pattern']}'")
            if 'type' in filters:
                filter_desc.append(f"of type '{filters['type']}'")

            filter_text = ' '.join(filter_desc) if filter_desc else ''
            response = f"Based on the production support knowledge base, I found **{count} records** {filter_text}."

            if count > 0 and count <= 5:
                response += "\n\nHere are the details:\n\n"
                for i, doc in enumerate(filtered_docs[:5], 1):
                    formatted_text = self.format_document_text(doc['text'], doc['metadata'])
                    response += f"\n**Record {i}:**\n{formatted_text}\n\n---\n"

        elif 'show me' in query_lower or 'list' in query_lower or 'find' in query_lower:
            # Listing query
            if count == 0:
                response = "I do not have the answer for this."
            elif count == 1:
                formatted_text = self.format_document_text(filtered_docs[0]['text'], filtered_docs[0]['metadata'])
                response = f"I found 1 matching record:\n\n{formatted_text}"
            else:
                response = f"I found **{count} matching records**. Here are the top results:\n\n"
                for i, doc in enumerate(filtered_docs[:3], 1):
                    formatted_text = self.format_document_text(doc['text'], doc['metadata'])
                    response += f"**Record {i}:**\n{formatted_text}\n\n---\n\n"

                if count > 3:
                    response += f"\n_(Showing top 3 of {count} total records)_"
        else:
            # Default analytical response
            if count == 0:
                response = "I do not have the answer for this."
            else:
                response = f"Based on your query, I found **{count} relevant records** in the knowledge base."

                if count > 0:
                    response += "\n\nHere are some examples:\n\n"
                    for i, doc in enumerate(filtered_docs[:3], 1):
                        formatted_text = self.format_document_text(doc['text'], doc['metadata'])
                        response += f"**Example {i}:**\n{formatted_text}\n\n---\n\n"

        return response, filtered_docs[:10]  # Return top 10 for context

    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate response based on query and retrieved context

        Args:
            query: User query
            context_docs: Retrieved context documents

        Returns:
            Generated response
        """
        if not context_docs:
            return "I do not have the answer for this."

        # Format context documents
        formatted_contexts = []
        for i, doc in enumerate(context_docs):
            formatted_text = self.format_document_text(doc['text'], doc['metadata'])
            formatted_contexts.append(f"Document {i+1}:\n{formatted_text}")
        
        context_text = "\n\n".join(formatted_contexts)

        # If Google Gemini is available, use it for better responses
        if self.use_gemini:
            try:
                prompt = f"""You are a polite and helpful production support assistant. 
Answer the user's question based ONLY on the following context from the knowledge base.

Context:
{context_text}

User Question: {query}

Instructions:
- Provide accurate and to-the-point answers
- Be polite and professional
- If the question asks for counts or numbers, provide them clearly
- If the question asks for specific records (like JIRA tickets), identify them clearly with their ticket numbers
- If the question asks for specific data (like queries or scripts), provide them in formatted code blocks
- If the context doesn't contain the answer, respond with: "I do not have the answer for this."
- Do not make up information outside the provided context
- Keep your response concise and relevant
- Use markdown formatting for better readability

Answer:"""

                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Error using Gemini: {e}")
                # Fall back to simple retrieval

        # Simple response without LLM - return formatted first document
        # Check if context is relevant (basic distance threshold)
        if context_docs[0].get('distance', 0) > 1.5:
            return "I do not have the answer for this."

        # Return the most relevant formatted document
        formatted_text = self.format_document_text(context_docs[0]['text'], context_docs[0]['metadata'])
        response = f"Based on the production support knowledge base:\n\n{formatted_text}"
        return response

    def answer_query(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """
        Main method to answer user queries

        Args:
            query: User query
            top_k: Number of context documents to retrieve

        Returns:
            Tuple of (response, context_documents)
        """
        # Check if this is an analytical query
        if self.is_analytical_query(query):
            # Use analytical query processing
            return self.perform_analytical_query(query, top_k=100)
        else:
            # Use standard RAG retrieval
            context_docs = self.retrieve_context(query, top_k)
            response = self.generate_response(query, context_docs)
            return response, context_docs


def main():
    """
    Test the RAG retriever
    """
    retriever = RAGRetriever()

    test_query = "What are the common production issues?"
    print(f"Query: {test_query}")

    response, context = retriever.answer_query(test_query)
    print(f"\nResponse: {response}")
    print(f"\nRetrieved {len(context)} context documents")


if __name__ == "__main__":
    main()
