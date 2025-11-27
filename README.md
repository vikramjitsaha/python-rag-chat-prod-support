# Production Support RAG Chatbot 🤖



AI-powered chatbot for production support queries using RAG (Retrieval-Augmented Generation).A Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit that answers production support queries based on data from an Excel file. The chatbot uses ChromaDB for vector storage and similarity search, ensuring all answers are grounded in the provided knowledge base.

## Pre-requisite ##
1. Install python 3.13 or above
2. Install PIP
3. create a new file .env inside "Config" folder
4. copy the content of .env.template into .env file that you just created
5. Make sure you have create gemini api key

## Quick Start ## Commands ✨

# 1. Create virtual environment
command : python -m virtualenv .venv

# 2. Activate it
command : .\.venv\Scripts\Activate.ps1

# 3. Upgrade pip (recommended)
command : python -m pip install --upgrade pip

# 4. Install dependencies
command : pip install -r requirements.txt --force-reinstall

# 5. Manual Clean Existing ChromaDB
command : Remove-Item -Recurse -Force chroma_db

# 6. Run the app
command : python run_app.py


### Sidebar Controls

- **System Status**: Shows if the system is ready and document count
- **Data Management**: Reload data from Excel when updated
- **Settings**:
  - Adjust number of context documents (1-5)
  - Toggle source document display
- **Clear Chat History**: Reset the conversation
- **Google API Key**: Optional configuration for enhanced responses


### Standard Queries:
- "What are the common production issues?"
- "How do I resolve database connection errors?"

### Analytical Queries :
- "How many JIRA tickets are total present?"
- "How many tickets are created in October?"
- "How many tickets are created on meter to bill pattern in September?"
- "Show me the DB script executed to delete a call note"
- "List all tickets from November"
- "Find all database scripts"


### Embedding Model Download
On first run, Sentence Transformers will download the embedding model (~90MB). This is normal and happens once.

## Updating the Knowledge Base 🔄
- Make sure you have an updated Input.xlsx (with same header but modified content)
- Run the application and go to browser to access the application 
- Click "🔄 Reload Data from Excel" in the sidebar


## Performance Tips ⚡

- **Smaller Excel Files**: Faster loading and querying
- **Clear Data**: Well-structured data improves response quality
- **Context Documents**: Adjust the slider (1-5) based on query complexity
- **Google Gemini**: Provides more natural responses but requires API key

## Security Notes 🔒

- The application runs locally by default
- No data is sent to external services (unless using Google Gemini)
- API keys are stored in environment variables
- Vector database is stored locally

## Limitations ⚠️

- Answers are limited to information uploaded from k.E.D.B
- Cannot access external knowledge or real-time data
- Response quality depends on Excel data quality
- Without Google API, responses are simpler (but still accurate)

## Technology Stack 🛠️

- **Python 3.8+**: Programming language
- **Streamlit**: Web UI framework
- **ChromaDB**: Vector database
- **Sentence Transformers**: Local embedding model (all-MiniLM-L6-v2)
- **Pandas**: Excel file processing
- **Google Generative AI**: Optional enhanced responses

## Troubleshooting 🤝

For issues or improvements:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure `Input.xlsx` is properly formatted

## License 📄

This project is provided as-is for production support use.

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Developed By**: Vikramjit Saha

Enjoy Production Support RAG Chatbot! 🎉
