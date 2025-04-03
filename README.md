# Ollama Chat Application

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-FFA500?style=for-the-badge&logo=ollama&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-00ADD8?style=for-the-badge&logo=langchain&logoColor=white)

A versatile chat application powered by Ollama LLMs with support for multiple data sources including Wikipedia, databases, documents, and websites.

## Features

- **Multiple LLM Models**: Supports various Ollama models (qwen2.5, gemma3, codellama, etc.)
- **Multiple Interaction Modes**:
  - Direct chat with the LLM
  - Wikipedia knowledge retrieval
  - SQL database querying
  - Document analysis (PDF and Word)
  - Website content processing
- **Retrieval Augmented Generation (RAG)**: For document and website content
- **Conversation Memory**: Maintains chat history context
- **Real-time Streaming**: Responses streamed as they're generated
- **Logging**: Comprehensive logging of all interactions

## Prerequisites

Before running the application, ensure you have:

1. [Ollama](https://ollama.ai) installed and running locally
2. Required models pulled (e.g., `ollama pull qwen2.5:1.5b`)
3. Python 3.8 or higher

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ritigit7/Chat-with-all.git
   cd Chat-with-all
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Ollama server (if not already running):
   ```bash
   ollama serve
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. The application will open in your default browser at `http://localhost:8501`

### Application Interface

1. **Sidebar Configuration**:
   - Select your preferred Ollama model
   - Choose a data source type (Direct Chat, Wikipedia, Database, etc.)
   - Configure the selected source (e.g., provide database path or document location)

2. **Main Chat Area**:
   - Type your questions in the chat input
   - View the conversation history
   - Responses will be streamed in real-time
  
![Chat-With-All](images/WhatsApp Image 2025-03-29 at 12.55.47 AM (1).jpeg)

## Supported Data Sources

### 1. Direct Chat
Basic chat interface with the selected Ollama model.

### 2. Wikipedia
Specialized agent for querying Wikipedia knowledge.

### 3. Database
Connect to SQLite databases:
- The agent will analyze the database schema
- Generate and execute SQL queries
- Return results in a readable format

### 4. Document Processing
Supports both PDF and Word documents:
- Extracts text content
- Builds a vector store for semantic search
- Uses RAG for context-aware responses

### 5. Website Processing
- Fetches and processes website content
- Builds a vector store for semantic search
- Uses RAG for context-aware responses

## Configuration

The application creates several directories automatically:
- `logs/` - Stores query logs in JSON format
- `vector_store/` - Stores FAISS vector indexes for documents

## Dependencies

- streamlit
- ollama
- langchain
- langchain-ollama
- PyPDF2
- python-docx
- beautifulsoup4
- faiss-cpu (or faiss-gpu if you have CUDA)
- sqlite3 (built-in)

## Troubleshooting

1. **Ollama connection issues**:
   - Ensure Ollama server is running (`ollama serve`)
   - Verify the model is downloaded (`ollama pull <model-name>`)

2. **Document processing errors**:
   - Check file paths are correct
   - Ensure documents are not password protected

3. **Database connection issues**:
   - Verify the SQLite database exists at the specified path
   - Check the database is not corrupted

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.
