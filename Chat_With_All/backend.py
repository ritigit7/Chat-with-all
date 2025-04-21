import os
import time
import json
import logging
import sqlite3
import urllib.request
import base64
from datetime import datetime
from io import BytesIO

import pandas as pd
import PyPDF2
import docx
import requests
import wikipedia
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from openai import OpenAI

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.messages import HumanMessage, AIMessage


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("/home/ritik/Documents/VScode/Chat_With_All/logs/chat_app.log"),

        logging.StreamHandler()
    ],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ollama_chat_app')

# Create necessary directories
for directory in ['/home/ritik/Documents/VScode/Chat_With_All/logs', '/home/ritik/Documents/VScode/Chat_With_All/vector_store', '/home/ritik/Documents/VScode/Chat_With_All/uploads', '/home/ritik/Documents/VScode/Chat_With_All/static', '/home/ritik/Documents/VScode/Chat_With_All/templates']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Global variables to store state
states = {}

def get_ollama_llm(model="qwen2.5:1.5b"):
    """Initialize Ollama LLM through langchain."""
    try:
        logger.info(f"Initializing Ollama LLM with model {model}")
        return OllamaLLM(model=model, temperature=0.5)
    except Exception as e:
        logger.error(f"Error initializing Ollama LLM: {str(e)}")
        return None

def get_embeddings_model():
    """Get the embeddings model for vector storage."""
    try:
        logger.info("Initializing embeddings model")
        return OllamaEmbeddings(model="nomic-embed-text:latest")
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {str(e)}")
        return None

def setup_wikipedia_agent(model_name, session_id):
    """Set up an agent for Wikipedia queries."""
    try:
        logger.info("Setting up Wikipedia agent")
        llm = get_ollama_llm(model_name)
        if not llm:
            return None
            
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        tools = [
            Tool(
                name="Wikipedia",
                func=wikipedia_tool.run,
                description="Useful for querying information from Wikipedia. The input should be a search query."
            )
        ]
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
            verbose=True, 
            memory=memory
        )
        
        logger.info("Wikipedia agent setup complete")
        states[session_id]['agent'] = agent
        return {"status": "success", "message": "Wikipedia agent initialized successfully"}
    except Exception as e:
        logger.error(f"Error setting up Wikipedia agent: {str(e)}")
        return {"status": "error", "message": f"Error setting up Wikipedia agent: {str(e)}"}

def setup_database_agent(model_name, db_path, session_id):
    """Set up an agent for SQL database queries."""
    try:
        logger.info(f"Setting up database agent for {db_path}")
        llm = get_ollama_llm(model_name)
        if not llm:
            return {"status": "error", "message": "Failed to initialize LLM"}
            
        # SQL Query generation prompt
        sql_prompt = PromptTemplate(
            input_variables=["question", "table_info"],
            template="""
            You are an expert SQL assistant. Based on the database information below, 
            write a SQL query to answer the question: {question}
            
            Database tables and schema:
            {table_info}
            
            SQL Query:
            """
        )
        
        sql_chain = LLMChain(llm=llm, prompt=sql_prompt)
        
        def get_table_info():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            table_info = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                column_desc = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
                table_info.append(f"Table: {table_name}, Columns: {column_desc}")
            
            conn.close()
            return "\n".join(table_info)
        
        def execute_sql_query(query):
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df.to_string()
            except Exception as e:
                return f"Error executing SQL query: {str(e)}"
        
        def sql_agent(question):
            table_info = get_table_info()
            sql_query = sql_chain.run(question=question, table_info=table_info)
            
            # Extract query if it's wrapped in ```
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].strip()
                
            logger.info(f"Generated SQL query: {sql_query}")
            
            result = execute_sql_query(sql_query)
            
            return f"""
            Question: {question}
            
            Generated SQL Query: 
            ```sql
            {sql_query}
            ```
            
            Query Result:
            {result}
            """
        
        logger.info("Database agent setup complete")
        states[session_id]['agent'] = sql_agent
        states[session_id]['db_details'] = {"path": db_path}
        return {"status": "success", "message": "Database agent initialized successfully"}
    except Exception as e:
        logger.error(f"Error setting up database agent: {str(e)}")
        return {"status": "error", "message": f"Error setting up database agent: {str(e)}"}

def setup_rag_for_document(model_name, content, source_type, session_id):
    """Set up RAG for document content."""
    try:
        logger.info(f"Setting up RAG for {source_type}")
        llm = get_ollama_llm(model_name)
        if not llm:
            return {"status": "error", "message": "Failed to initialize LLM"}
            
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(content)
        logger.info(f"Split {source_type} into {len(chunks)} chunks")
        
        # Create embeddings and vectorstore
        embeddings = get_embeddings_model()
        if not embeddings:
            return {"status": "error", "message": "Failed to initialize embeddings model"}
            
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create conversation chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )

        
        logger.info(f"RAG setup complete for {source_type}")
        states[session_id]['conversation_chain'] = conversation_chain
        states[session_id]['vectorstore'] = vectorstore
        return {"status": "success", "message": f"{source_type} processed successfully"}
    except Exception as e:
        logger.error(f"Error setting up RAG for {source_type}: {str(e)}")
        return {"status": "error", "message": f"Error setting up RAG for {source_type}: {str(e)}"}

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        logger.info(f"Opening PDF file: {file_path}")
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(file_path):
    """Extract text from a Word document."""
    try:
        logger.info(f"Opening Word document: {file_path}")
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        logger.info(f"Extracted {len(text)} characters from Word document")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from Word document: {str(e)}")
        return f"Error extracting text from Word document: {str(e)}"

def extract_text_from_website(url):
    """Fetch and extract text from a website."""
    try:
        logger.info(f"Fetching website content: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Extracted {len(text)} characters from website")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from website: {str(e)}")
        return f"Error extracting text from website: {str(e)}"

def query_ollama(prompt, model="qwen2.5:1.5b"):
    """Send a query to the local Ollama model."""
    try:
        logger.info(f"Querying Ollama with model: {model}")
        start_time = time.time()

        # Initialize Ollama client
        ollama_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        # Prepare message payload
        messages = [
            {"role": "system", "content": "Answer the following question:"},
            {"role": "user", "content": prompt},
        ]

        # Send request to Ollama
        response = ollama_client.chat.completions.create(
            model=model,
            messages=messages
        )

        result = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        logger.info(f"Ollama response received in {elapsed_time:.2f} seconds")

        return result or "No response from model"

    except Exception as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        return f"Error connecting to Ollama: {str(e)}"

def chat_with_image(prompt, image_data, model="qwen2.5:1.5b"):
    """Process an image with text prompt using Ollama."""
    try:
        logger.info(f"Processing image with model {model}")
        
        # Initialize Ollama client
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        # Create the message with text and image content
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
        )
        
        logger.info("Image processing complete")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}"

# Initialize session data
@app.route('/api/init_session', methods=['POST'])
def init_session():
    session_id = request.json.get('session_id')
    if not session_id:
        session_id = f"session_{int(time.time())}"
    
    if session_id not in states:
        states[session_id] = {
            'messages': [],
            'model': "qwen2.5:1.5b",
            'selected_source': "Direct Chat",
            'agent': None,
            'conversation_chain': None,
            'vectorstore': None,
            'db_details': None,
            'document_path': None,
            'chat_history': []
        }
    
    return jsonify({
        'status': 'success',
        'session_id': session_id,
        'model': states[session_id]['model'],
        'source': states[session_id]['selected_source']
    })

# Get models
@app.route('/api/models', methods=['GET'])
def get_models():
    models = ['qwen2.5:1.5b','gemma3:1b', 'codellama', 'llama3']
    return jsonify(models)

# Set model
@app.route('/api/set_model', methods=['POST'])
def set_model():
    session_id = request.json.get('session_id')
    model = request.json.get('model')
    
    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})
    
    states[session_id]['model'] = model
    return jsonify({'status': 'success', 'model': model})

# Set source
@app.route('/api/set_source', methods=['POST'])
def set_source():
    session_id = request.json.get('session_id')
    source = request.json.get('source')
    
    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})
    
    states[session_id]['selected_source'] = source
    return jsonify({'status': 'success', 'source': source})

# Initialize Wikipedia agent
@app.route('/api/init_wikipedia', methods=['POST'])
def init_wikipedia():
    session_id = request.json.get('session_id')
    
    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})
    
    result = setup_wikipedia_agent(states[session_id]['model'], session_id)
    return jsonify(result)

# Initialize database agent
@app.route('/api/init_database', methods=['POST'])
def init_database():
    session_id = request.json.get('session_id')
    db_path = request.json.get('db_path')
    
    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})
    
    if not os.path.exists(db_path):
        return jsonify({'status': 'error', 'message': f'Database file not found: {db_path}'})
    
    result = setup_database_agent(states[session_id]['model'], db_path, session_id)
    return jsonify(result)

# Process PDF document
@app.route('/api/process_pdf', methods=['POST'])
def process_pdf():
    session_id = request.form.get('session_id')
    
    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file and file.filename.endswith('.pdf'):
        filename = f"/home/ritik/Documents/VScode/Chat_With_All/uploads/{int(time.time())}_{file.filename}"
        file.save(filename)
        states[session_id]['document_path'] = filename
        
        text_content = extract_text_from_pdf(filename)
        if not "Error" in text_content:
            result = setup_rag_for_document(
                states[session_id]['model'],
                text_content,
                f"PDF ({filename})",
                session_id
            )
            return jsonify(result)
        else:
            return jsonify({'status': 'error', 'message': text_content})
    
    return jsonify({'status': 'error', 'message': 'Invalid file type'})

# Process Word document
@app.route('/api/process_docx', methods=['POST'])
def process_docx():
    session_id = request.form.get('session_id')
    
    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file and file.filename.endswith('.docx'):
        filename = f"/home/ritik/Documents/VScode/Chat_With_All/uploads/{int(time.time())}_{file.filename}"
        file.save(filename)
        states[session_id]['document_path'] = filename
        
        text_content = extract_text_from_docx(filename)
        if not "Error" in text_content:
            result = setup_rag_for_document(
                states[session_id]['model'],
                text_content,
                f"Word document ({filename})",
                session_id
            )
            return jsonify(result)
        else:
            return jsonify({'status': 'error', 'message': text_content})
    
    return jsonify({'status': 'error', 'message': 'Invalid file type'})

# Process website
@app.route('/api/process_website', methods=['POST'])
def process_website():
    session_id = request.json.get('session_id')
    website_url = request.json.get('url')
    
    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})
    
    text_content = extract_text_from_website(website_url)
    if not "Error" in text_content:
        result = setup_rag_for_document(
            states[session_id]['model'],
            text_content,
            f"Website ({website_url})",
            session_id
        )
        return jsonify(result)
    else:
        return jsonify({'status': 'error', 'message': text_content})

# Process image
@app.route('/api/process_image', methods=['POST'])
def process_image():
    session_id = request.form.get('session_id')
    prompt = request.form.get('prompt', 'Describe this image')
    
    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image part'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected image'})
    
    # Process the image file
    image_content = file.read()
    image_b64 = base64.b64encode(image_content).decode('utf-8')
    
    # Use multimodal capabilities with Gemma
    model = "qwen2.5:1.5b"  # Multimodal model
    response = chat_with_image(prompt, image_b64, model)
    
    # Add to chat history
    states[session_id]['messages'].append({"role": "user", "content": f"{prompt} [Image uploaded]"})
    states[session_id]['messages'].append({"role": "assistant", "content": response})
    
    return jsonify({
        'status': 'success',
        'response': response,
        'messages': states[session_id]['messages']
    })

# Send chat message
@app.route('/api/chat', methods=['POST'])
def chat():
    session_id = request.json.get('session_id')
    message = request.json.get('message')

    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})

    # Add user message to history
    states[session_id]['messages'].append(HumanMessage(content=message))

    # Log the query
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": states[session_id]['model'],
        "source_type": states[session_id]['selected_source'],
        "query": message
    }

    with open(os.path.join('/home/ritik/Documents/VScode/Chat_With_All/logs', f'queries_{datetime.now().strftime("%Y%m%d")}.json'), 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    # Process based on source type
    response = ""
    source_type = states[session_id]['selected_source']

    if source_type == "Direct Chat":
        response = query_ollama(message, states[session_id]['model'])

    elif source_type == "Wikipedia":
        if states[session_id]['agent']:
            try:
                result = states[session_id]['agent'].invoke(message)
                if isinstance(result, str):
                    response = result
                elif isinstance(result, dict) and "output" in result:
                    response = result["output"]
                else:
                    response = str(result) # Fallback to string conversion
            except Exception as e:
                response = f"Error from Wikipedia agent: {str(e)}"
        else:
            response = "Wikipedia agent has not been initialized. Please initialize it first."

    elif source_type == "Database":
        if states[session_id]['agent'] and states[session_id]['db_details']:
            try:
                response = states[session_id]['agent'](message)
            except Exception as e:
                response = f"Error from database agent: {str(e)}"
        else:
            response = "Database agent has not been initialized. Please connect to a database first."

    elif source_type in ["PDF Document", "Word Document", "Website"]:
        if states[session_id]['conversation_chain']:
            try:
                result = states[session_id]['conversation_chain']({"question": message, "chat_history": states[session_id]['chat_history']})
                states[session_id]['chat_history'].append((message, result["answer"]))
                response = result["answer"]
            except Exception as e:
                response = f"Error from RAG system: {str(e)}"
        else:
            source_name = source_type.lower()
            response = f"{source_type} has not been processed. Please process a {source_name} first."

    # Add response to chat history
    states[session_id]['messages'].append(AIMessage(content=response))

    # Convert messages to a list of dictionaries for JSON serialization
    history_for_json = [
        {"role": msg.type, "content": msg.content}
        for msg in states[session_id]['messages']
    ]

    return jsonify({
        'status': 'success',
        'response': response,
        'messages': history_for_json
    })

# Get chat history
@app.route('/api/history', methods=['POST'])
def get_history():
    session_id = request.json.get('session_id')

    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})

    # Convert messages to a list of dictionaries for JSON serialization
    history_for_json = [
        {"role": msg.type, "content": msg.content}
        for msg in states[session_id]['messages']
    ]

    return jsonify({
        'status': 'success',
        'messages': history_for_json
    })

# Clear chat history
@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    session_id = request.json.get('session_id')

    if session_id not in states:
        return jsonify({'status': 'error', 'message': 'Session not found'})

    states[session_id]['messages'] = []
    states[session_id]['chat_history'] = []

    return jsonify({'status': 'success', 'message': 'Chat history cleared'})
# Render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)