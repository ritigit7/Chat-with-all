import streamlit as st
import requests
import logging
import os
import time
import pandas as pd
from datetime import datetime
import json
import wikipedia
import sqlite3
import PyPDF2
from openai import OpenAI 
import docx
import urllib.request
from bs4 import BeautifulSoup
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate

# Set up logging
logging.basicConfig(
    filename='chat_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ollama_chat_app')

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a vector store directory if it doesn't exist
if not os.path.exists('vector_store'):
    os.makedirs('vector_store')

def setup_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_source' not in st.session_state:
        st.session_state.selected_source = "Direct Chat"
    if 'source_content' not in st.session_state:
        st.session_state.source_content = ""
    if 'model' not in st.session_state:
        st.session_state.model = "qwen2.5:1.5b"
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'db_details' not in st.session_state:
        st.session_state.db_details = None
    if 'document_path' not in st.session_state:
        st.session_state.document_path = None
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def get_ollama_llm(model="qwen2.5:1.5b"):
    """Initialize Ollama LLM through langchain."""
    try:
        logger.info(f"Initializing Ollama LLM with model {model}")
        return Ollama(model=model, temperature=0.5)
    except Exception as e:
        logger.error(f"Error initializing Ollama LLM: {str(e)}")
        st.error(f"Error initializing Ollama LLM: {str(e)}")
        return None

def get_embeddings_model():
    """Get the embeddings model for vector storage."""
    try:
        logger.info("Initializing embeddings model")
        return  OllamaEmbeddings(model="nomic-embed-text:latest")
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {str(e)}")
        st.error(f"Error initializing embeddings model: {str(e)}")
        return None

def setup_wikipedia_agent(model_name):
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
        return agent
    except Exception as e:
        logger.error(f"Error setting up Wikipedia agent: {str(e)}")
        st.error(f"Error setting up Wikipedia agent: {str(e)}")
        return None

def setup_database_agent(model_name, db_details):
    """Set up an agent for SQL database queries."""
    try:
        logger.info(f"Setting up database agent for {db_details['path']}")
        llm = get_ollama_llm(model_name)
        if not llm:
            return None
            
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
            conn = sqlite3.connect(db_details['path'])
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
                conn = sqlite3.connect(db_details['path'])
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
        return sql_agent
    except Exception as e:
        logger.error(f"Error setting up database agent: {str(e)}")
        st.error(f"Error setting up database agent: {str(e)}")
        return None

def setup_rag_for_document(model_name, content, source_type):
    """Set up RAG for document or PDF content."""
    try:
        logger.info(f"Setting up RAG for {source_type}")
        llm = get_ollama_llm(model_name)
        if not llm:
            return None
            
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
            return None
            
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
        return conversation_chain
    except Exception as e:
        logger.error(f"Error setting up RAG for {source_type}: {str(e)}")
        st.error(f"Error setting up RAG for {source_type}: {str(e)}")
        return None

def setup_website_rag(model_name, url):
    """Set up RAG for website content."""
    try:
        logger.info(f"Setting up RAG for website: {url}")
        
        # Fetch website content
        content = extract_text_from_website(url)
        if "Error" in content:
            return None
            
        # Use the document RAG setup function
        return setup_rag_for_document(model_name, content, f"website ({url})")
    except Exception as e:
        logger.error(f"Error setting up website RAG: {str(e)}")
        st.error(f"Error setting up website RAG: {str(e)}")
        return None


def query_ollama(prompt, model="qwen2.5:1.5b"):
    """Send a query to the local Ollama model and stream the response dynamically."""
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

        # Send request to Ollama with streaming enabled
        response = ollama_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        result = ""
        placeholder = st.empty()  # Create a placeholder for real-time updates

        # Stream and update response in real-time
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                result += content  # Accumulate response
                placeholder.write(result)  # Update displayed text dynamically

        elapsed_time = time.time() - start_time
        logger.info(f"Ollama response received in {elapsed_time:.2f} seconds")

        return result or "No response from model"

    except Exception as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        return f"Error connecting to Ollama: {str(e)}"


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

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Chat with Ollama",
        page_icon="💬",
        layout="wide"
    )
    
    setup_session_state()
    
    st.title("🤖 Chat with Ollama")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        # available_models = ["qwen2.5:1.5b", "llama3", "mistral", "orca-mini", "phi", "gemma"]
        available_models = ['qwen2.5:1.5b','nomic-embed-text','gemma3:1b','qwen2.5:1.5b','codellama','qwen2.5:1.5b']

        st.session_state.model = st.selectbox(
            "Select Ollama Model", 
            available_models,
            index=available_models.index(st.session_state.model)
        )
        
        # Source selection
        st.header("Data Source")
        source_type = st.radio(
            "Select Source Type",
            ["Direct Chat", "Wikipedia", "Database", "PDF Document", 
             "Word Document", "Website"]
        )
        
        # Reset agents and chains if source type changes
        if st.session_state.selected_source != source_type:
            st.session_state.agent = None
            st.session_state.conversation_chain = None
            st.session_state.db_details = None
            st.session_state.vectorstore = None
            st.session_state.document_path = None
            
        st.session_state.selected_source = source_type
        
        # Show appropriate input fields based on source type
        if source_type == "Wikipedia":
            st.info("This mode uses a specialized Wikipedia agent to answer your questions.")
            if st.button("Initialize Wikipedia Agent"):
                with st.spinner("Setting up Wikipedia agent..."):
                    st.session_state.agent = setup_wikipedia_agent(st.session_state.model)
                    if st.session_state.agent:
                        st.success("Wikipedia agent is ready!")
                    else:
                        st.error("Failed to initialize Wikipedia agent")
        
        elif source_type == "Database":
            st.info("This mode connects to a SQLite database and uses an agent to generate SQL queries.")
            db_path = st.text_input("Database Path (SQLite)", "example.db")
            
            if st.button("Connect to Database"):
                if os.path.exists(db_path):
                    st.session_state.db_details = {"path": db_path}
                    with st.spinner("Setting up database agent..."):
                        st.session_state.agent = setup_database_agent(
                            st.session_state.model, 
                            st.session_state.db_details
                        )
                        if st.session_state.agent:
                            st.success("Database agent is ready!")
                        else:
                            st.error("Failed to initialize database agent")
                else:
                    st.error(f"Database file not found: {db_path}")
        
        elif source_type == "PDF Document":
            st.info("This mode uses RAG (Retrieval Augmented Generation) to answer questions about PDF content.")
            pdf_path = st.text_input("PDF File Path", "document.pdf")
            
            if st.button("Process PDF Document"):
                if os.path.exists(pdf_path):
                    st.session_state.document_path = pdf_path
                    with st.spinner("Processing PDF document..."):
                        text_content = extract_text_from_pdf(pdf_path)
                        if not "Error" in text_content:
                            st.session_state.conversation_chain = setup_rag_for_document(
                                st.session_state.model,
                                text_content,
                                f"PDF ({pdf_path})"
                            )
                            if st.session_state.conversation_chain:
                                st.success("PDF processed successfully! Ready for questions.")
                            else:
                                st.error("Failed to setup RAG for PDF document")
                        else:
                            st.error(text_content)
                else:
                    st.error(f"PDF file not found: {pdf_path}")
        
        elif source_type == "Word Document":
            st.info("This mode uses RAG to answer questions about Word document content.")
            docx_path = st.text_input("Word File Path", "document.docx")
            
            if st.button("Process Word Document"):
                if os.path.exists(docx_path):
                    st.session_state.document_path = docx_path
                    with st.spinner("Processing Word document..."):
                        text_content = extract_text_from_docx(docx_path)
                        if not "Error" in text_content:
                            st.session_state.conversation_chain = setup_rag_for_document(
                                st.session_state.model,
                                text_content,
                                f"Word document ({docx_path})"
                            )
                            if st.session_state.conversation_chain:
                                st.success("Word document processed successfully! Ready for questions.")
                            else:
                                st.error("Failed to setup RAG for Word document")
                        else:
                            st.error(text_content)
                else:
                    st.error(f"Word file not found: {docx_path}")
        
        elif source_type == "Website":
            st.info("This mode uses RAG to answer questions about website content.")
            website_url = st.text_input("Website URL", "https://example.com")
            
            if st.button("Process Website"):
                with st.spinner("Processing website content..."):
                    st.session_state.conversation_chain = setup_website_rag(
                        st.session_state.model,
                        website_url
                    )
                    if st.session_state.conversation_chain:
                        st.success("Website processed successfully! Ready for questions.")
                    else:
                        st.error("Failed to process website")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            logger.info("Chat cleared by user")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response based on source
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ""
                
                # Log the query
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": st.session_state.model,
                    "source_type": st.session_state.selected_source,
                    "query": prompt
                }
                
                with open(os.path.join('logs', f'queries_{datetime.now().strftime("%Y%m%d")}.json'), 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                
                # Process based on source type
                if st.session_state.selected_source == "Direct Chat":
                    response = query_ollama(prompt, st.session_state.model)
                
                elif st.session_state.selected_source == "Wikipedia":
                    if st.session_state.agent:
                        try:
                            response = st.session_state.agent.run(prompt)
                        except Exception as e:
                            response = f"Error from Wikipedia agent: {str(e)}"
                    else:
                        response = "Wikipedia agent has not been initialized. Please initialize it from the sidebar."
                
                elif st.session_state.selected_source == "Database":
                    if st.session_state.agent and st.session_state.db_details:
                        try:
                            response = st.session_state.agent(prompt)
                        except Exception as e:
                            response = f"Error from database agent: {str(e)}"
                    else:
                        response = "Database agent has not been initialized. Please connect to a database from the sidebar."
                
                elif st.session_state.selected_source in ["PDF Document", "Word Document", "Website"]:
                    if st.session_state.conversation_chain:
                        try:
                            result = st.session_state.conversation_chain({"question": prompt, "chat_history": st.session_state.chat_history})
                            st.session_state.chat_history.append((prompt, result["answer"]))
                            response = result["answer"]
                        except Exception as e:
                            response = f"Error from RAG system: {str(e)}"
                    else:
                        source_name = st.session_state.selected_source.lower()
                        response = f"{st.session_state.selected_source} has not been processed. Please process a {source_name} from the sidebar."
                
                st.markdown(response)
                
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Log the response length
                logger.info(f"Generated response of {len(response)} characters")

if __name__ == "__main__":
    main()