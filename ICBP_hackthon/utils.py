import os
import google.generativeai as genai
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key="AIzaSyCaFaJwTQ9rh0mSUd4YaQxG3fuDGjKrj1Q")

class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.model = "models/text-embedding-004"
    
    def embed_documents(self, texts):
        """Embed a list of texts."""
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        embeddings = []
        for text in texts:
            if not text.strip():
                continue
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                logger.error(f"Error embedding document: {str(e)}")
                raise
        return embeddings
    
    def embed_query(self, text):
        """Embed a single piece of text."""
        if not text.strip():
            raise ValueError("Empty query text provided")
        
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

def ask_gemini(prompt):
    """
    Sends a prompt to the Gemini model and returns the response.
    """
    if not prompt or not prompt.strip():
        return "Error: Please provide a non-empty prompt."
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error in ask_gemini: {str(e)}")
        return f"Error generating response: {str(e)}"

def rag_with_url(target_url, prompt):
    """
    Retrieves relevant documents from a target URL and generates an AI response based on the prompt.
    """
    if not target_url.strip() or not prompt.strip():
        return "Error: Please provide both URL and prompt."

    try:
        loader = WebBaseLoader(target_url)
        raw_document = loader.load()
        
        if not raw_document:
            return "Error: No content could be loaded from the URL."

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len
        )

        splited_document = text_splitter.split_documents(raw_document)
        
        if not splited_document:
            return "Error: No content could be extracted from the document."

        embeddings = GeminiEmbeddings()
        vector_store = FAISS.from_documents(splited_document, embeddings)

        retriever = vector_store.as_retriever()
        relevant_documents = retriever.get_relevant_documents(prompt)

        if not relevant_documents:
            return "No relevant information found. Please try a different query."

        final_prompt = f"""Based on the following context, please answer this question: {prompt}

Context:
{' '.join([doc.page_content for doc in relevant_documents])}

Answer:"""

        return ask_gemini(final_prompt)
    
    except Exception as e:
        logger.error(f"Error in rag_with_url: {str(e)}")
        return f"Error processing request: {str(e)}"

def rag_with_pdf(file_path, prompt):
    """
    Performs RAG using a PDF file with Gemini embeddings and generation.
    """
    if not file_path or not prompt.strip():
        return "Error: Please provide both PDF file and prompt.", []

    try:
        loader = PyPDFLoader(file_path)
        raw_document = loader.load()
        
        if not raw_document:
            return "Error: No content could be loaded from the PDF.", []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len
        )

        splited_document = text_splitter.split_documents(raw_document)
        
        if not splited_document:
            return "Error: No content could be extracted from the PDF.", []

        embeddings = GeminiEmbeddings()
        vector_store = FAISS.from_documents(splited_document, embeddings)

        retriever = vector_store.as_retriever()
        relevant_documents = retriever.get_relevant_documents(prompt)

        if not relevant_documents:
            return "No relevant information found. Please try a different query.", []

        final_prompt = f"""Based on the following context, please answer this question: {prompt}

Context:
{' '.join([doc.page_content for doc in relevant_documents])}

Answer:"""

        return ask_gemini(final_prompt), relevant_documents
    
    except Exception as e:
        logger.error(f"Error in rag_with_pdf: {str(e)}")
        return f"Error processing request: {str(e)}", []