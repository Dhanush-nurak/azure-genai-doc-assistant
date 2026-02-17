import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

class DocAssistant:
    def __init__(self):
        # Initialize Azure components
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_EMBEDDING_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION")
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=0
        )
        self.vector_store = None

    def ingest_document(self, pdf_path):
        """Loads a PDF and creates a vector store."""
        print(f"Loading {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        print("Creating embeddings...")
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print("Ingestion complete.")

    def ask_question(self, query):
        """Queries the vector store."""
        if not self.vector_store:
            return "Please ingest a document first."
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        return qa_chain.run(query)