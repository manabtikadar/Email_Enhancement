import os
import warnings
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# Set path to save/load FAISS index
FAISS_INDEX_PATH = r"C:\Users\manab\Downloads\Manab_Goflow\Manab_Goflow\Goflow_Data"
FAISS_INDEX_NAME = "index"  # Keep consistent naming

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# Load API key for Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Initialize the Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

# URLs to GoFlow documentation
goflow_urls = [
    "https://goflow.com/docs?utm_source=chatgpt.com",
    "https://goflow.com/api-spec?utm_source=chatgpt.com",
    "https://goflow.com/docs/purchasing/inventory-forecasting?utm_source=chatgpt.com",
    "https://goflow.com/docs/general/managing-users-and-roles?utm_source=chatgpt.com",
    "https://goflow.com/docs/listings/creating-and-managing-listings?utm_source=chatgpt.com",
    "https://goflow.com/docs/store/create-and-manage-stores?utm_source=chatgpt.com",
    "https://goflow.com/docs/products/creating-and-managing-products?utm_source=chatgpt.com",
]

# Path to FAISS files
index_file_path = os.path.join(FAISS_INDEX_PATH, f"{FAISS_INDEX_NAME}.faiss")
pkl_file_path = os.path.join(FAISS_INDEX_PATH, f"{FAISS_INDEX_NAME}.pkl")

if os.path.exists(index_file_path) and os.path.exists(pkl_file_path):
    print("✅ Loading vector store from local disk...")
    vectorstore = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH,
        embeddings=embeddings,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True
    )
else:
    print("⚙️ Creating new vector store...")

    # Load and split docs
    web_loader = WebBaseLoader(goflow_urls)
    web_docs = web_loader.load()

    # Optional: Add PDF loading if needed in the future
    # pdf_folder = "path/to/pdfs"
    # goflow_pdfs = []
    # for file in os.listdir(pdf_folder):
    #     if file.endswith(".pdf"):
    #         loader = PyPDFLoader(os.path.join(pdf_folder, file))
    #         goflow_pdfs.extend(loader.load())

    all_docs = web_docs  # + goflow_pdfs (if used)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(all_docs)

    # Create and save vector store
    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=embeddings
    )
    vectorstore.save_local(FAISS_INDEX_PATH, index_name=FAISS_INDEX_NAME)
    print("💾 Saved vector store locally.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})