import pdfplumber
from docx import Document
import nltk
from sentence_transformers import SentenceTransformer

# Load sentence transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def parse_document(file_path: str) -> str:
    """Parse the document and return text."""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        raise ValueError("Unsupported file type")

def chunk_document(text: str, max_tokens=512):
    """Chunk document text into smaller pieces."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = sentence.split()
        if current_length + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks