import torch

# Device and Model Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'ViT-B-32'
MODEL_PRETRAINED = 'openai'
LLM_MODEL = "gemini-1.5-flash"

# Path Constants
UPLOAD_FOLDER = "data/uploads"
CHROMA_DB_PATH = "data/chromadb"
# IMAGE_FOLDER = "data/images"

# Gemini Constants
GEMINI_MODEL = 'gemini-1.5-flash'
MAX_IMAGE_SIZE = (512, 512)
MAX_TAGS = 8

# Prompt Templates
PROMPT_TEMPLATES = {
    "QUERY_CLASSIFIER": """
Given this user input, classify it as one of these types:
- IMAGE_SEARCH: Queries about finding or describing images
- GENERAL_KNOWLEDGE: Questions about facts, information, or explanations
- CONVERSATION: General chat or follow-up questions

User Input: {input}

Return only one word: IMAGE_SEARCH, GENERAL_KNOWLEDGE, or CONVERSATION
""",
    "IMAGE_SEARCH": """
You are a helpful gallery assistant. Present only the image paths and a brief summary.

User Query: {query}

Available Images:
{results}

Return ONLY in this format:
PATHS:
[path1.jpg]
[path2.jpg]
[path3.jpg]

SUMMARY:
A single paragraph summarizing all found images together.
""",
    "GENERAL_KNOWLEDGE": """
You are a knowledgeable assistant. Answer the user's question accurately and concisely.

Conversation History:
{history}

Current Question: {query}

Provide a clear, factual response that:
1. Directly answers the question
2. Includes relevant context if helpful
3. Acknowledges any limitations in knowledge
4. Stays focused on the specific query
""",
    "DESCRIPTION": "Describe the contents of this image in one paragraph of 2-4 sentences.",
    "TAGS": "Provide eight relevant tags (comma-separated) for this image."
}

# Error Messages
ERROR_MESSAGES = {
    "STORE_ERROR": "Error storing image: {}",
    "FETCH_ERROR": "Error fetching images: {}",
    "NO_IMAGES": "No images found in ChromaDB.",
    "NO_METADATA": "No metadata or IDs found in ChromaDB.",
    "DELETE_ERROR": "Error deleting image: {}",
    "IMAGE_NOT_FOUND": "Image with ID {} not found in ChromaDB.",
    "EMBEDDING_ERROR": "Error generating embedding: {}",
    "API_KEY_MISSING": "GOOGLE_API_KEY is missing in your environment variables!",
    "METADATA_ERROR": "Error generating metadata: {}",
    "IMAGE_FILE_NOT_FOUND": "Image file not found: {}",
    "NO_DESCRIPTION": "No description generated.",
    "NO_INPUT": "Please provide a question or upload an image to continue our conversation.",
    "PROCESSING_ERROR": "Error in process_user_input: {}",
    "GENERAL_ERROR": "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question. Error: {}",
    "NO_EMBEDDINGS": "No embeddings found in ChromaDB."
}

# Success Messages
SUCCESS_MESSAGES = {
    "STORE_SUCCESS": "Image stored in ChromaDB with ID: {}",
    "DELETE_SUCCESS": "Image with ID {} has been deleted from ChromaDB."
}

# Search Constants
SIMILARITY_THRESHOLD = 0.1
DEFAULT_SEARCH_LIMIT = 1000
TOP_K_RESULTS = 3
ALPHA_WEIGHT = 0.5

# Search Formats
SEARCH_FORMATS = {
    "NO_RESULTS": "No matching images found in the gallery.",
    "IMAGE_FORMAT": "Image: [{filename}]\nDescription: {description}\nTags: {tags}\nSimilarity: {similarity}\n"
}