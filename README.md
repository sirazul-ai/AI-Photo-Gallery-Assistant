# Conversational Memory Bot with Image Gallery

An intelligent AI-powered application that combines image management with natural language interaction. Built with FastAPI, ChromaDB, and Google's Gemini model, this bot can understand, store, and retrieve images through conversation.

## 🌟 Features

- **Smart Image Gallery**
  - Upload and manage images
  - Automatic metadata generation
  - Tag-based organization
  - Newest-first display order

- **AI-Powered Search**
  - Natural language image search
  - Image similarity search
  - Combined text and image search
  - Semantic understanding

- **Conversational Interface**
  - Natural language interaction
  - Context-aware responses
  - Image-based discussions
  - Memory of past interactions

- **Advanced Image Processing**
  - Automatic description generation
  - Smart tagging
  - Vector embeddings
  - Similarity matching

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI
- **Database**: ChromaDB (Vector Database)
- **AI Models**: 
  - Google Gemini 1.5
  - CLIP (ViT-B-32)
  
- **Frontend**: 
  - HTML5/CSS3
  - JavaScript
- **Image Processing**: PIL/Pillow
- **Vector Operations**: NumPy, scikit-learn
- **Retrieval-Augmented Generation (RAG)** for improved search results powered by LangChain.

## 📋 Prerequisites

- Python 3.9+
- Google API Key
- 4GB+ RAM
- Storage space for image database

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone [your-repository-url]
   cd memory-bot-ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ENVIRONMENT=development
   PORT=8000
   HOST=0.0.0.0
   ```


## 🚀 Running the Application

1. **Start the Server**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the Application**
   - Main Interface: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

## 📱 Usage Guide

### Image Upload
1. Navigate to the Upload page
2. Select image(s) to upload
3. Wait for automatic processing
4. Review generated descriptions and tags

### Gallery Navigation
- Browse images in chronological order
- Use filters and tags for organization
- Click images for detailed view
- Delete unwanted images

### Image Search
- Text Search: "Show me images of cats"
- Image Search: Upload a similar image
- Combined Search: Use both text and image

### Chat Interface
- Ask about specific images
- Query the image database
- Get image descriptions
- Natural conversation about images

## 🔧 API Endpoints

### Image Management
```
POST /upload - Upload new images
GET /gallery - Retrieve image gallery
DELETE /image/{id} - Remove an image
```

### Search Operations
```
POST /search/text - Text-based search
POST /search/image - Image similarity search
POST /search/multimodal - Combined search
```

### Chat Interface
```
POST /chat - Chat with the bot
GET /chat/history - Retrieve chat history
```

## 📁 Project Structure
```
memory-bot-ai/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── constants.py         # Global constants and messages
│   ├── services/
│   │   ├── chromadb.py     # Vector database operations
│   │   ├── embeddings.py   # Image/text embeddings
│   │   ├── gemini.py       # AI model integration
│   │   └── llm_langchain.py # Language model chain
│   ├── routes/
│   │   └── app_route.py    # API endpoints
│   ├── templates/
│   │   ├── chat.html       # Chat page
│   │   ├── gallery.html    # Image gallery page
│   │   ├── home.html       # Home page
│   │   └── upload.html     # Image upload page
│   └── static/
│       ├── css/
│          └── style.css   # Main stylesheet
│   ├── chromadb/          # Vector database storage
│   ├── uploads/           # Uploaded image storage
│   └── images/            # Processed images
└── requirements.txt       # Project dependencies

```


## 🙏 Acknowledgments

- Google Gemini AI for natural language processing
- OpenAI's CLIP model for image understanding
- ChromaDB for vector storage
- FastAPI community for the excellent framework

