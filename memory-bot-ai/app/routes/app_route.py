# Import necessary modules
from fastapi import APIRouter, Request, File, UploadFile, HTTPException, Form, Query, Body
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional

import os
import shutil

# Import services & models
from app.services.chromadb import ChromaDBService
from app.services.embeddings import get_image_embedding
from app.services.gemini import generate_image_metadata
from app.services.llm_langchain import process_user_input  # Import the new service

# Define constants
UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Update the uploads directory mounting with absolute path
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "uploads")
# UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/uploads"))


# Initialize FastAPI router
app_router = APIRouter()

# Initialize services
chroma_service = ChromaDBService()

# Setup templates for rendering HTML responses
templates = Jinja2Templates(directory="app/templates")

# Serve static files
static_files = StaticFiles(directory="app/static")

# Update the uploads directory mounting with absolute path
# app_router.mount("/data", StaticFiles(directory=UPLOADS_DIR), name="data")

# ------------------------ ROUTES ----------------------------------------------------------------

@app_router.get("/", tags=["UI"])
def home(request: Request):
    """Render the homepage."""
    return templates.TemplateResponse("home.html", {"request": request})

@app_router.get("/gallery", tags=["UI"])
def gallery(request: Request):
    """Render the gallery page."""
    return templates.TemplateResponse("gallery.html", {"request": request})

@app_router.get("/upload", tags=["UI"])
def upload(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse("upload.html", {"request": request})

@app_router.get("/chat", tags=["UI"])
def chat(request: Request):
    """Render the chat page."""
    return templates.TemplateResponse("chat.html", {"request": request})

#  Search images by text query.
@app_router.get("/search-text", tags=["Search"])
def search_by_text(query: str):
    """Search images by text query."""
    results = chroma_service.text_to_image_search(query)
    return JSONResponse(content={"results": results})

@app_router.post("/search-image", tags=["Search"])
async def search_by_image(file: UploadFile = File(...)):
    """Search for similar images using an uploaded image."""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = chroma_service.image_to_image_search(file_path)
    # clean up temporary files
    os.remove(file_path)
    
    return JSONResponse(content={"results": results})


@app_router.post("/multimodal-search/", tags=["Search"])
async def multimodal_search(image: UploadFile = File(...), query_text: str = Form(default="")): 
    #  Important Note: If you're uploading both a file and a text field, you must use Form(...) for the text field because multipart/form-data is required for file uploads.
    # Best when sending data from an HTML form or when uploading a file (image + text query).
    """Search for images using both an uploaded image and a text query."""
    try:
        # Save the uploaded image temporarily
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Perform multimodal search
        results = chroma_service.multimodal_search(image_path, query_text)

        # Clean up: Delete the temporary uploaded image
        os.remove(image_path)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# #################################################################


@app_router.get("/description/{image_id}", tags=["Image Metadata"])
def get_image_description(image_id: str):
    """Retrieve the description of an image using its unique ID."""
    description = chroma_service.get_image_description(image_id)
    return JSONResponse(content={"description": description})


@app_router.post("/upload-batch", tags=["Upload"])
async def upload_images(files: list[UploadFile] = File(...)):
    """Upload multiple images, generate metadata, store them in ChromaDB, and return their IDs & descriptions."""
    uploaded_images = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Save the file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate  description
        description, tags = generate_image_metadata(file_path)

        filename = f"{UPLOAD_FOLDER}/{file.filename}"
        print(f"Filename: {filename}")

        # Store image in ChromaDB with a unique ID
        image_id = chroma_service.store_image_in_chromadb(filename, description, tags)
        
        # Append details to the response list
        uploaded_images.append({
            "image_id": image_id,
            "filename": file.filename,
            "description": description,
            "tags": tags
        })

    return JSONResponse(content={"uploaded_images": uploaded_images})


@app_router.get("/fetch-images", tags=["Fetch"])
def fetch_images():
    """Fetch all stored images and descriptions from ChromaDB in descending order."""
    try:
        images = chroma_service.fetch_images_from_chromadb()
        if isinstance(images, list):
            # Images are already sorted by timestamp in ChromaDBService
            return JSONResponse(content={"images": images})
        else:
            return JSONResponse(content={"images": [], "error": "No images found"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching images: {str(e)}"}
        )



@app_router.delete("/delete_image/{image_id}", tags=["Images"])
async def delete_image(image_id: str):
    try:
        # First, get the image details from ChromaDB
        image_details = chroma_service.get_image_description(image_id)
        if "error" in image_details:
            raise HTTPException(status_code=404, detail="Image not found in database")

        # Get the filename from ChromaDB
        filename = None
        results = chroma_service.collection.get(ids=[image_id], include=["metadatas"])
        if results and results["metadatas"]:
            filename = results["metadatas"][0].get("filename")

        # Delete from ChromaDB
        result = chroma_service.delete_image_from_chromadb(image_id)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Delete the physical file if it exists
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Warning: Could not delete file {filename}: {str(e)}")
                # Continue even if file deletion fails, as the DB entry is already removed

        return {"message": "Image deleted successfully", "id": image_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@app_router.post("/chat_bot", tags=["Chat"])
async def chat_bot(
    request: Request,
    text: Optional[str] = Body(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Handle chat requests with text, image, or both, and return a conversational response.
    Parameters:
    - text (str, optional): Text query from the user
    - image (UploadFile, optional): Image file for visual queries
    """
    try:
        # Get text from either JSON body or form data
        if request.headers.get("content-type") == "application/json":
            body = await request.json()
            text = body.get("text", "")
        elif not text:  # If text wasn't provided in body
            form = await request.form()
            text = form.get("text", "")
            image = form.get("image")

        # Validate input
        if not text and not image:
            return JSONResponse(
                content={"response": "Please provide either a question or an image to continue our conversation."}
            )

        image_path = None
        if image and hasattr(image, 'filename') and image.filename:
            # Save uploaded image temporarily
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

        # Process input and get response
        response = process_user_input(text=text, image_path=image_path)

        # Clean up temporary image file
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        return JSONResponse(content={"response": response})

    except Exception as e:
        # Clean up on error
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        print(f"Error in chat_bot: {str(e)}")  # Add logging
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    

