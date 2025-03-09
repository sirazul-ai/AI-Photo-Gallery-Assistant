# Import necessary modules


from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app.routes.app_route import app_router
import os

# Initialize FastAPI app
app = FastAPI(
    title="Conversational Memory Bot - AI",
    description="An AI-powered photo gallery assistant using CLIP, ChromaDB, and FastAPI.",
    version="1.0.0"
)

# Define constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of main.py
UPLOAD_FOLDER = os.path.join(BASE_DIR, "..", "data", "uploads")  # Absolute path

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/data/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="data")  # Now guaranteed to exist


app.include_router(app_router)  # Include your custom routes in here. 

