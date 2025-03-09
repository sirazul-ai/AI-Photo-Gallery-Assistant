import os

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
# print(API_KEY)



IMAGE_FOLDER = "./data/uploads"
CHROMA_DB_PATH = "./data/chroma_db"
