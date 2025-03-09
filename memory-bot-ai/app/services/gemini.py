import os
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from app.config import API_KEY
from app.constants import (
    GEMINI_MODEL,
    MAX_IMAGE_SIZE,
    MAX_TAGS,
    PROMPT_TEMPLATES,
    ERROR_MESSAGES
)

# Validate API key
if not API_KEY:
    raise ValueError(ERROR_MESSAGES["API_KEY_MISSING"])

# Configure Gemini with your API key
genai.configure(api_key=API_KEY)

def generate_image_metadata(image_path):
    """Generate a short image description and eight tags using Google Gemini AI."""
    try:
        # Open and resize image
        image = Image.open(image_path).convert("RGB")
        image.thumbnail(MAX_IMAGE_SIZE)

        # Generate content using Gemini
        model = genai.GenerativeModel(GEMINI_MODEL)
        response_desc = model.generate_content([PROMPT_TEMPLATES["DESCRIPTION"], image])
        response_tags = model.generate_content([PROMPT_TEMPLATES["TAGS"], image])

        # Extract response text
        description = response_desc.text.strip() if hasattr(response_desc, "text") else ERROR_MESSAGES["NO_DESCRIPTION"]
        tags = response_tags.text.strip().split(",") if hasattr(response_tags, "text") else []
        tags = [tag.strip() for tag in tags][:MAX_TAGS]  # Limit to max tags

        return description, tags
    except Exception as e:
        return ERROR_MESSAGES["METADATA_ERROR"].format(str(e)), []


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    test_image_path = "path/to/your/image.jpg"
    
    if os.path.exists(test_image_path):
        description, tags = generate_image_metadata(test_image_path)
        print("Description:", description)
        print("Tags:", tags)
    else:
        print(ERROR_MESSAGES["IMAGE_FILE_NOT_FOUND"].format(test_image_path))

        