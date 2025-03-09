from app.constants import (
    DEVICE, MODEL_NAME, MODEL_PRETRAINED,
    ERROR_MESSAGES
)
import torch
import open_clip
from PIL import Image

# Load model globally
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.to(DEVICE)

def get_image_embedding(image_path):
    """Generate image embedding using CLIP."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model.encode_image(image)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        raise Exception(ERROR_MESSAGES["EMBEDDING_ERROR"].format(str(e)))

def get_text_embedding(text):
    """Generate text embedding using CLIP."""
    try:
        tokens = tokenizer([text]).to(DEVICE)
        with torch.no_grad():
            text_embedding = model.encode_text(tokens)
        return text_embedding.cpu().numpy().flatten()
    except Exception as e:
        raise Exception(ERROR_MESSAGES["EMBEDDING_ERROR"].format(str(e)))
