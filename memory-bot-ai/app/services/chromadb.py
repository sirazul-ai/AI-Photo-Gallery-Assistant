## chromadb.py
import chromadb
import numpy as np
import os
import uuid
import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from app.services.embeddings import get_image_embedding
from app.services.embeddings import get_text_embedding
from app.config import CHROMA_DB_PATH
import time
from app.constants import (
    ERROR_MESSAGES, SUCCESS_MESSAGES,
    SIMILARITY_THRESHOLD, DEFAULT_SEARCH_LIMIT, TOP_K_RESULTS,
    ALPHA_WEIGHT
)

class ChromaDBService:
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name="image_embeddings")

        # Load CLIP model for text embedding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to(self.device)

    def store_image_in_chromadb(self, file_path, description, tags):
        """Generate a unique ID and store image metadata in ChromaDB."""
        try:
            image_embedding = get_image_embedding(file_path)
            image_id = str(uuid.uuid4())
            timestamp = int(time.time())

            tags_str = ", ".join(tags)
            self.collection.add(
                ids=[image_id],
                embeddings=[image_embedding.tolist()],
                metadatas=[{
                    "id": image_id,
                    "filename": file_path,
                    "tags": tags_str,
                    "description": description,
                    "timestamp": timestamp
                }]
            )
            print(SUCCESS_MESSAGES["STORE_SUCCESS"].format(image_id))
            return image_id
        except Exception as e:
            print(ERROR_MESSAGES["STORE_ERROR"].format(str(e)))
            return None

    def fetch_images_from_chromadb(self, limit=DEFAULT_SEARCH_LIMIT):
        """Fetch stored images & descriptions from ChromaDB in descending order."""
        try:
            count = self.collection.count()
            if count == 0:
                return {"error": ERROR_MESSAGES["NO_IMAGES"]}

            results = self.collection.get(
                include=["metadatas"],
                limit=limit
            )
            
            if "metadatas" not in results or "ids" not in results:
                return {"error": ERROR_MESSAGES["NO_METADATA"]}

            image_data = [
                {
                    "id": meta["id"],
                    "filename": meta["filename"],
                    "tags": meta.get("tags", []),
                    "description": meta["description"],
                    "timestamp": meta.get("timestamp", 0)
                }
                for meta in results["metadatas"]
            ]
            
            image_data.sort(key=lambda x: x["timestamp"], reverse=True)
            return image_data
        except Exception as e:
            return {"error": ERROR_MESSAGES["FETCH_ERROR"].format(str(e))}

    def get_image_description(self, image_id):
        """Retrieve image description and tags using its ID."""
        try:
            results = self.collection.get(ids=[image_id], include=["metadatas"])
            if "metadatas" not in results or len(results["metadatas"]) == 0:
                return {"error": ERROR_MESSAGES["IMAGE_NOT_FOUND"].format(image_id)}
            
            metadata = results["metadatas"][0]
            return {
                "description": metadata["description"],
                "tags": metadata.get("tags", [])
            }
        except Exception as e:
            return {"error": str(e)}

    def text_to_image_search(self, query_text, top_k=TOP_K_RESULTS):
        """Search for images based on a text query using cosine similarity."""
        try:
            query_embedding = get_text_embedding(query_text).reshape(1, -1)

            results = self.collection.get(include=["embeddings", "metadatas"])
            
            if results["embeddings"] is None or len(results["embeddings"]) == 0:
                return {"error": ERROR_MESSAGES["NO_EMBEDDINGS"]}

            stored_embeddings = np.array(results["embeddings"])
            similarity_scores = cosine_similarity(query_embedding, stored_embeddings)[0]
            top_indices = np.argsort(similarity_scores)[::-1][:top_k]

            matched_images = [
                {
                    "id": results["metadatas"][i]["id"], 
                    "filename": results["metadatas"][i]["filename"],
                    "tags": results["metadatas"][i]["tags"],
                    "description": results["metadatas"][i]["description"],
                    "similarity": round(float(similarity_scores[i]), 4)
                }
                for i in top_indices
            ]
            return {"query": query_text, "results": matched_images}
        except Exception as e:
            return {"error": str(e)}

    def image_to_image_search(self, image_path, top_k=TOP_K_RESULTS):
        """Find similar images based on an uploaded image."""
        try:
            query_embedding = get_image_embedding(image_path).reshape(1, -1)

            results = self.collection.get(include=["embeddings", "metadatas"])
            
            if results["embeddings"] is None or len(results["embeddings"]) == 0:
                return {"error": ERROR_MESSAGES["NO_EMBEDDINGS"]}

            stored_embeddings = np.array(results["embeddings"])
            similarity_scores = cosine_similarity(query_embedding, stored_embeddings)[0]
            top_indices = np.argsort(similarity_scores)[::-1][:top_k]

            matched_images = [
                {
                    "id": results["metadatas"][i]["id"], 
                    "filename": results["metadatas"][i]["filename"],
                    "tags": results["metadatas"][i]["tags"],
                    "description": results["metadatas"][i]["description"],
                    "similarity": round(float(similarity_scores[i]), 4)
                }
                for i in top_indices
            ]
            return {"query_image": image_path, "results": matched_images}
        except Exception as e:
            return {"error": str(e)}

    def multimodal_search(self, image_path, query_text, top_k=TOP_K_RESULTS, alpha=ALPHA_WEIGHT):
        """Search for images based on both an image and a text query using a weighted similarity approach."""
        try:
            # Generate embeddings
            image_embedding = get_image_embedding(image_path).reshape(1, -1)
            text_embedding = get_text_embedding(query_text).reshape(1, -1)

            # Normalize embeddings for numerical stability
            image_embedding = normalize(image_embedding)
            text_embedding = normalize(text_embedding)

            # Weighted combination of embeddings
            combined_embedding = alpha * image_embedding + (1 - alpha) * text_embedding

            # Fetch stored embeddings
            results = self.collection.get(include=["embeddings", "metadatas"])
            if results["embeddings"] is None or len(results["embeddings"]) == 0:
                return {"error": ERROR_MESSAGES["NO_EMBEDDINGS"]}

            stored_embeddings = np.array(results["embeddings"])
            stored_embeddings = normalize(stored_embeddings)

            # Compute similarity scores
            similarity_scores = cosine_similarity(combined_embedding, stored_embeddings)[0]
            top_indices = np.argsort(similarity_scores)[::-1][:top_k]

            # Retrieve matched images
            matched_images = [
                {
                    "id": results["metadatas"][i]["id"],
                    "filename": results["metadatas"][i]["filename"],
                    "tags": results["metadatas"][i]["tags"],
                    "description": results["metadatas"][i]["description"],
                    "similarity": round(float(similarity_scores[i]), 4)
                }
                for i in top_indices
            ]
            return {"query_text": query_text, "query_image": image_path, "results": matched_images}
        except Exception as e:
            return {"error": str(e)}

    def delete_image_from_chromadb(self, image_id):
        """Delete an image entry from ChromaDB based on its ID."""
        try:
            # Check if the image exists
            results = self.collection.get(ids=[image_id])
            if "metadatas" not in results or len(results["metadatas"]) == 0:
                return {"error": ERROR_MESSAGES["IMAGE_NOT_FOUND"].format(image_id)}
            
            # Delete the image entry
            self.collection.delete(ids=[image_id])
            return {"success": SUCCESS_MESSAGES["DELETE_SUCCESS"].format(image_id)}
        except Exception as e:
            return {"error": str(e)}
        