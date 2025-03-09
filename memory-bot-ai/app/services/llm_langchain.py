# /llm_langchain.py

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from app.config import API_KEY 
from app.services.chromadb import ChromaDBService
from app.constants import (
    PROMPT_TEMPLATES,
    ERROR_MESSAGES,
    SEARCH_FORMATS,
    LLM_MODEL
)

# Initialize services
chroma_service = ChromaDBService()
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=API_KEY)
memory = ConversationBufferMemory(return_messages=True)

# Create prompts using constants
query_classifier_prompt = PromptTemplate(
    input_variables=["input"],
    template=PROMPT_TEMPLATES["QUERY_CLASSIFIER"]
)

image_search_prompt = PromptTemplate(
    input_variables=["query", "results"],
    template=PROMPT_TEMPLATES["IMAGE_SEARCH"]
)

general_knowledge_prompt = PromptTemplate(
    input_variables=["query", "history"],
    template=PROMPT_TEMPLATES["GENERAL_KNOWLEDGE"]
)

# Create chains
query_classifier_chain = LLMChain(llm=llm, prompt=query_classifier_prompt)
image_search_chain = LLMChain(llm=llm, prompt=image_search_prompt)
general_knowledge_chain = LLMChain(llm=llm, prompt=general_knowledge_prompt)

def format_search_results(results):
    """Format search results for the prompt."""
    if not results or "error" in results or not results.get("results"):
        return SEARCH_FORMATS["NO_RESULTS"]
    
    formatted_results = []
    for img in results["results"][:3]:  # Limit to top 3 results
        formatted_results.append(
            SEARCH_FORMATS["IMAGE_FORMAT"].format(
                filename=img['filename'],
                description=img['description'],
                tags=img['tags'],
                similarity=img.get('similarity', 'N/A')
            )
        )
    
    return "\n".join(formatted_results)

def process_user_input(text: str = None, image_path: str = None) -> str:
    """Process user input and generate an appropriate response."""
    try:
        # Handle empty input
        if not text and not image_path:
            return ERROR_MESSAGES["NO_INPUT"]

        # Get conversation history
        history = memory.load_memory_variables({})["history"]
        
        # Default text for image-only queries
        input_text = text or "Describe this image"

        # Determine query type
        query_type = query_classifier_chain.invoke({"input": input_text})["text"] if text else "IMAGE_SEARCH"

        # Handle different query types
        if query_type == "IMAGE_SEARCH" or image_path:
            # Perform appropriate search
            if image_path and text:
                results = chroma_service.multimodal_search(image_path, text)
            elif image_path:
                results = chroma_service.image_to_image_search(image_path)
            else:
                results = chroma_service.text_to_image_search(text)

            # Format results and generate response
            formatted_results = format_search_results(results)
            response = image_search_chain.invoke({
                "query": input_text,
                "results": formatted_results
            })["text"]

        elif query_type == "GENERAL_KNOWLEDGE":
            # Handle general knowledge questions
            response = general_knowledge_chain.invoke({
                "query": input_text,
                "history": history
            })["text"]

        else:  # CONVERSATION
            # Handle general conversation
            response = llm.predict(
                f"Conversation History:\n{history}\n\nUser: {input_text}\nAssistant:"
            )

        # Update conversation memory
        memory.save_context({"input": input_text}, {"output": response})
        
        return response

    except Exception as e:
        print(ERROR_MESSAGES["PROCESSING_ERROR"].format(str(e)))
        return ERROR_MESSAGES["GENERAL_ERROR"].format(error=str(e))

def clear_memory():
    """Clear the conversation memory."""
    global memory
    memory = ConversationBufferMemory(return_messages=True)