import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
hf_client = InferenceClient(token=HF_API_TOKEN)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Chat model - UPDATED MODEL NAME
llm = ChatGroq(
    temperature=0.3,
    model_name="llama-3.1-8b-instant",  # ✅ Updated model
    groq_api_key=GROQ_API_KEY
)

prompt = PromptTemplate.from_template(
    """You are a helpful healthcare AI assistant. Answer the following question based only on the provided context.
    
Question: {question}

Context: {context}

Include the document source if relevant in your answer."""
)

rag_chain = prompt | llm


async def answer_query(query: str, user_role: str):
    """
    Query the RAG system with role-based filtering.
    
    Args:
        query: User's question
        user_role: Role to filter documents (e.g., 'doctor', 'patient')
    
    Returns:
        Dictionary with answer and sources
    """
    # Generate embedding using Hugging Face MiniLM (384 dimensions)
    embedding_result = await asyncio.to_thread(
        hf_client.feature_extraction,
        query,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Convert to list - handle both nested lists and numpy arrays
    if hasattr(embedding_result, 'tolist'):
        # It's a numpy array
        embedding = embedding_result.tolist()
    elif isinstance(embedding_result, list):
        # It's already a list
        embedding = embedding_result[0] if isinstance(embedding_result[0], list) else embedding_result
    else:
        embedding = list(embedding_result)
    
    # Ensure it's a flat list
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]

    # Query Pinecone
    results = await asyncio.to_thread(
        index.query,
        vector=embedding,
        top_k=5,
        include_metadata=True
    )

    filtered_contexts = []
    sources = set()

    # Filter results by role
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        
        # Check if role matches
        if metadata.get("role") == user_role:
            text_content = metadata.get("text", "")
            if text_content:
                filtered_contexts.append(text_content)
                sources.add(metadata.get("source", "Unknown"))

    # Handle no results case
    if not filtered_contexts:
        return {
            "answer": f"No relevant information found for role: {user_role}. Please ensure documents are uploaded for this role.",
            "sources": []
        }

    # Combine contexts
    docs_text = "\n\n".join(filtered_contexts)
    
    # Generate answer using LLM
    final_answer = await asyncio.to_thread(
        rag_chain.invoke,
        {"question": query, "context": docs_text}
    )

    return {
        "answer": final_answer.content,
        "sources": list(sources)
    }