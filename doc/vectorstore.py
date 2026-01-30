import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from huggingface_hub import InferenceClient
import asyncio

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
hf_client = InferenceClient(token=HF_API_TOKEN)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
UPLOAD_DIR ="./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud='aws',region=PINECONE_ENV)
existing_index=[i["name"] for i in pc.list_indexes()]

# 1536 in google gen ai and 384 in HF all-MiniLM-L6-v2
if PINECONE_INDEX_NAME not in existing_index:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("Waiting for index to be ready in Pinecone...")
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# async def load_vetorstore(uploaded_files,role:str,doc_id:str):
#     # embed_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     for file in uploaded_files:
#         save_path=Path(UPLOAD_DIR)/file.filename
#         with open(save_path,"wb") as f:
#             f.write(file.file.read())
        
#         loader=PyPDFLoader(str(save_path))
#         documents=loader.load()

#         splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
#         chunks=splitter.split_documents(documents)

#         texts=[chunk.page_content for chunk in chunks]
#         ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
#         metadata=[
#             {
#             "source":file.filename,
#             "role":role,
#             "doc_id":doc_id,
#             "page":chunk.metadata.get("page",0)
#             } 
#             for i,chunk in enumerate(chunks)
#         ]

#         print(f"Embedding {len(texts)} chunk ...")
#         # embeddings=await asyncio.to_thread(embed_model.embed_documents,texts)
#         embeddings = []
#         for text in texts:
#             result = await asyncio.to_thread(
#                 hf_client.text_embeddings,
#                 model="sentence-transformers/all-MiniLM-L6-v2",
#                 input=text
#             )
#             embeddings.append(result['embedding'])  # the API returns dict with 'embedding' key

#         print('Uploading to Pinecone ...')
#         with tqdm(total=len(embeddings), desc="Upserting in Pinecone") as progress:
#             index.upsert(vectors=zip(ids,embeddings,metadata))
#             progress.update(len(embeddings))

#         print(f"Upload comlete for {file.filename}")

async def load_vetorstore(uploaded_files, role: str, doc_id: str):
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        
        loader = PyPDFLoader(str(save_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadata = [
            {
                "source": file.filename,
                "role": role,
                "doc_id": doc_id,
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content  #Store the text
            } for i, chunk in enumerate(chunks)
        ]

        print(f"Embedding {len(texts)} chunks ...")
        embeddings = []
        
        for text in tqdm(texts, desc="Creating embeddings"):
            emb = await asyncio.to_thread(
                hf_client.feature_extraction,
                text,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            #Convert numpy array to list
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            elif isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                emb = emb[0]
            
            embeddings.append(emb)

        print("Uploading to Pinecone ...")
        batch_size = 100
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Upserting to Pinecone"):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            
            vectors = list(zip(batch_ids, batch_embeddings, batch_metadata))
            index.upsert(vectors=vectors)

        print(f"Upload complete for {file.filename}")