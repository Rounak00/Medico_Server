from fastapi import UploadFile,File,Form,Depends,HTTPException,APIRouter
from auth.routes import authenticate
from .vectorstore import load_vetorstore
import uuid

router=APIRouter()

@router.post("/upload_docs")
async def upload_docs(
    user=Depends(authenticate),
    file:UploadFile=File(...),
    role:str=Form(...),
):  
    if user["role"] != "admin":
        raise HTTPException(status_code=403,detail="Only admin can upload files")
    doc_id=str(uuid.uuid4())
    await load_vetorstore([file],role,doc_id)
    return {"message":f"{file.filename} uploaded and processed successfully","doc_id":doc_id,"accesible_to":role}
