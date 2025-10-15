from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import time

from auth.routes import router as auth_router
from doc.routes import router as doc_router
from chat.routes import router as chat_router

app=FastAPI()
start_time = time.time()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(doc_router)
app.include_router(chat_router)


@app.get("/health")
async def health_check():
    uptime = round(time.time() - start_time, 2)
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": uptime
    }


# def main():
#     print("Hello from server!")


# if __name__ == "__main__":
#     main()
