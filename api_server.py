from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Request, Header
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from enum import Enum
from typing import Dict
import asyncio
import os
import uvicorn
import torch
import httpx
import base64
import redis
import json
from rq import Queue
import logging
from config import *
from fastapi.responses import FileResponse

app = FastAPI()

# 로깅 설정
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Redis 연결
redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
task_queue = Queue(QUEUE_NAME, connection=redis_conn)

class TrainingStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 학습 상태를 저장할 딕셔너리
training_status: Dict[str, TrainingStatus] = {}

async def send_request_to_new_endpoint(training_result: str, user_id: str, user_name: str, room_name: str):
    encoded_user_name = base64.b64encode(user_name.encode('utf-8')).decode('utf-8')
    encoded_room = base64.b64encode(room_name.encode('utf-8')).decode('utf-8')

    data = {
         "user_name": encoded_user_name,
         "user_id": user_id,
         "room": encoded_room,
         "reply_list": str(training_result)
    }

    logger.info(f"Sending result to DB | user_name: {encoded_user_name}, user_id: {user_id}, room: {encoded_room}")
    async with httpx.AsyncClient() as client:
        response = await client.post(API_ENDPOINT, json=data)
    
    return response.status_code

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.post("/test")
async def upload_file(
    user_name: str = Form(...),
    user_id: str = Form(...),
    file: UploadFile = File(...)):

    user_name_decoded = base64.b64decode(user_name).decode('utf-8')
    file_content = await file.read()

    if len(file_content) == 0:
        logger.warning(f"Empty file uploaded by user: {user_id}")
        return JSONResponse(content={"error": "빈 파일이 업로드되었습니다."}, status_code=400)
    
    # 파일 내용을 임시 저장
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(file_content)

    # Redis 큐에 작업 추가
    job = task_queue.enqueue('worker.process_file', temp_file, user_name_decoded, user_id, job_timeout='1h')
    print(task_queue)

    # 상태 업데이트
    training_status[user_id] = TrainingStatus.QUEUED.value

    logger.info(f"File received and queued for processing. User: {user_id}, Filename: {file.filename}")

    response_data = {
        "filename": file.filename,
        "user_name": user_name_decoded,
        "user_id": user_id,
        "message": "File received and queued for processing",
        "status": training_status[user_id],
        "job_id": job.id
    }

    return JSONResponse(content=jsonable_encoder(response_data), status_code=202)

@app.get("/training-status/{user_id}")
async def get_training_status(user_id: str):
    status = redis_conn.get(f"training_status:{user_id}")
    status = status.decode('utf-8') if status else TrainingStatus.QUEUED.value
    logger.info(f"Training status requested for user: {user_id}, Status: {status}")
    return {"user_id": user_id, "status": status}

@app.get("/")
def read_root():
    return {"Welcome to Model Server Served By fastApi, GCP GPU instance"}

@app.get("/gpu-info")
def get_gpu_info():
    gpu_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }
    logger.info(f"GPU info requested: {gpu_info}")
    return gpu_info

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0",
        port=443, 
        ssl_keyfile="/etc/letsencrypt/live/itsmeweb.net/privkey.pem", 
         ssl_certfile="/etc/letsencrypt/live/itsmeweb.net/fullchain.pem"
    )
