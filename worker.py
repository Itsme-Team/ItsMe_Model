import redis
from rq import Worker, Queue, Connection
from rq.job import Job
from modeling import upload
import os
import asyncio
import httpx
import base64
import time
from functools import wraps
import logging
import multiprocessing as mp
from config import *
import psutil
import threading

# 로깅 설정
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
task_queue = Queue(QUEUE_NAME, connection=redis_conn)

def monitor_resources():
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"Resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%")
        time.sleep(60)  # 1분마다 체크

def retry_on_exception(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.critical(f"All {max_retries} attempts failed for function {func.__name__}", exc_info=True)
                        raise
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_on_exception()
def process_file(temp_file, user_name, user_id):
    try:
        # Redis에 학습 시작 상태 저장
        redis_conn.set(f"training_status:{user_id}", "processing")

        # 모델 학습 코드
        room_name, group, users, result = upload(temp_file, user_name)
    
        logger.info(f"Model Learned | result : {result}")
        logger.info(f"Model Learned | room_name : {room_name}")
        logger.info(f"Model Learned | user_id : {user_id}")
        logger.info(f"Model learned | user_name : {user_name}")

        # 임시 파일 삭제
        os.remove(temp_file)

        # 결과를 DB에 전송
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status_code = loop.run_until_complete(send_request_to_new_endpoint(result, user_id, user_name, room_name))
        loop.close()

        logger.info(f"Send Request to New EndPoint | Status code: {status_code}")

        # Redis에 학습 완료 상태 저장
        redis_conn.set(f"training_status:{user_id}", "completed")

    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        # Redis에 학습 실패 상태 저장
        redis_conn.set(f"training_status:{user_id}", "failed")
        raise  # 재시도를 위해 예외를 다시 발생시킴

@retry_on_exception()
async def send_request_to_new_endpoint(training_result: str, user_id: str, user_name: str, room_name: str):
    encoded_user_name = base64.b64encode(user_name.encode('utf-8')).decode('utf-8')
    encoded_room = base64.b64encode(room_name.encode('utf-8')).decode('utf-8')

    data = {
         "user_name": encoded_user_name,
         "user_id": user_id,
         "room": encoded_room,
         "reply_list": str(training_result)
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(API_ENDPOINT, json=data)
    print("DB로 전송완료 | ", response.status_code )
    return response.status_code

def run_worker():
    with Connection(redis_conn):
        worker = Worker([task_queue], default_worker_ttl=3600)
        worker.work()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    threading.Thread(target=monitor_resources, daemon=True).start()
    run_worker()
