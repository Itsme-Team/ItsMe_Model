import os

# Redis 설정
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# 큐 설정
QUEUE_NAME = 'model_training'

# 재시도 설정
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', 5))  # seconds

# API 엔드포인트
API_ENDPOINT = "https://itsmeweb.site/api/model_result/"

# 로깅 설정
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 워커 설정
WORKER_RATIO = float(os.getenv('WORKER_RATIO', 0.75))  # CPU 코어 대비 워커 비율