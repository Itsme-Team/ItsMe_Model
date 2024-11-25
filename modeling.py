from user_speech_modeling import user_modeling
from hug import hugging, make_pipeline
import text_preprocessing as txt
import multiprocessing as mp
import time

# 정중체, 상냥체 모델로 학습데이터 생성
def process_1(queue1, df, hug_obj):
    formal_pipeline = make_pipeline(model=hug_obj.formal_model, tokenizer=hug_obj.tokenizer, device=hug_obj.device)
    gentle_pipeline = make_pipeline(model=hug_obj.gentle_model, tokenizer=hug_obj.tokenizer, device=hug_obj.device)
    
    def formal_data():
        for row in df['user']:
            yield str("상냥체 말투로 변환:" + row).strip()
            
    def gentle_data():
        for row in df['user']:
            yield str("정중체 말투로 변환:" + row).strip()
            
    formal_outputs = []
    gentle_outputs = []
    
    for out in formal_pipeline(formal_data()):
        formal_outputs.append([x['generated_text'] for x in out][0])
    
    for out in gentle_pipeline(gentle_data()):
        gentle_outputs.append([x['generated_text'] for x in out][0])
    
    df['formal'] = formal_outputs
    df['gentle'] = gentle_outputs
    queue1.put(df)

# 사용자 말투 학습 + 답장 리스트 생성
def process_2(queue2, df, hug_obj):
    user_model = user_modeling(df, hug_obj)
    test_texts = ["알겠습니다", "가고있습니다", "무슨일있어요?", "나중에 연락드릴게요", "전화로 해주세요", "알바중이에요", "운전중입니다", "회의중입니다"]
    result = []
    user_pipeline = make_pipeline(model=user_model, tokenizer=hug_obj.tokenizer, device=hug_obj.device)
    
    def user_data():
        for txt in test_texts:
            yield f"사용자 말투로 변환:" + txt
            
    for out in user_pipeline(user_data()):
        result.append([x['generated_text'] for x in out][0])
    
    queue2.put(str(result))

def upload(text_file, user_name):
    total_time = 0
    hug_obj = hugging()
    
    # mp.set_start_method('spawn')
    print("<<<카톡 데이터 csv로 변환 중>>>")
    start_time = time.time()
    room_name, df, group, users = txt.txt_to_csv(text_file, user_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"<<<변환 완료>>>{elapsed_time:.2f} 초")
    total_time += elapsed_time
    
    ####################################################
    print(f"방 이름: {room_name}")
    print(f"그룹 채팅: {'예' if group else '아니오'}")
    print(f"사용자 목록: {users}")
    print(f"학습용 데이터 수: {len(df)}")
    #####################################################

    df.columns = ['user']
    df['formal'] = None
    df['gentle'] = None
    print("총 데이터 수:", len(df))
    limit = min(1000, len(df))
    df = df.sample(n=limit)
    print("학습용 데이터 수 (최대 1000 제한):", len(df))
    print("<<<학습 데이터 생성 파이프라인 동작 중>>>")
    start_time = time.time()
    queue1 = mp.Queue()
    p1 = mp.Process(target=process_1, args=(queue1, df, hug_obj))
    p1.start()
    df = queue1.get()
    p1.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"<<<학습 데이터 생성 완료>>> {elapsed_time // 60:.0f} 분 {elapsed_time % 60:.0f} 초")
    total_time += elapsed_time
    
    # process_1이 완료되었으므로 그 결과를 process_2에 넘겨줌
    print("<<<모델 학습 코드 동작 중>>>")
    queue2 = mp.Queue()
    p2 = mp.Process(target=process_2, args=(queue2, df, hug_obj))
    p2.start()
    result = queue2.get()
    p2.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"<<<모델 학습 및 답장 변환 완료>>> {elapsed_time // 60:.0f} 분 {elapsed_time % 60:.0f} 초")
    total_time += elapsed_time

    print("\n변환된 답장 목록:")
    print(result)

    result = result[:30]

    #####################################################
    print("총 소요 시간: ", total_time // 60, "분")
    return room_name, group, users, result
    #####################################################

if __name__ == '__main__' :
    mp.set_start_method('spawn')
    upload('chat/민서좀짧음.txt', '정성호')
