import pandas as pd
import re
from kss import split_sentences
import multiprocessing as mp
import chardet
import os

def remove_some(text):
    if("이모티콘" in text or "사진" in text):
        text=text.replace("이모티콘", "").replace("사진","").replace("샵검색:", "")
    text = re.sub(r'\[.*?\]\s*', '', text)
    text = re.sub(r'http\S+', '', text)
    return text

## 새로추가 함수
def regexp(sentence):
    pattern1 = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+")  # 한글 자모음만 반복되면 공백으로 대체
    pattern2 = re.compile(r":\)|[@#$^\*\(\)\[\]\{\}<>\/\"'=+\|_]+")  # 특수문자 공백으로 대체 (~, !, %, &, -, ,, ., ;, :, ?는 유지)
    # 특수문자 공백으로 대체 (~, !, %, &, -, ,, ., ;, :, ?는 유지)
    # pattern3 = re.compile(r"([^\d])\1{2,}")  # 숫자를 제외한 동일한 문자 3개 이상이면 공백으로 대체
    pattern3 = re.compile(  # 이모티콘 공백으로 대체
        "["                               
        "\U0001F600-\U0001F64F"  # 감정 관련 이모티콘
        "\U0001F300-\U0001F5FF"  # 기호 및 픽토그램
        "\U0001F680-\U0001F6FF"  # 교통 및 지도 기호
        "\U0001F1E0-\U0001F1FF"  # 국기 이모티콘
        "]+", flags=re.UNICODE
    )
    new_sent1 = pattern1.sub(' ', sentence)
    new_sent2 = pattern2.sub(' ', new_sent1)
    new_sent3 = pattern3.sub(' ', new_sent2)
    # new_sent4 = pattern4.sub(' ', new_sent3)
    return new_sent3

def txt_process(lines, user_name): 
    # \u202f를 일반 공백으로 치환
    lines = [line.replace('\u202f', ' ') for line in lines]

    df = pd.DataFrame(columns=['text'])

    for line in lines:
        if line is not None:
            decoded_line = line
            df.loc[len(df)] = [decoded_line.strip()]
    
    past_pattern = r"(\d{4}년 \d{1,2}월 \d{1,2}일) (오후|오전) (\d{1,2}:\d{2}), (.*?) :"
    now_pattern = r"\[(.*?)\] \[(오전|오후) (\d{1,2}:\d{2})\]"
    eml_pattern = r"(.*?\d{1,2},\s*\d{4})\s*at\s*(\d{1,2}:\d{2}\s*[AP]M),\s*(.*?)\s*:"
    csv_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),"([^"]+)","([^"]+)"'

    match = re.search(past_pattern, str(df['text'].iloc[10]))
    eml_match = re.search(eml_pattern, str(df['text'].iloc[10]))
    csv_match = re.search(csv_pattern, str(df['text'].iloc[10]))
    
    df['user']=""

    if match:
        pattern = past_pattern
        for index, message in df['text'].items():
            match = re.search(pattern, str(message))
            if match:
                sender = match.group(4)
                df.at[index, 'user'] = sender
                df.at[index, 'text'] = re.sub(pattern, '', df.at[index, 'text'].lstrip())
    else:
        if eml_match: 
            pattern = eml_pattern
            sender_group = 3
            for index, message in df['text'].items():
                match = re.search(pattern, str(message))
                if match:
                    sender = match.group(sender_group)
                    df.at[index, 'user'] = sender
                    df.at[index, 'text'] = re.sub(pattern, '', df.at[index, 'text'].lstrip())
        elif csv_match :
            pattern = csv_pattern
            sender_group = 2
            for index, message in df['text'].items():
                match = re.search(pattern, str(message))
                if match:
                    sender = match.group(sender_group)
                    df.at[index, 'user'] = sender
                    df.at[index, 'text'] = match.group(3)
        else : 
            pattern = now_pattern
            for index, message in df['text'].items():
                match = re.search(pattern, str(message))
                if match:
                    sender = match.group(1)
                    df.at[index, 'user'] = sender
                    df.at[index, 'text'] = re.sub(pattern, '', df.at[index, 'text'].lstrip())
    df.dropna(inplace=True)


    df['text'] = df['text'].apply(str)
    df['text'] = df['text'].apply(remove_some)    # 사진, 이모티콘, 샵검색, https:, [사용자] 등의 불용어 제거
    df['text'] = df['text'].apply(regexp) ### 새로추가 -- 이모티콘 제거

    
    # 채팅방 내의 사용자 리스트 추출
    users = list(df['user'].dropna().unique())
    users = [user for user in users if user != ''] ### 새로추가

    # 사용자가 보낸 문장의 인덱스 추출 --> 이전문장도 사용자가 보낸 문장이면 합치기
    user_index = list(df[df['user'] == user_name].index)
    for index in range(1, len(user_index)):
        if user_index[index] - 1 == user_index[index-1]:
            df.at[user_index[index], 'text'] = df.at[user_index[index-1], 'text'] + " " + df.at[user_index[index], 'text']
            df.drop(index=user_index[index-1], inplace=True)

    # 문장 분리
    user_df = df[df['user'] == user_name]
    user_df['text'] = user_df['text'].apply(split_sentences)

    # 분리된 문장 기준으로 하나씩 데이터프레임에 추가
    splited_df = pd.DataFrame(columns=['text'])
    for row in user_df['text']:
        for txt in row:
            splited_df.loc[len(splited_df)] = [txt]

    splited_df = splited_df[(splited_df['text'].str.len() <= 100)]
    splited_df.dropna(inplace=True)
    # 문장은 최소 5글자
    splited_df = splited_df[splited_df['text'].str.len() >= 5]
    
    print("splited_df! : ", splited_df)
    print("users! : " ,users)

    return splited_df, users

def txt_to_csv(file, user_name):
    file_name = ""
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        file_name = f.name

    #print("==============================")
    #print(lines)
    #print("==============================")

    print(file_name)
    # with open(file, 'rb') as f:
    #     raw_data = f.read()
    #     result = chardet.detect(raw_data)
    #     encoding = result['encoding']
    #     print(f'Detected encoding: {encoding}')

    # # Open the file with the detected encoding
    # with open(file, 'r', encoding=encoding) as f:
    #     lines = f.readlines()

    # 텍스트 파일을 읽어와 리스트로 저장
    #file = open(file, 'r')
    #lines = file.readlines()

    past_pattern = r"(\d{4}년 \d{1,2}월 \d{1,2}일) (오후|오전) (\d{1,2}:\d{2}), (.*?) :"
    now_pattern = r"\[(.*?)\] \[(오전|오후) (\d{1,2}:\d{2})\]"
    eml_pattern = r"(.*?\d{1,2},\s*\d{4})\s*at\s*(\d{1,2}:\d{2}\s*[AP]M),\s*(.*?)\s*:"
    csv_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),"([^"]+)","([^"]+)"'

    match = re.search(past_pattern, lines[5].strip())
    eml_match = re.search(eml_pattern, lines[5].strip())
    csv_match = re.search(csv_pattern, lines[5].strip())

    """ 방 이름 확인 """
    if match:
        room_name = lines[0].replace(" 카카오톡 대화", "")
    elif eml_match: 
        room_name = lines[0].replace("KakaoTalk Chats with ", "")
    elif csv_match:
        file_name = os.path.basename(file)
        room_name = file_name
    else:
        room_name = lines[0].replace(" 님과 카카오톡 대화", "")
    
    """30000문장 아래일 경우 분할 처리 X"""
    if(len(lines)<30000):
        df, users = txt_process(lines, user_name)
        group = len(users) > 2
        return room_name, df, group, str(users)

    else:
        """과거 텍스트 파일 여부 확인"""
        past_pattern = r"(\d{4}년 \d{1,2}월 \d{1,2}일) (오후|오전) (\d{1,2}:\d{2}), (.*?) :"
        # now_pattern = r"\[(.*?)\] \[(오전|오후) (\d{1,2}:\d{2})\]"
        match = re.search(past_pattern, lines[30].strip())
        
        """ 방 이름 확인 """
        # if match:
        #     room_name = lines[0].strip().replace(" 카카오톡 대화", "")
        # else:
        #     room_name = lines[0].strip().replace(" 님과 카카오톡 대화", "")
        
        # 논리 프로세서 수
        chunk_num = mp.cpu_count()
        # num_max_process 수만큼 나눠서 멀티 프로세싱
        each_chunk_length = len(lines) // chunk_num
        # 프로세서 1 : 0~ each_chunk_length*1
        # 프로세서 2 : each_chunk_length*1~ each_chunk_length*2
        chunk_list = []
        for num in range(chunk_num-1):
            chunk_list.append(lines[each_chunk_length * num:each_chunk_length * (num + 1)])
        if len(lines) % chunk_num != 0:
            chunk_list.append(lines[each_chunk_length*chunk_num:])
        pool = mp.Pool(processes=len(chunk_list))
        dfs = pool.starmap(txt_process, [(chunk, user_name) for chunk in chunk_list])
        pool.close()
        pool.join()
        
        """채팅방 전체 txt에서 사용자가 보낸 문장만 뽑아서 데이터프레임으로 변환 완료"""
        df = pd.concat([df for df, _ in dfs], ignore_index=True)
        users = list(set([user for _, user_list in dfs for user in user_list]))
        
        """ 단체방 유무 확인 """
        group = len(users) > 2

        return room_name, df, group, str(users)

"""
target 문체 하나와 다른 문체들을 1:1 매칭한 데이터 프레임 생성
"""
def text_pairing(data, target):
    target_df = pd.DataFrame(data[target])
    use_df=data.drop(columns=[target])
    # target 열과 나머지 열을 한번씩 매칭시켜서 하나의 데이터프레임으로 만듦
    dfs = []
    for col in use_df.columns:
        sub_df = pd.DataFrame()
        sub_df[f"{target}"] = target_df[target]
        sub_df['random'] = use_df[col]
        sub_df.dropna(inplace=True)
        dfs.append(sub_df)

    matching_df = pd.concat(dfs, axis=0, ignore_index=True)    

    return matching_df

import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')