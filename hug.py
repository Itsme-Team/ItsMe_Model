from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    pipeline
    )
import torch

class hugging():
    # 허깅페이스 레포 주소
    def __init__(self):
        cache_dir = "./hugging_face"
        origin_model_path="paust/pko-t5-base"
        formal_model_path='9unu/formal_speech_translation'
        gentle_model_path='9unu/gentle_speech_translation'
        self.formal_model = T5ForConditionalGeneration.from_pretrained(formal_model_path, cache_dir=cache_dir)
        self.gentle_model = T5ForConditionalGeneration.from_pretrained(gentle_model_path, cache_dir=cache_dir)
        
        self.origin_model = T5ForConditionalGeneration.from_pretrained(origin_model_path, cache_dir=cache_dir)
        self.tokenizer = T5TokenizerFast.from_pretrained(formal_model_path, cache_dir=cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# 파이프라인 실행 -> 변환 문장 return
def make_pipeline(model, tokenizer, device):
    print("torch.cuda.is_available : ", torch.cuda.is_available())
    print("device : ", device)
    nlg_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=device, max_length=60) # "auto" -> 자동으로 분산
    return nlg_pipeline

