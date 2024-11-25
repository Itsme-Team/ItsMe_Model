from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback)
from tokenizers import Tokenizer
from torch.utils.data import Dataset
import text_preprocessing as txt
import os, shutil
import pandas as pd
from sklearn.model_selection import train_test_split

"""학습 데이터셋 생성 함수"""
class TextStyleTransferDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.style_map = {
            'user': '사용자',
            'random': '무작위'
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index, :].dropna().sample(2)
        text1 = row.iloc[0]
        text2 = row.iloc[1]
        target_style = row.index[1]
        target_style_name = self.style_map[target_style]

        encoder_text = f"{target_style_name} 말투로 변환:{text1}"
        decoder_text = f"{text2}{self.tokenizer.eos_token}"
        model_inputs = self.tokenizer(encoder_text, max_length=64, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(decoder_text, max_length=64, truncation=True)
            model_inputs['labels'] = labels['input_ids']

        return model_inputs

"""모델 학습"""
"""현재는 용량문제로인해 모델 결과물은 저장하지 않음 --> 추후에 용량 확보되면 save_path지정필요"""
def user_modeling(df, hug_obj):
    df=txt.text_pairing(df, 'user')
    model =hug_obj.origin_model
    tokenizer = hug_obj.tokenizer

    # 데이터셋 분리
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(len(df_train), len(df_test))

    train_dataset = TextStyleTransferDataset(
        df_train,
        tokenizer
    )
    test_dataset = TextStyleTransferDataset(
        df_test,
        tokenizer
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model
    )

    directory_to_delete = "saved_model"

    training_args = Seq2SeqTrainingArguments(
                    directory_to_delete,
                    evaluation_strategy = "epoch",
                    save_strategy = "epoch",
                    eval_steps = 10,
                    load_best_model_at_end = True,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    gradient_accumulation_steps=2,
                    weight_decay=0.01,
                    save_total_limit=1,
                    num_train_epochs=10,
                    predict_with_generate=True,
                    fp16=False,
            )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 모델 학습
    trainer.train()

    
    # trainer.save_model(save_path) # 용량으로 인해 저장은 안함.
    
    if os.path.exists(directory_to_delete):
        shutil.rmtree(directory_to_delete)

    return trainer.model