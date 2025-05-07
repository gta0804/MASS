import sys
sys.path.append("./")
from stock_prediction_benchmark import OpenAIModel
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import logging
from stock_prediction_benchmark import SUMMARIZE_EXAMPLE, SUMMARIZE_INSTRUCTION

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="any-key", 
    # api_key="sk-3e9ad1ee5c0942ce967965afb8a73fcc",
    # base_url="http://14.103.37.13:54318/chat/completions",
    base_url="http://10.129.163.13:11434/v1"
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
openai_model = OpenAIModel("qwen2.5:3b-instruct-fp16", None, 2048)

def clean_news(raw_data: str) -> str:
    prompts = SUMMARIZE_INSTRUCTION.format(
        example=SUMMARIZE_EXAMPLE,
        input_data=raw_data)
    res, reason = openai_model.generate(client, prompts)
    res = res.strip()
    logging.warning(f"before cleaning:{raw_data} \n after cleaning: {res}")
    return res

if __name__ == "__main__":
    start_date, end_date = "20230701", "20240101"
    logging.basicConfig(
    level=logging.WARNING,  # 设置日志级别为 WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    filename=f'{start_date}-{end_date}.log',  # 日志输出到文件
    filemode='w'  # 文件模式：'a' 为追加模式，'w' 为覆盖模式
    )
    news_data = pd.read_parquet("/home/disk3/gta/stock_prediction/stock_prediction_benchmark/stock_disagreement/dataset/wind-financial-news-info.parq")
    news_data = news_data[(news_data["Date"] >= start_date) & (news_data["Date"] <= end_date)].copy()
    tqdm.pandas(desc="cleaning and summarizing news")
    news_data["clean_news"] = news_data["NewsContent"].progress_apply(clean_news)
    news_data.to_parquet(f"/home/disk3/gta/stock_prediction/stock_prediction_benchmark \
                         /stock_disagreement/dataset/wind-financial-news-info-{start_date}-{end_date}.parq")