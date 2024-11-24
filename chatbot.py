import google.generativeai as genai
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import re
import pandasql as ps
import numpy as np
from ast import literal_eval
import pickle
import time
from datetime import datetime

genai.configure(api_key="")
assistant = genai.GenerativeModel("gemini-1.5-flash")

def response_1(question):
    now = datetime.now()
    formatted_time = now.strftime("%m-%d-%A-%p-%I-%M")
    print(f"현재 시간: {formatted_time}")
    print("---------------------------------------------response_1-----------------------------------------")
    with open('data/festival.pkl', 'rb') as file:  # 'rb'는 바이너리 읽기 모드
        data_ = pickle.load(file)
    response1 = assistant.generate_content(f"""system: 너는 축제 추천을 위한 데이터 filter야. 주어진 질문을 보고 이 중 답변에 필요한 사전정보를 얻기 위한 pandasql에 적용 가능한 쿼리를 작성하고, 왜 그렇게 생각했는지 이유를 "reason"태그를 달아서 작성해줘

Tip: '근처'의 정의는 같은 도나 시에 있다는 뜻이야.
시작일, 마감일 - 각각 20241022와 같은 int형태로 되어 있어. 앞의 예시에서 2024는 년도 10은 월, 22는 일이야
주소는 "해변가", "산 주위" 와 같은 값은 가지고 있지 않아. "대구광역시 달서구", "제주특별자치도 서귀포시 남원읍"와 같은 값을 가지고 있어
축제 소개 컬럼에서는 최대한 %바베큐%, %바닷가%, %노래자랑%과 같은 형식을 사용해.
비용은 무료, 유료, 부분유료 중 하나의 값을 지니고 있어
테이블의 이름은 df야.
현재 시간: {formatted_time}

columns: 축제이름, 축제소개, 시작일, 마감일, 주소, 비용
질문: {question}""")
    print("-----response-----")
    print(response1.text)
    query = response1.text.split("```")
    query = query[1][4:]
    query = re.sub(r"^SELECT\s+[^\s]+", "SELECT *", query)
    print("-----query-----")
    print(query)
    try:
        result_df = ps.sqldf(query, {'df': data_})
    except:
        exit()
        result_df, _ = response_1(question)
    print("-----query_result-----")
    print(result_df)
    if len(result_df) <= 9: 
        data = data_
    else:
        data = result_df
        data["embedding"] = data["primary_key"].apply(lambda x: data_.iloc[int(x)]["embedding"])
    data_embeded = data['embedding']
    embedding_dim = len(data_embeded[0])
    try:
        data_embeded = np.array(data_embeded.tolist(), dtype=np.float32)
    except:
        data['embedding'] = data['embedding'].apply(lambda x: np.frombuffer(x, dtype=np.float32))
        embedding_dim = len(data_embeded[0])
        data_embeded = np.array(data_embeded.tolist(), dtype=np.float32)

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(data_embeded)
    model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    query_vector = model.encode(question)
    query_vector = query_vector.reshape(1, -1)
    query_vector = np.array(query_vector, dtype=np.float32)
    k = 9
    _, labels = index.search(query_vector, k)
    if len(result_df) <= 3:
        result_df = pd.concat([result_df, data.iloc[labels[0]]], axis=0, ignore_index=True)
    elif len(result_df) >= 10:
        result_df = data.iloc[labels[0]]
    return result_df

def response_3(question, filtered_df):
    now = datetime.now()
    formatted_time = now.strftime("%m-%d-%A-%p-%I-%M")
    filtered_data = []
    filtered_df = filtered_df.drop(columns = ["embedding", "primary_key"])
    for i in range(len(filtered_df)):
        filtered_data.append(" ".join([col + ":"+ str(filtered_df.iloc[i][col]) for col in filtered_df.columns]))
    filtered_data = "\n".join(filtered_data)
    print("-----filtered_data-----")
    print(filtered_data)
    print("\n---------------------------------------------response_3-----------------------------------------\n")
    response_3 = assistant.generate_content(f"""system: 너는 축제들을 소개하는 가이드야, 한번에 최대 열개의 축제를 추천한 이유와 함께 추천해줘. 다만, 질문에 부합하지 않는 축제는 추천하면 안돼.
과하지 않은 이모티콘은 사용해도 돼.
현재 시간: {formatted_time}
네 이름: 축젯티피
question: {question}
축제 리스트: {filtered_data}""")
    print("-----final_response-----")
    print(response_3.text.replace("*", ""))
    text = re.sub(r"(\]\(https?://[^\)]+\))(?!\n)", r"\1\n", response_3.text.replace("*", ""))
    return text

def generate_response(question):
    result_df = response_1(question)

    final_answer = response_3(question, result_df)

    return final_answer

if __name__ == "__main__":
    while 1:
        question = input("질문 입력: ")
        generate_response(question)

