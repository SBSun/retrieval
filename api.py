import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import models
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from embedding_faiss import load_documents_from_files

load_dotenv(verbose=True)
openai.api_key = os.getenv('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-3.5-turbo")

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/chat", response_model=ChatResponse)
async def chat():
    try:
        # GPT-3.5-turbo 모델 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "hello"}
            ]
        )
        # 모델의 응답 추출
        reply = response['choices'][0]['message']['content']
        return ChatResponse(answer=reply)
    except Exception as e:
        # 에러 처리
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/document/question")
async def question():
    return load_documents_from_files()