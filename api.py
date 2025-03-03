import os
import openai
from dotenv import load_dotenv
from fastapi.params import Query, Body
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from embedding_faiss import load_documents_from_files

load_dotenv(verbose=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini"
)

template = "아래 질문에 대한 답변을 해주세요. \n{query}"
prompt_template = PromptTemplate(input_variables=["query"], template=template)
chain = prompt_template | llm | StrOutputParser()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await chain.ainvoke({"query": request.query})
        return response
    except Exception as e:
        return str(e)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    def event_stream():
        try:
            for chunk in chain.stream({"query": request.query}):
                if len(chunk) > 0:
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/document/question")
async def question():
    return load_documents_from_files()
