# FastAPI 관련 라이브러리 임포트
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

# LangChain 관련 라이브러리 임포트
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="RAG QA API",
    description="LangChain을 사용한 RAG 기반 질의응답 API",
    version="1.0.0"
)

# 요청 모델 정의
class Question(BaseModel):
    query: str

# 응답 모델 정의
class Answer(BaseModel):
    answer: str
    context: List[str]

# RAG 시스템 초기화 함수
def initialize_rag():
    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 문서 로드 및 분할
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), "data", "테슬라_KR.txt")
    loader = TextLoader(data_path, encoding='utf-8')
    data = loader.load()
    
    text_splitter = CharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separator='\n\n'
    )
    texts = text_splitter.split_documents(data)
    
    # 벡터 저장소 생성
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name="chroma_test",
        persist_directory="./chroma_db"
    )
    
    # LLM 초기화
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    # 프롬프트 템플릿 설정
    prompt = ChatPromptTemplate.from_template("""
    다음 컨텍스트를 바탕으로 질문에 답변해주세요. 컨텍스트에 관련 정보가 없다면, 
    "주어진 정보로는 답변할 수 없습니다."라고 말씀해 주세요.

    컨텍스트: {context}

    질문: {input}

    답변:
    """)
    
    # 체인 생성
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return rag_chain

# RAG 시스템 초기화
rag_chain = initialize_rag()

# API 엔드포인트 정의
@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        # RAG 체인 실행
        response = rag_chain.invoke({"input": question.query})
        
        # 컨텍스트 추출
        contexts = [doc.page_content for doc in response["context"]]
        
        return Answer(
            answer=response["answer"],
            context=contexts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 