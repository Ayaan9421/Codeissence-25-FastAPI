from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import datetime
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

import tempfile
from typing import List
from fastapi.responses import JSONResponse
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from transformers import pipeline
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnableParallel
from fastapi import APIRouter, File, UploadFile
import os
from pydub import AudioSegment


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=60,
)

chat_model = ChatHuggingFace(llm=llm)

supportive_prompt = PromptTemplate(
    template="""
You are a supportive friend. 
Your goal is to make the user feel relaxed and less stressed. 
Keep your replies short (1-3 sentences). 
Sound casual and natural, like chatting on WhatsApp. 
Never lecture or over-explain.

Conversation so far:
{history}

User: {input}
Friend:""",
    input_variables=["history", "input"],
)

# ------------------------------
#  Session Management
# ------------------------------

SESSION_EXPIRY_HOURS = 24

# Store { session_id: { "chain": ConversationChain, "last_reset": datetime } }
sessions = {}

def get_session(session_id: str):
    """Fetch or create a chat session for a given user/session_id."""
    now = datetime.datetime.now()

    if session_id not in sessions:
        # Create new memory + chain
        memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        chain = ConversationChain(
            llm=chat_model,
            memory=memory,
            prompt=supportive_prompt,
            verbose=True,
        )
        sessions[session_id] = {"chain": chain, "last_reset": now}

    # Check expiry
    last_reset = sessions[session_id]["last_reset"]
    if (now - last_reset).total_seconds() > SESSION_EXPIRY_HOURS * 3600:
        # TODO: Save chat history to DB
        sessions[session_id]["chain"].memory.clear()
        sessions[session_id]["last_reset"] = now

    return sessions[session_id]["chain"]

# ------------------------------
#  FastAPI
# ------------------------------

app = FastAPI(title="Supportive Friend Chatbot")

class ChatRequest(BaseModel):
    session_id: str
    user_input: str

class ChatResponse(BaseModel):
    reply: str

class EmotionDetail(BaseModel):
    label: str = Field(..., description="Emotion label detected")
    score: float = Field(..., description="Confidence score of the detected emotion (0-1)")

class DepressionReport(BaseModel):
    transcript: str = Field(
        None, description="Raw transcribed text from speech input"
    )
    depression_score: int = Field(
        ..., 
        description="Depression level on a scale of 1 to 10", 
        ge=0, le=10
    )
    description: str = Field(
        ..., 
        description="Explanation of depression level based on emotion analysis"
    )
    risks: List[str] = Field(
        ..., 
        description="Potential risks or warning signs detected from emotions"
    )
    advice: List[str] = Field(
        ..., 
        description="Practical advice or next steps for the user, as a list of suggestions"
    )
    emotions: List[EmotionDetail] = Field(
        ..., 
        description="Detailed breakdown of detected emotions with scores"
    )

parser = PydanticOutputParser(pydantic_object=DepressionReport)

analysis_prompt = PromptTemplate(
    template="""
You are a mental health professional. Analyze the following conversation history and provide insights.

Conversation History:
{chat_history}

Instructions:
1. Always set `"transcript"` to the **exact conversation history text as a string** (never null).
2. Provide an overall `"depression_score"` (1-10).
3. Write a `"description"` explaining the user's emotional state.
4. List possible `"risks"`.
5. Suggest `"advice"`.
6. Provide `"emotions"` with labels and scores.

Return strictly in JSON format:
{format_instructions}
""",
    input_variables=["chat_history"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation"
)

llm_model = ChatHuggingFace(llm=llm)

chain_analysis = analysis_prompt | llm_model | parser

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    chat_chain = get_session(request.session_id)
    response = chat_chain.run(request.user_input)
    print(sessions)
    return ChatResponse(reply=response)

@app.post("/chat_analysis", response_model=DepressionReport)
async def chat_analysis(request: ChatRequest):
    chat_chain = get_session(request.session_id)

    # Get chat history
    history_messages = chat_chain.memory.load_memory_variables({})["history"]
    if isinstance(history_messages, list):
        chat_history = "\n".join(
            [f"{msg.type.capitalize()}: {msg.content}" for msg in history_messages]
        )
    else:
        chat_history = str(history_messages)

    # Run LLM
    raw_result = chain_analysis.invoke({"chat_history": chat_history})

    # Fix transcript if null
    if raw_result.get("transcript") is None:
        raw_result["transcript"] = chat_history or ""

    result = DepressionReport.model_validate(raw_result)
    return JSONResponse(content=result.model_dump())


from parallel_chain import DepressionReport, router_pc
from tta import router_tta
app.include_router(router=router_pc)
app.include_router(router=router_tta)