from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import datetime
from fastapi import FastAPI
import os

from pydantic import BaseModel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=60
)

chat_model = ChatHuggingFace(llm=llm)

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)

chat_chain = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=True
)

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

chat_chain.prompt = supportive_prompt

SESSION_EXPIRY_HOURS = 24
last_reset_time = datetime.datetime.now()

def check_and_reset_session():
    global last_reset_time, memory
    
    now = datetime.datetime.now()
    if (now - last_reset_time).total_seconds() > SESSION_EXPIRY_HOURS * 3600:
        # TODO: Save chat history to DB
        # Reset memory
        memory.clear()
        last_reset_time = now

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["quit", "exit"]:
#         break
    
#     check_and_reset_session()
    
#     response = chat_chain.run(user_input)
#     print("Friend:", response)

# print(memory)

app = FastAPI(title="Supportive Friend Chatbot")

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    check_and_reset_session()
    response = chat_chain.run(request.user_input)
    return ChatResponse(reply=response)