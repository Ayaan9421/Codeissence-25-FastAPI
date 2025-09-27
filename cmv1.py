from fastapi import FastAPI
from pydantic import BaseModel
import datetime
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

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


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    chat_chain = get_session(request.session_id)
    response = chat_chain.run(request.user_input)
    print(sessions)
    return ChatResponse(reply=response)

from parallel_chain import router_pc
from tta import router_tta
app.include_router(router=router_pc)
app.include_router(router=router_tta)