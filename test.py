from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=60
)

chat_model = ChatHuggingFace(llm=llm)

print(chat_model.predict("Hello! How are you today?"))
