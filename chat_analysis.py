from typing import List
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from chatmodel import memory

load_dotenv()

class EmotionDetail(BaseModel):
    label: str = Field(..., description="Emotion label detected")
    score: float = Field(..., description="Confidence score of the detected emotion (0-1)")

class DepressionReport(BaseModel):
    depression_score: int = Field(..., description="Depression level on a scale of 1 to 10", ge=0, le=10)
    description: str = Field(..., description="Explanation of depression level based on emotion analysis")
    risks: List[str] = Field(..., description="Potential risks or warning signs detected from emotions")
    advice: str = Field(..., description="Practical advice or next steps for the user")
    emotions: List[EmotionDetail] = Field(..., description="Detailed breakdown of detected emotions with scores")

parser = PydanticOutputParser(pydantic_object=DepressionReport)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)
llm_model = ChatHuggingFace(llm=llm)


analysis_prompt = PromptTemplate(
    template="""
You are a mental health professional. Analyze the following conversation and provide insights
on the user's mental health, including depression levels, risks, advice, and emotional breakdown.

Conversation:
{chat_history}

Instructions:
1. Provide an overall depression_score (1-10, 10 being most severe).
2. Give a concise description explaining the user's emotional state and depression assessment.
3. List potential mental health risks or warning signs.
4. Provide practical advice or next steps for the user.
5. Provide a detailed breakdown of emotions with estimated confidence scores (0-1) if possible.

Return the output strictly in the following JSON format:
{format_instructions}
""",
    input_variables=["chat_history"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = analysis_prompt | llm_model | parser
result = chain.invoke(memory.load_memory_variables({})['history'])
print(result.model_dump())

