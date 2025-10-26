from typing import List
from fastapi import APIRouter, HTTPException
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
    summary: str = Field(
        ..., description="The input chat history summary"
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

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)
llm_model = ChatHuggingFace(llm=llm)


analysis_prompt = PromptTemplate(
    template="""
You are a mental health professional. Analyze the following conversation between a User and an AI Assistant (Calmora)
and provide insights ONLY on the USER's mental health, including depression levels, risks, advice, and emotional breakdown.
Focus solely on the User's messages for the analysis. Ignore the AI's messages when assessing mood or risk.

Conversation:
{chat_history}

Instructions:
1. Provide an overall depression_score for the User (0-10, 10 being most severe).
2. Give a concise description explaining the User's emotional state and depression assessment, based *only* on their messages.
3. List potential mental health risks or warning signs shown by the User.
4. Provide practical advice or next steps specifically for the User.
5. Provide a detailed breakdown of the User's emotions with estimated confidence scores (0-1) if possible.
6. Set the 'summary' field in the JSON to the full {chat_history} provided.

Return the output strictly in the following JSON format:
{format_instructions}
""",
    input_variables=["chat_history"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = analysis_prompt | llm_model | parser

router_chat_analysis = APIRouter()

class ChatAnalysisRequest(BaseModel):
    chat_history: str

@router_chat_analysis.post("/analyze-chat", response_model=DepressionReport)
async def analyze_chat_endpoint(request: ChatAnalysisRequest):
    try:
        # Invoke the Langchain chain
        result: DepressionReport = await chain.invoke({"chat_history": request.chat_history})
        # Ensure the transcript field is populated with the input history
        result.transcript = request.chat_history
        return result
    except Exception as e:
        print(f"Chat Analysis Error: {e}")
        # Log the specific error for debugging
        # Consider returning a more specific error based on the exception type
        raise HTTPException(status_code=500, detail=f"Error during chat analysis: {e}")
