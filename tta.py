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

remap = RunnableLambda(lambda x: {
    "report_asr": x["stta"].model_dump_json(),
    "report_ver": x["ste"].model_dump_json(),
    "transcript": x["transcript"]  # keep raw transcription
})


llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation"
)

llm_model = ChatHuggingFace(llm=llm)

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

promptTemplate = PromptTemplate(
    template="""
You are a mental health expert. Analyze the following journal text from a user 
and provide insights on their depression level and potential mental health risks.

User Journal Text:
{text_or_emotion}

Instructions:
1. Provide a depression_score on a scale of 1 to 10 (1 = very low, 10 = very high).
2. Give a concise description explaining how the text indicates the user's emotional state and depression assessment.
3. Highlight any potential risks or warning signs inferred from the text.
4. Suggest practical advice or next steps for the user.

Return the output strictly in the following JSON format:

{format_instructions}
""",
    input_variables=["text_or_emotion"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = promptTemplate | llm_model | parser 

router_tta = APIRouter()
@router_tta.post('/text', response_model=DepressionReport)
async def analyze_text(text: str):
    result : DepressionReport = chain.invoke(text)
    return JSONResponse(content= result.model_dump())


