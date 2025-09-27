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

asr = pipeline(
    task = "automatic-speech-recognition", 
    model="Oriserve/Whisper-Hindi2Hinglish-Swift",
    device=0 
)

def asr_run(audio_file: str)-> str:
    return asr(audio_file)['text']

asr_runnable = RunnableLambda(asr_run)

ver = pipeline(
    task="audio-classification",
    model="harshit345/xlsr-wav2vec-speech-emotion-recognition",
    device=0 
)

def ver_run(audio_file: str):
    return ver(audio_file)

ver_runnable = RunnableLambda(ver_run)

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

promptTemplate_sttta = PromptTemplate(
    template="""
You are a mental health expert. Analyze the following transcribed speech text from a user 
and provide insights on their depression level and potential mental health risks.

Transcribed Text:
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

promptTemplate_ste = PromptTemplate(
    template="""
You are a mental health expert specializing in analyzing emotional states from speech. 
Analyze the following emotion recognition output from a user and provide insights on their 
depression level and potential mental health risks.

Emotion Data (label and confidence scores): 
{text_or_emotion}

Instructions:
1. Provide a depression_score on a scale of 1 to 10 (1 = very low, 10 = very high).
2. Give a concise description explaining how the emotional distribution relates to the depression assessment.
3. Highlight any potential risks or warning signs based on the emotions detected.
4. Suggest practical advice or next steps for the user.

Return the output strictly in the following JSON format:

{format_instructions}
""",
    input_variables=["text_or_emotion"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

promptTemplate_merge = PromptTemplate(
    template="""
You are a senior mental health expert. You will receive two structured depression assessment reports:  
1. One based on **speech-to-text analysis** (journal content)  
2. One based on **vocal emotion recognition** (tone/emotion distribution).  
depression_score=0 means No depression whereas depression_score=10 means Highly depressed.
Your job is to carefully combine and normalize the insights from both reports into a **final unified assessment**.

Inputs:
- Report from speech-to-text analysis: {report_asr}
- Report from vocal emotion recognition: {report_ver}

Instructions:
1. Normalize both depression scores to a final single `depression_score` (integer 1–10).  
   - If both reports agree, keep the same score.  
   - If they differ, average them (round sensibly).  
   - If one has strong risks flagged, lean slightly towards the higher score.  

2. Write a **concise but clear description** summarizing why this depression level was chosen, referencing evidence from both inputs.  

3. Merge all potential `risks` from both reports (avoid duplicates).  

4. Suggest practical `advice` based on combined findings.  

5. Provide a merged list of `emotions` from both reports with their scores.  

Return the output strictly in the following JSON format:  
{format_instructions}
""",
    input_variables=["report_asr", "report_ver"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

sttta_chain = asr_runnable | promptTemplate_sttta | llm_model | parser
ste_chain = ver_runnable | promptTemplate_ste | llm_model | parser

parallel_chain = RunnableParallel({
    'stta': sttta_chain,
    'ste': ste_chain,
    'transcript': asr_runnable
})

final_chain = parallel_chain | remap | promptTemplate_merge | llm_model | parser

# print(final_chain.get_graph().print_ascii())
# result = final_chain.invoke(input("Enter Audio File Name: "))
# print(result.dict())

router_pc = APIRouter()

@router_pc.post("/analyze", response_model=DepressionReport)
async def analyze_audio(file: UploadFile = File(...)):
    print("hello")
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Run final chain
    result: DepressionReport = final_chain.invoke(tmp_path)

    # Inject transcript into result object
    result.transcript = result.transcript or result.transcript

    return JSONResponse(content=result.model_dump())
