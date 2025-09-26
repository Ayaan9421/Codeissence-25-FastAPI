from typing import List
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from transformers import pipeline

load_dotenv()

asr = pipeline("automatic-speech-recognition", model="Oriserve/Whisper-Hindi2Hinglish-Swift")
res1 = asr("test4.mpeg")
print(res1)

ver = pipeline(
    task="audio-classification",
    model="harshit345/xlsr-wav2vec-speech-emotion-recognition"
)
res2 = ver("test4.mpeg")
print(res2)

llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation"
)

llm_model = ChatHuggingFace(llm=llm)

class EmotionDetail(BaseModel):
    label: str = Field(..., description="Emotion label detected")
    score: float = Field(..., description="Confidence score of the detected emotion (0-1)")

class DepressionReport(BaseModel):
    depression_score: int = Field(..., description="Depression level on a scale of 1 to 10")
    description: str = Field(..., description="Explanation of depression level based on emotion analysis")
    risks: List[str] = Field(..., description="Potential risks or warning signs detected from emotions")
    advice: str = Field(..., description="Practical advice or next steps for the user")
    emotions: List[EmotionDetail] = Field(..., description="Detailed breakdown of detected emotions with scores")

parser = PydanticOutputParser(pydantic_object=DepressionReport)

promptTemplate1 = PromptTemplate(
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

promptTemplate2 = PromptTemplate(
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

chain1 =  promptTemplate1 | llm_model | parser

result1 = chain1.invoke(res1['text'])
print(result1.dict())


