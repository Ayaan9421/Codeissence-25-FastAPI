from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from transformers import pipeline
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Step 1: Define LLM and parser
# -----------------------------
llm_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)
llm_model = ChatHuggingFace(llm=llm_endpoint)

class DepressionReport(BaseModel):
    depression_score: int = Field(..., description="Depression level on a scale of 1 to 10")
    description: str = Field(..., description="Short explanation for the score")
    advice: str = Field(..., description="Important notes or advice for the user")

parser = PydanticOutputParser(pydantic_object=DepressionReport)

prompt_template = PromptTemplate(
    template="""
You are a mental health expert. Analyze the following input:

{text_or_emotion}

{format_instructions}
""",
    input_variables=["text_or_emotion"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm_chain = prompt_template | llm_model | parser

# -----------------------------
# Step 2: Define ASR and VER Runnables
# -----------------------------
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model="Oriserve/Whisper-Hindi2Hinglish-Swift"
)
ver_pipeline = pipeline(
    task="audio-classification",
    model="harshit345/xlsr-wav2vec-speech-emotion-recognition"
)

# RunnableLambda for ASR
asr_runnable = RunnableLambda(lambda audio_file: asr_pipeline(audio_file)['text'])

# RunnableLambda for VER (summarize as string)
def summarize_ver(audio_file):
    results = ver_pipeline(audio_file)
    # Convert labels and scores into a simple string for LLM
    return ", ".join([f"{r['label']}({r['score']:.2f})" for r in results])

ver_runnable = RunnableLambda(summarize_ver)

# -----------------------------
# Step 3: ParallelChain
# -----------------------------
parallel_chain = RunnableParallel({
    "asr_analysis": asr_runnable | llm_chain,
    "ver_analysis": ver_runnable | llm_chain
})

# -----------------------------
# Step 4: Run the chain
# -----------------------------
audio_file = "test4.mpeg"
results = parallel_chain.invoke({"audio_file": audio_file})

# -----------------------------
# Step 5: Normalize / combine scores
# -----------------------------
asr_result: DepressionReport = results['asr_analysis']
ver_result: DepressionReport = results['ver_analysis']

combined_score = round((asr_result.depression_score + ver_result.depression_score) / 2)

combined_report = {
    "combined_depression_score": combined_score,
    "asr_analysis": asr_result.dict(),
    "ver_analysis": ver_result.dict()
}

print(combined_report)

# -----------------------------
# Optional: print chain graph
# -----------------------------
parallel_chain.get_graph().print_ascii()
