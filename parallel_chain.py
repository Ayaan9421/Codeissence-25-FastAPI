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
import soundfile as sf

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

remap_audio = RunnableLambda(lambda x: {
    "report_asr_audio": x["stta_audio"].model_dump_json(),
    "report_ver_audio": x["ste_audio"].model_dump_json(),
    "transcript_audio": x["transcript_audio"]
})

remap_call = RunnableLambda(lambda x: {
    "report_asr_call": x["stta_call"].model_dump_json(),
    "report_ver_call": x["ste_call"].model_dump_json(),
    "summary_call": x["summary_call"]
})

llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation"
)

llm_model = ChatHuggingFace(llm=llm)

from enum import Enum

class MoodStage(str, Enum):
    ACCEPTANCE = "Acceptance"
    DENIAL = "Denial"
    BARGAINING = "Bargaining"
    ANGER = "Anger"
    DEPRESSION = "Depression"


class EmotionDetail(BaseModel):
    label: MoodStage = Field(..., description="Emotion stage detected (one of the five standard labels)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the detected emotion (0-1)")

class DepressionReport(BaseModel):
    transcript: str = Field(
        ..., description="Raw transcribed text from speech input"
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

class CallAnalysisReport(BaseModel):
    summary: str = Field(
        ..., 
        description="Short AI-generated summary of the conversation or call content, 4-5 sentences, just the summary of the call and no analysis"
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

parser_audio = PydanticOutputParser(pydantic_object=DepressionReport)
parser_call = PydanticOutputParser(pydantic_object=CallAnalysisReport)

# =====================================
#  AUDIO PROMPTS (for journal entries)
# =====================================

promptTemplate_sttta_audio = PromptTemplate(
    template="""
You are a mental health expert. Analyze the following transcribed speech text from a user 
and assess their emotional wellbeing on a scale of 0–10, where **lower is healthier**.

Transcribed Text:
{text_or_emotion}

### Emotion Scale (0–10)
- 0–2 → **Acceptance (😌)** — Calm, balanced, emotionally healthy.
- 2–4 → **Denial (🤔)** — Mild avoidance or unwillingness to confront emotions.
- 4–6 → **Bargaining (🥺)** — Trying to rationalize or control emotional pain.
- 6–8 → **Anger (😠)** — Irritation, frustration, or emotional volatility.
- 8–10 → **Depression (😔)** — Sadness, hopelessness, or emotional withdrawal.

### Instructions:
1. Assign a `depression_score` between 0 and 10 using the above emotion mapping.
2. Write a short `description` explaining which emotional stage this score corresponds to and why.
3. Identify any potential `risks` or warning signs if distress appears.
4. Suggest simple, compassionate `advice` or next steps for emotional regulation.
5. Keep tone **empathetic, non-diagnostic**, and friendly.

### Instructions for Emotion Output
- Only return **these five emotion stages**:
  Acceptance, Denial, Bargaining, Anger, Depression
- Each must have a confidence score between 0.0 and 1.0
- Do not invent or use any other labels
- If the emotion is not strongly detected, you can assign a low score (e.g., 0.0–0.2)
- Make sure the sum of scores does **not need to equal 1**, but each score should reflect the LLM's confidence

### Extremely Important Note:
- Only return these five emotion stages:
  Acceptance, Denial, Bargaining, Anger, Depression
- Do not invent or use any other labels

Return the output strictly in this JSON format:

{format_instructions}
""",
    input_variables=["text_or_emotion"],
    partial_variables={"format_instructions": parser_audio.get_format_instructions()},
)


promptTemplate_ste_audio = PromptTemplate(
    template="""
You are a mental health expert analyzing emotion recognition data from a user's speech.
Each label represents a detected emotion and its confidence score.

Emotion Data:
{text_or_emotion}

### Emotion Scale (0–10)
- 0–2 → **Acceptance (😌)** — Calm and emotionally stable tone.
- 2–4 → **Denial (🤔)** — Slight tension or emotional avoidance.
- 4–6 → **Bargaining (🥺)** — Signs of inner conflict or seeking reassurance.
- 6–8 → **Anger (😠)** — Raised energy, frustration, or agitation in tone.
- 8–10 → **Depression (😔)** — Low energy, sadness, or emotional fatigue.

### Instructions:
1. Assign a `depression_score` (0–10) using the above emotional stages.
2. Write a concise `description` describing the emotional state and reasoning.
3. Identify any potential `risks` (e.g., signs of distress, irritability, or hopelessness).
4. Provide kind and actionable `advice` for self-care or coping.
5. Keep tone neutral, warm, and supportive.

### Instructions for Emotion Output
- Only return **these five emotion stages**:
  Acceptance, Denial, Bargaining, Anger, Depression
- Each must have a confidence score between 0.0 and 1.0
- Do not invent or use any other labels
- If the emotion is not strongly detected, you can assign a low score (e.g., 0.0–0.2)
- Make sure the sum of scores does **not need to equal 1**, but each score should reflect the LLM's confidence

### Extremely Important Note:
- Only return these five emotion stages:
  Acceptance, Denial, Bargaining, Anger, Depression
- Do not invent or use any other labels

Return the output strictly in this JSON format:

{format_instructions}
""",
    input_variables=["text_or_emotion"],
    partial_variables={"format_instructions": parser_audio.get_format_instructions()},
)


promptTemplate_merge_audio = PromptTemplate(
    template="""
You are a senior mental health expert. You will receive two structured reports:
1. A **speech-to-text-based** analysis.
2. A **vocal emotion recognition** analysis.

Your goal is to merge them into one unified, balanced mental wellbeing report.

Inputs:
- Speech Analysis Report: {report_asr_audio}
- Emotion Analysis Report: {report_ver_audio}

### Emotion Scale (0–10)
- 0–2 → **Acceptance (😌)** — Calm, grounded, emotionally well.
- 2–4 → **Denial (🤔)** — Avoiding discomfort or stress.
- 4–6 → **Bargaining (🥺)** — Trying to rationalize or seek control.
- 6–8 → **Anger (😠)** — Irritable, emotionally intense, or reactive.
- 8–10 → **Depression (😔)** — Sad, hopeless, or emotionally withdrawn.

### Instructions:
1. Normalize both reports into a final `depression_score` (0–10).
   - If both agree, keep that value.
   - If different, average and round sensibly.
   - If one shows strong risk, lean slightly higher.
2. Write a short, clear `description` summarizing which emotional stage fits best and why.
3. Merge all `risks` (avoid duplicates).
4. Combine `advice` from both, focusing on positivity and coping.
5. Merge `emotions` from both reports.

### Additional Instructions for Emotion Output
- Merge the `emotions` list from both reports
- Only keep the five standard emotions: Acceptance, Denial, Bargaining, Anger, Depression
- Avoid creating new labels
- Keep confidence scores (0–1) and deduplicate if necessary

### Extremely Important Note:
- Only return these five emotion stages:
  Acceptance, Denial, Bargaining, Anger, Depression
- Do not invent or use any other labels

Return the output strictly in this JSON format:

{format_instructions}
""",
    input_variables=["report_asr_audio", "report_ver_audio"],
    partial_variables={"format_instructions": parser_audio.get_format_instructions()},
)


# =====================================
#  CALL PROMPTS
# =====================================

promptTemplate_sttta_call = PromptTemplate(
    template="""
You are an AI summarizer for call recordings.

Your task:
1. Write a **neutral, human-like summary** (2–4 sentences) describing what was discussed in the call.
   - Do NOT include emotional or mental health analysis here.
   - Focus only on what happened or was discussed.

Then assess the **emotional wellbeing** of the user based on call content.

### Emotion Scale (0–10)
- 0–2 → **Acceptance (😌)** — Calm, composed, emotionally clear communication.
- 2–4 → **Denial (🤔)** — Avoidant or dismissive tone or statements.
- 4–6 → **Bargaining (🥺)** — Uncertainty, reasoning, or seeking reassurance.
- 6–8 → **Anger (😠)** — Frustration or tension in words or phrasing.
- 8–10 → **Depression (😔)** — Hopeless, sad, or emotionally low expressions.

### Instructions:
2. Assign a `depression_score` using this emotional scale.
3. Add a short `description` that matches the corresponding emotional state.
4. List any clear `risks` (stress cues, hopelessness, etc.).
5. Provide gentle `advice` for next steps.
6. Keep `emotions` list empty or neutral if tone isn’t clearly emotional.

### Instructions for Emotion Output
- Only return **these five emotion stages**:
  Acceptance, Denial, Bargaining, Anger, Depression
- Each must have a confidence score between 0.0 and 1.0
- Do not invent or use any other labels
- If the emotion is not strongly detected, you can assign a low score (e.g., 0.0–0.2)
- Make sure the sum of scores does **not need to equal 1**, but each score should reflect the LLM's confidence

### Extremely Important Note:
- Only return these five emotion stages:
  Acceptance, Denial, Bargaining, Anger, Depression
- Do not invent or use any other labels

Return the output strictly in this JSON format:

{format_instructions}
""",
    input_variables=["text_or_emotion"],
    partial_variables={"format_instructions": parser_call.get_format_instructions()},
)


promptTemplate_ste_call = PromptTemplate(
    template="""
You are analyzing vocal emotion data extracted from a user's call.
Each emotion label shows how the user sounded during the call.

Emotion Data:
{text_or_emotion}

### Emotion Scale (0–10)
- 0–2 → **Acceptance (😌)** — Calm, positive, or composed tone.
- 2–4 → **Denial (🤔)** — Defensive, avoiding discomfort.
- 4–6 → **Bargaining (🥺)** — Nervous or uncertain tone, seeking reassurance.
- 6–8 → **Anger (😠)** — Frustration, irritation, or emotional strain.
- 8–10 → **Depression (😔)** — Low tone, sadness, emotional fatigue.

### Instructions:
1. Write a **neutral call summary** (2–4 sentences) about what was discussed (no emotional judgments).
2. Assign a `depression_score` (0–10) based on tone and emotion levels.
3. Explain briefly in `description` which stage fits best.
4. List any `risks` if the emotions suggest high stress or sadness.
5. Give short, warm, practical `advice` (e.g., taking rest, talking to friends).
6. Keep tone supportive and neutral.

### Instructions for Emotion Output
- Only return **these five emotion stages**:
  Acceptance, Denial, Bargaining, Anger, Depression
- Each must have a confidence score between 0.0 and 1.0
- Do not invent or use any other labels
- If the emotion is not strongly detected, you can assign a low score (e.g., 0.0–0.2)
- Make sure the sum of scores does **not need to equal 1**, but each score should reflect the LLM's confidence

### Extremely Important Note:
- Only return these five emotion stages:
  Acceptance, Denial, Bargaining, Anger, Depression
- Do not invent or use any other labels

Return the output strictly in this JSON format:

{format_instructions}
""",
    input_variables=["text_or_emotion"],
    partial_variables={"format_instructions": parser_call.get_format_instructions()},
)


promptTemplate_merge_call = PromptTemplate(
    template="""
You are a senior mental health reviewer. Two reports are provided from a call:
1. A transcript-based analysis.
2. A vocal emotion-based analysis.

Your task is to merge them into one unified, human-like call summary and emotional wellbeing report.

Inputs:
- Transcript Report: {report_asr_call}
- Emotion Report: {report_ver_call}

### Emotion Scale (0–10)
- 0–2 → **Acceptance (😌)** — Calm and balanced.
- 2–4 → **Denial (🤔)** — Avoidance or mild defensiveness.
- 4–6 → **Bargaining (🥺)** — Uncertainty or internal conflict.
- 6–8 → **Anger (😠)** — Tension or frustration.
- 8–10 → **Depression (😔)** — Sadness, hopelessness, or emotional fatigue.

### Instructions:
1. Combine both summaries into one **neutral and factual** summary (2–5 sentences).
2. Normalize the two `depression_score`s into one final value.
3. Write a short `description` explaining the emotional stage and reasoning.
4. Merge and deduplicate all `risks`, `advice`, and `emotions`.
5. Keep everything aligned to the above emotional range.

### Additional Instructions for Emotion Output
- Merge the `emotions` list from both reports
- Only keep the five standard emotions: Acceptance, Denial, Bargaining, Anger, Depression
- Avoid creating new labels
- Keep confidence scores (0–1) and deduplicate if necessary

### Extremely Important Note:
- Only return these five emotion stages:
  Acceptance, Denial, Bargaining, Anger, Depression
- Do not invent or use any other labels

Return the output strictly in this JSON format:

{format_instructions}
""",
    input_variables=["report_asr_call", "report_ver_call"],
    partial_variables={"format_instructions": parser_call.get_format_instructions()},
)


sttta_chain_audio = asr_runnable | promptTemplate_sttta_audio | llm_model | parser_audio
ste_chain_audio = ver_runnable | promptTemplate_ste_audio | llm_model | parser_audio

parallel_chain_audio = RunnableParallel({
    'stta_audio': sttta_chain_audio,
    'ste_audio': ste_chain_audio,
    'transcript_audio': asr_runnable
})
final_chain_audio = parallel_chain_audio | remap_audio | promptTemplate_merge_audio | llm_model | parser_audio


wrap_for_prompt = RunnableLambda(lambda text: {"text_or_emotion": text})
sttta_chain_call = asr_runnable | wrap_for_prompt | promptTemplate_sttta_call | llm_model | parser_call
ste_chain_call = ver_runnable | wrap_for_prompt | promptTemplate_ste_call | llm_model | parser_call

parallel_chain_call = RunnableParallel({
    'stta_call': sttta_chain_call,
    'ste_call': ste_chain_call,
    'summary_call': asr_runnable
})
final_chain_call = parallel_chain_call | remap_call | promptTemplate_merge_call | llm_model | parser_call

# print(final_chain.get_graph().print_ascii())
# result = final_chain.invoke(input("Enter Audio File Name: "))
# print(result.dict())

router_pc = APIRouter()

from pydub import AudioSegment

def fix_audio_format(in_path: str, out_path: str):
    try:
        audio = AudioSegment.from_file(in_path)
    except Exception:
        audio = AudioSegment.from_file(in_path, format="wav")
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(out_path, format="wav")


@router_pc.post("/analyze-journal", response_model=DepressionReport)
async def analyze_audio(file: UploadFile = File(...)):
    tmp_in_path = None
    tmp_out_path = None
    try:
        # 1. Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
            tmp_in.write(await file.read())
            tmp_in_path = tmp_in.name

        # 2. Use pydub to standardize the audio into the correct format for the AI
        audio = AudioSegment.from_file(tmp_in_path)
        audio = audio.set_channels(1)       # Mono
        audio = audio.set_frame_rate(16000) # 16kHz sample rate

        # 3. Export the clean audio to a new temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            audio.export(tmp_out.name, format="wav")
            tmp_out_path = tmp_out.name # This is the clean file we'll use

        # 4. Run the final analysis chain on the clean audio file
        result: DepressionReport = final_chain_audio.invoke(tmp_out_path)

        # 5. Inject the transcript into the result object for consistency
        if result:
             result.transcript = result.transcript or "Transcript not available."

        return JSONResponse(content=result.model_dump())

    except Exception as e:
        print(f"An error occurred during audio analysis: {e}")
        # Return a more informative error to the frontend
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to process audio file: {e}"}
        )
    finally:
        # 6. Clean up both temporary files
        if tmp_in_path and os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if tmp_out_path and os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)


@router_pc.post("/analyze-call", response_model=CallAnalysisReport)
async def analyze_call(file: UploadFile = File(...)):
    tmp_in_path = None
    tmp_out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in:
            tmp_in.write(await file.read())
            tmp_in_path = tmp_in.name

        audio = AudioSegment.from_file(tmp_in_path)
        audio = audio.set_channels(1)       
        audio = audio.set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            audio.export(tmp_out.name, format="wav")
            tmp_out_path = tmp_out.name

        result: CallAnalysisReport = final_chain_call.invoke(tmp_out_path)

        if result:
            result.summary = result.summary or "Summary not available."

        return JSONResponse(content=result.model_dump())

    except Exception as e:
        print(f"An error occurred during audio analysis: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to process audio file: {e}"}
        )
    finally:
        if tmp_in_path and os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if tmp_out_path and os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)
