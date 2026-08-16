import os
from dotenv import load_dotenv 
load_dotenv()
from groq import Groq
from pydantic import BaseModel
import json

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API key not found")
client = Groq(api_key=my_api_key)

#Planner Agent
class interviewRequest(BaseModel):
    role: str
    topic: str
    experience: int


class interviewResponse(BaseModel):
    difficulty: str
    num_questions: int
    question_types: list[str]

system_prompt = """
You are an expert interviewer. You must decide the difficulty, question_types, and num_questions based on the candidate's role, topic, and experience.
Respond in JSON like this:
{
    "difficulty": "medium",
    "question_types": ["multiple choice", "coding", "behavioral"],
    "num_questions": 5
}
IMPORTANT:
Return ONLY the JSON object.
Do NOT write any explanation.
Do NOT write any text before or after the JSON.
Do NOT use Markdown code blocks
"""
model="llama-3.3-70b-versatile"
def ask_llm(interview_request: interviewRequest):
    user_prompt=f"""
role:{interview_request.role}
topic:{interview_request.topic}
experience={interview_request.experience}"""
    prompt=[user_prompt, system_prompt]
    user_messages={
        "role": "user",
        "content": user_prompt
    }
    system_messages={
        "role": "system", 
        "content": system_prompt
    }
    messages=[system_messages, user_messages]
    response=client.chat.completions.create(
        model=model,messages=messages,temperature=0.7
    )
    answer=response.choices[0].message.content
    a=json.loads(answer)
    return interviewResponse.model_validate(a)
request = interviewRequest(
    role="AI Engineer",
    topic="RAG",
    experience=1
)

plan=ask_llm(request)


#generator
def generate_answer(interview_response: interviewResponse):
    systemm_prompt = """
You are an expert interviewer. Generate questions according to input provided in the form of:
{
    "difficulty": "medium",
    "question_types": ["multiple choice", "coding", "behavioral"],
    "num_questions": 5
}
response should be in json format 
{
"questions":["1.what is rag",,.....]
}
MPORTANT:
Return ONLY the JSON object.
Do NOT write any explanation.
Do NOT write any text before or after the JSON.
Do NOT use Markdown code blocks
"""

    user_prompt = f"""
difficulty: {interview_response.difficulty}
question_types: {interview_response.question_types}
num_questions: {interview_response.num_questions}
"""

    system_messages = {
        "role": "system",
        "content": systemm_prompt
    }
    user_messages = {
        "role": "user",
        "content": user_prompt
    }
    messages = [system_messages, user_messages]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.7,stream=True)
    full_response=""
    content=""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content
    print(full_response)


generate_answer(plan)

