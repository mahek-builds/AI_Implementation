import os
from unittest import result
from dotenv import load_dotenv
import json

load_dotenv()
from groq import Groq
from pydantic import BaseModel
from typing import List

# interview question generator
class InterviewQuestion(BaseModel):
    topic: str
    difficulty: str
    no_of_questions: int
class InterviewResponse(BaseModel):
    questions: List[str]



my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API_KEY_NOT_FOUND")
client = Groq(api_key=my_api_key)

topic=input("Enter the topic for interview questions: ")
difficulty=input("Enter the difficulty level (easy, medium, hard): ")
no_of_questions=int(input("Enter the number of questions to generate: "))

system_prompt=f"""
You are an expert interviewer,you will generate questions based on {topic}, {difficulty}, and {no_of_questions}"""
user_prompt=f"""
generate answers for the {topic} with {difficulty}level and{no_of_questions}
IMPORTANT:
Return ONLY the JSON object.
Do NOT write any explanation.
Do NOT write any text before or after the JSON.
Do NOT use Markdown code blocks

{{
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "no_of_questions": {no_of_questions}
}}
give output in json format 
{{questions: [question1, question2, question3,...]}}
"""
messages={
    "role":"system",
    "content":system_prompt
}
user_message={
    "role":"user",
    "content":user_prompt
}
messages=[messages,user_message]
model="llama-3.3-70b-versatile"
response=client.chat.completions.create(model=model,messages=messages,temperature=0.7)

answer=response.choices[0].message.content

ans=json.loads(answer)
result=InterviewResponse.model_validate(ans)
for i,question in enumerate(result.questions):
    print(f"{i+1}.{question}")











