import os
from pathlib import Path

from dotenv  import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq



#load variables
load_dotenv()
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")


#CONNECT to Qdrant
client=QdrantClient(
    api_key=QDRANT_API_KEY,
    url=QDRANT_URL

)
print("Connect to Qdrant Client")


#QDRANT COLLECTION

COLLECTION_NAME="knowledge"
EMBEDDING_SIZE=384


#delate collection if exists
if(client.collection_exists(COLLECTION_NAME)):
    client.delete_collection(COLLECTION_NAME)

#create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)
print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")


# LOAD KNOWLEDGE
docs=[]
with open(Path(__file__).with_name("rag.txt"), "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]


#create embeddings
print("Model embeddings...")
model=SentenceTransformer("all-MiniLM-L6-v2")
embedddings=model.encode(docs)
print(f"len{embedddings}")
print(f"len{embedddings[0]}")


#create qdrant client
points=[]
for i,embedding in enumerate(embedddings):
    point=PointStruct(
    id=i+1 ,
    vector=embedding.tolist(),
    payload={
        "text":docs[i]

    })
    points.append(point)

# upsert qdrant(update+insert)
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

# search in qdrant 

def search_query(query,top_k=3):
    query_vector=model.encode(query).tolist()
    results=client.query_points(
        query=query_vector,
        collection_name=COLLECTION_NAME,
        limit=top_k,
        with_payload=True

    )
    return results.points
results=search_query("what are embeddings in rag?")
for result in results:
    print(f"Score={result.score}")
    print(result.payload["text"])

# groq_client
groq_client=Groq(api_key=GROQ_API_KEY)

# ask the llm 
def ask_llm(question,context):
    prompt=f"""answer the following on the basis of information provided:
    Context=
    {context},
    Question=
    {question}

    """
    response=groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role":"user",
             "content":prompt }


        ]

    )
    return response.choices[0].message.content
# complete rag pipeline

question="what are ebeddings in rag in gen ai??"
results=search_query(question,top_k=3)
context="\n".join(result.payload["text"] for result in results)
answer=ask_llm(question,context)
print(answer)

    






















