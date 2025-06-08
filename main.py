# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# Load saved embeddings and dataframe
resume_embeddings = joblib.load('resume_embeddings.pkl')
job_desc_embeddings = joblib.load('job_desc_embeddings.pkl')
df = joblib.load('resume_df.pkl')

# Load your embedding function and model
from my_utils import embed_long_text  # assume this is defined in my_utils.py

app = FastAPI()

class ResumeRequest(BaseModel):
    resume_text: str
    category: str  # used to fetch job description

@app.post("/match_resume")
def match_resume(data: ResumeRequest):
    if data.category not in df['Category'].values:
        raise HTTPException(status_code=404, detail="Category not found")

    job_desc = df[df['Category'] == data.category]['job_desc_text'].values[0]
    job_embedding = embed_long_text(job_desc)
    resume_embedding = embed_long_text(data.resume_text)

    sim_score = cosine_similarity(
        np.array(resume_embedding).reshape(1, -1),
        np.array(job_embedding).reshape(1, -1)
    )[0][0]

    return {
        "similarity_score": float(sim_score),
        "matched_category": data.category
    }


