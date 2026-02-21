from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
from .clients.rq_client import queue
from .queues.worker import process_query

app = FastAPI()

@app.get("/")
def root():
    return {"server": "Server is up and running"}

@app.post("/chat")
def chat(
    query: str = Query(..., description="The input message"),
):
    job = queue.enqueue(process_query, query)
    
    return {"status": "queued", "job_id": job.get_id()}


@app.get("/job-status")
def get_result(
    job_id: str = Query(..., description="The ID of the job"),
):
    job = queue.fetch_job(job_id)
    result = job.return_value()
    return {"result": result}