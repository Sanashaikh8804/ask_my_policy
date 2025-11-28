import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

# Import existing agent and comparer
from src.Ai_Agent import InsuranceAgent
from comparison.compare import PolicyComparer
from routes import recommender

# Import your database connection and routes
from database.connection import mongodb
from routes import cashless, branches

# ----------------------------- Load Environment -----------------------------
load_dotenv()

# ----------------------------- FastAPI Setup -----------------------------
app = FastAPI(
    title="Insurance Agent API",
    description="An API for insurance-related services — Chat, Comparison, Cashless, and Branch Info.",
    version="2.0.0"
)

# Allow CORS (frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Environment Variables -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_CONNECTION = os.getenv("MONGODB_CONNECTION")

if not GROQ_API_KEY or not MONGODB_CONNECTION:
    raise ValueError("❌ GROQ_API_KEY or MONGODB_CONNECTION is missing in environment variables!")

# ----------------------------- Core AI Components -----------------------------
agent = InsuranceAgent(
    groq_api_key=GROQ_API_KEY,
    mongodb_connection=MONGODB_CONNECTION,
    database_name="AskMyPolicy"
)

# ----------------------------- Request Models -----------------------------
class QueryRequest(BaseModel):
    query: str

class PolicyNamesRequest(BaseModel):
    policy_names: List[str]

# ----------------------------- Base & Core Endpoints -----------------------------
@app.get("/")
def root():
    return {"status": "✅ Insurance Agent API is running!"}

@app.post("/ask", tags=["AI Chat"])
async def ask_agent(request: QueryRequest) -> dict:
    """
    Receives a query and returns a response from the AI agent.
    """
    try:
        response_text = await agent.ask(user_message=request.query)
        return {"answer": response_text}
    except Exception as e:
        print(f"❌ Error in /ask: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.post("/compare", tags=["Policy Comparison"])
async def compare_policies(request: PolicyNamesRequest):
    """
    Accepts policy names and returns a detailed AI-based comparison.
    """
    if not request.policy_names:
        raise HTTPException(status_code=400, detail="The 'policy_names' list cannot be empty.")
    
    try:
        comparer = PolicyComparer(policy_names=request.policy_names)
        result = await comparer.compare()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# ----------------------------- MongoDB Lifecycle -----------------------------
@app.on_event("startup")
async def startup_event():
    await mongodb.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await mongodb.close()

# ----------------------------- Integrated APIs -----------------------------
# Cashless Network API
app.include_router(cashless.router, prefix="/api", tags=["Cashless Network"])

# Ombudsman / Branches API
app.include_router(branches.router, prefix="/api", tags=["Ombudsman Offices"])

# Recommender System API
app.include_router(recommender.router, prefix="/api")


# ----------------------------- Run Server -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
