# routes/recommender.py
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Any
from dotenv import load_dotenv
import numpy as np
from modules.policy_recommender import PolicyRecommender

# Load .env file
load_dotenv()

router = APIRouter(tags=["Policy Recommender"])

class RecommendationRequest(BaseModel):
    age: int
    budget: float
    city: str
    category: str
    coverage_requirement: str
    top_n: int = 5


def convert_numpy_types(obj: Any):
    """Recursively convert numpy data types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj


@router.post("/recommend-policies")
async def recommend_policies(request: RecommendationRequest):
    """
    Recommend top insurance policies based on user's age, budget, city, category, and coverage needs.
    """
    try:
        mongo_uri = os.getenv("MONGODB_URL")
        if not mongo_uri:
            raise ValueError("MONGODB_URL is not set in environment variables.")

        recommender = PolicyRecommender(
            mongo_uri=mongo_uri,
            db_name="AskMyPolicy",
            collection_name="compare"
        )

        recommendations = recommender.get_recommendations(
            user_age=request.age,
            user_budget=request.budget,
            user_city=request.city,
            policy_category=request.category,
            user_coverage_requirement=request.coverage_requirement,
            top_n=request.top_n
        )

        recommender.close()

        # 🔧 Convert all NumPy types to JSON-safe values
        safe_recommendations = convert_numpy_types(recommendations)

        return {"recommendations": safe_recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
