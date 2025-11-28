from fastapi import APIRouter, HTTPException
from database.connection import mongodb

router = APIRouter()
COLLECTION = "branches"


def _ensure_db_connected():
    if mongodb.database is None:
        raise HTTPException(status_code=503, detail="Database not connected")


# 1️⃣ Get all branches of an insurer
@router.get("/company/{company_name}")
async def get_branches_by_company(company_name: str):
    _ensure_db_connected()
    collection = mongodb.database[COLLECTION]

    regex = {"$regex": company_name, "$options": "i"}
    doc = await collection.find_one({"Company": regex}, {"_id": 0})

    if not doc:
        raise HTTPException(status_code=404, detail="Company not found")

    return doc


# 2️⃣ Get branches of a specific city under a specific insurer
@router.get("/company/{company_name}/city/{city_name}")
async def get_branches_by_city_and_company(company_name: str, city_name: str):
    _ensure_db_connected()
    collection = mongodb.database[COLLECTION]

    regex_company = {"$regex": company_name, "$options": "i"}
    regex_city = {"$regex": city_name, "$options": "i"}

    doc = await collection.find_one({"Company": regex_company}, {"_id": 0})

    if not doc:
        raise HTTPException(status_code=404, detail="Company not found")

    branches = [
        branch for branch in doc.get("branches", [])
        if regex_city["$regex"].lower() in branch.get("city", "").lower()
    ]

    if not branches:
        raise HTTPException(status_code=404, detail="City not found under this company")

    return {"company": doc["Company"], "city": city_name, "branches": branches}
