from fastapi import APIRouter, HTTPException
from database.connection import mongodb

router = APIRouter()

# 1️⃣ Get hospitals under an insurer (by insurer name)
@router.get("/cashless/insurer/{insurer_name}")
async def get_hospitals_by_insurer(insurer_name: str):
    """
    Returns all cashless hospitals under a given insurer by name.
    """
    collection = mongodb.database["cashless"]
    doc = await collection.find_one({"insurer_name": {"$regex": f"^{insurer_name}$", "$options": "i"}})

    if not doc:
        raise HTTPException(status_code=404, detail="Insurer not found")

    hospitals = []
    for city in doc.get("cities", []):
        for hosp in city.get("cashless_hospitals", []):
            hospitals.append({
                "hospital_name": hosp["hospital_name"],
                "city": city["city_name"],
                "address": hosp.get("address", {})
            })

    return {"insurer": doc["insurer_name"], "hospitals": hospitals}


# 2️⃣ Get hospitals in a particular city
@router.get("/cashless/city/{city_name}")
async def get_hospitals_by_city(city_name: str):
    """
    Returns all cashless hospitals in a city with insurer name.
    """
    collection = mongodb.database["cashless"]
    cursor = collection.find({"cities.city_name": {"$regex": f"^{city_name}$", "$options": "i"}})
    docs = await cursor.to_list(length=None)

    if not docs:
        raise HTTPException(status_code=404, detail="No hospitals found in this city")

    hospitals = []
    for doc in docs:
        insurer_name = doc["insurer_name"]
        for city in doc["cities"]:
            if city["city_name"].lower() == city_name.lower():
                for hosp in city.get("cashless_hospitals", []):
                    hospitals.append({
                        "hospital_name": hosp["hospital_name"],
                        "address": hosp.get("address", {}),
                        "insurer": insurer_name
                    })

    return {"city": city_name, "hospitals": hospitals}


# 3️⃣ Get all details
@router.get("/cashless/all")
async def get_all_cashless_details():
    """
    Returns insurer name, city, hospital, and address for all entries.
    """
    collection = mongodb.database["cashless"]
    cursor = collection.find({})
    docs = await cursor.to_list(length=None)

    details = []
    for doc in docs:
        insurer_name = doc["insurer_name"]
        for city in doc.get("cities", []):
            for hosp in city.get("cashless_hospitals", []):
                details.append({
                    "insurer": insurer_name,
                    "city": city["city_name"],
                    "hospital": hosp["hospital_name"],
                    "address": hosp.get("address", {})
                })

    return {"cashless_network": details}
