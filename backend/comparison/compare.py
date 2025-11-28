# compare.py

import asyncio
import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from .GroqChatCompletion import GroqChatCompletion

# Load environment variables from a .env file
load_dotenv()

class MongoDBService:
    """
    Service to interact with the MongoDB database for policy data.
    """
    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        # Accessing the 'compare' collection as specified
        self.policies_collection: Collection = self.db["compare"]

    async def get_policy_by_name(self, policy_name: str) -> Dict[str, Any] | None:
        """
        Fetches a single policy document by its name using a case-insensitive regex match.
        """
        # Using regex for flexible matching (e.g., "reassure 3.0" matches "ReAssure 3.0")
        return self.policies_collection.find_one(
            {"policyName": {"$regex": f"^{policy_name}$", "$options": "i"}, "isActive": True}
        )

class PolicyComparer:
    """
    Compares health insurance policies by fetching their details, structuring them,
    and generating an AI-powered analysis.
    """
    def __init__(self, policy_names: List[str]):
        # Ensure environment variables are loaded
        mongodb_connection = os.getenv("MONGODB_CONNECTION")
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not mongodb_connection or not groq_api_key:
            raise ValueError("MONGODB_CONNECTION and GROQ_API_KEY must be set in the .env file.")

        self.policy_names = policy_names
        self.mongo_service = MongoDBService(mongodb_connection, "AskMyPolicy")
        self.ai_client = GroqChatCompletion(api_key=groq_api_key, model_id="llama-3.3-70b-versatile")

    def _check_maternity_cover(self, policy: Dict[str, Any]) -> bool:
        """
        Checks if maternity cover is available either as a base benefit or an add-on.
        """
        # Check in special coverages (base benefit)
        for cover in policy.get('specialCoverages', []):
            if 'maternity' in cover.get('diseaseName', '').lower():
                return True
        # Check in optional add-ons
        for addon in policy.get('addOns_OptionalBenefits', []):
            if 'maternity' in addon.get('name', '').lower():
                return True
        return False

    def _extract_policy_features(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts and formats the required attributes from a raw policy document.
        """
        if not policy:
            return {"error": "Policy data not found."}
            
        coverage = policy.get('coverage', {})
        waiting_periods = policy.get('waitingPeriods', {})

        return {
            "preHospitalizationDays": coverage.get('preHospitalization', {}).get('durationDays'),
            "postHospitalizationDays": coverage.get('postHospitalization', {}).get('durationDays'),
            "sumInsuredRestoration": coverage.get('sumInsuredRestoration', {}).get('isAvailable', False),
            "waitingPeriodInitialDays": waiting_periods.get('initialDays'),
            "waitingPeriodPEDMonths": waiting_periods.get('preExistingDiseaseMonths'),
            "advancedTreatmentsCovered": coverage.get('advancedTechnology', {}).get('isCovered', False),
            "discountsAvailable": bool(policy.get('discounts')),
            "maternityCover": self._check_maternity_cover(policy),
            "specialCovers": [cov['diseaseName'] for cov in policy.get('specialCoverages', [])],
            "roomRentCover": coverage.get('inPatientHospitalization', {}).get('roomRentLimit', 'Not specified'),
            "dayCareCover": coverage.get('dayCare', {}).get('isCovered', False),
            "ambulanceCover": coverage.get('ambulanceCover', {}).get('isCovered', False),
            "optionalBenefits": [opt['name'] for opt in policy.get('addOns_OptionalBenefits', [])]
        }

    async def _generate_ai_analysis(self, comparison_data: Dict[str, Any]) -> str:
        """
        Sends the structured comparison data to an AI model for analysis.
        """
        system_prompt = """You are a senior health insurance analyst. Your task is to provide a clear, concise, and expert comparison of the health insurance policies based ONLY on the structured JSON data provided.

        **Instructions:**
        1.  **Analyze the Data**: Compare the policies across key features like restoration benefits, hospitalization coverage days, waiting periods, room rent limits, and unique special covers.
        2.  **Highlight Strengths**: For each policy, identify its strongest features. For example, one might have a better restoration benefit, while another has a shorter waiting period for specific diseases or more comprehensive special covers.
        3.  **Identify Weaknesses**: Point out any potential drawbacks, such as restrictive room rent limits, longer waiting periods, or lack of certain wellness benefits.
        4.  **Provide a Conclusion**: Conclude with a final recommendation, suggesting which policy offers the best overall value and for what kind of customer profile (e.g., young individuals, families, seniors, people with chronic conditions).
        5.  **Strict Constraints**: Do NOT use any information outside of the provided JSON. Keep the total analysis strictly under 300 words. Be objective and data-driven."""

        # Using a simple ChatHistory-like structure for the AI model
        chat_history = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the policy comparison data:\n\n{json.dumps(comparison_data, indent=2)}"}
            ]
        }
        
        try:
            analysis = await self.ai_client.get_chat_message_content(chat_history)
            return analysis
        except Exception as e:
            return f"Failed to generate AI analysis: {e}"

    async def compare(self) -> Dict[str, Any]:
        """
        The main method to perform the comparison and generate the final output.
        """
        comparison_result = {}
        for name in self.policy_names:
            policy_data = await self.mongo_service.get_policy_by_name(name)
            if policy_data:
                comparison_result[policy_data['policyName']] = self._extract_policy_features(policy_data)
            else:
                comparison_result[name] = {"error": "Policy not found in the database."}

        # Generate AI analysis based on the successfully fetched policies
        valid_policies_data = {k: v for k, v in comparison_result.items() if "error" not in v}
        
        ai_analysis_text = "No valid policies found to compare."
        if valid_policies_data:
            ai_analysis_text = await self._generate_ai_analysis(valid_policies_data)
        
        # Structure the final output
        final_output = {
            "policy_comparison": comparison_result,
            "ai_analysis": ai_analysis_text
        }
        
        return final_output

async def main():
    """
    Main function to execute the policy comparison.
    """
    # --- Define the policies you want to compare here ---
    policies_to_compare = [
        "ReAssure 3.0",
        "Activ One",
        "Care Advantage"
    ]

    print(f"🔍 Comparing policies: {', '.join(policies_to_compare)}...")
    
    comparer = PolicyComparer(policy_names=policies_to_compare)
    result = await comparer.compare()

    print("\n" + "="*80)
    print("                      POLICY COMPARISON RESULT")
    print("="*80 + "\n")
    print(json.dumps(result, indent=4))
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())