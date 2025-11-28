# Ai_Agent.py

from .GroqChatCompletion import GroqChatCompletion
import os
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
import json
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

from pymongo import MongoClient
from pymongo.collection import Collection
from bson import ObjectId

load_dotenv()


class MongoDBService:
    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.policies_collection: Collection = self.db["compare"]
    
    async def get_all_policies(self) -> List[dict]:
        return list(self.policies_collection.find({"isActive": True}))
    
    async def get_policy_by_name(self, policy_name: str) -> Optional[dict]:
        return self.policies_collection.find_one(
            {"policyName": {"$regex": policy_name, "$options": "i"}, "isActive": True}
        )
    
    async def get_policy_coverage(self, policy_name: str) -> Optional[dict]:
        """Get coverage details for a specific policy"""
        policy = await self.get_policy_by_name(policy_name)
        if policy:
            return {
                "policyName": policy.get("policyName"),
                "coverage": policy.get("coverage", {}),
                "specialCoverages": policy.get("specialCoverages", []),
                "addOns_OptionalBenefits": policy.get("addOns_OptionalBenefits", [])
            }
        return None
    
    async def get_policy_premiums(self, policy_name: str, city: str = None, age: int = None) -> Optional[dict]:
        """Get premium details for a specific policy"""
        policy = await self.get_policy_by_name(policy_name)
        if not policy:
            return None
        
        premiums = policy.get("premiums", [])
        result = {
            "policyName": policy.get("policyName"),
            "allPremiums": premiums
        }
        
        # Filter by city if provided
        if city:
            for zone in premiums:
                if city.lower() in [c.lower() for c in zone.get("cities", [])]:
                    result["applicableZone"] = zone
                    break
        
        return result
    
    async def get_policy_waiting_periods(self, policy_name: str) -> Optional[dict]:
        """Get waiting periods for a specific policy"""
        policy = await self.get_policy_by_name(policy_name)
        if policy:
            return {
                "policyName": policy.get("policyName"),
                "waitingPeriods": policy.get("waitingPeriods", {})
            }
        return None
    
    async def get_policy_exclusions(self, policy_name: str) -> Optional[dict]:
        """Get exclusions for a specific policy"""
        policy = await self.get_policy_by_name(policy_name)
        if policy:
            return {
                "policyName": policy.get("policyName"),
                "exclusions": policy.get("exclusions", {})
            }
        return None
    
    async def search_policies_by_coverage(self, coverage_type: str) -> List[dict]:
        """Search policies that offer specific coverage"""
        policies = list(self.policies_collection.find({"isActive": True}))
        matching_policies = []
        
        coverage_lower = coverage_type.lower()
        
        for policy in policies:
            # Search in coverage object
            coverage_str = json.dumps(policy.get("coverage", {})).lower()
            
            # Search in special coverages
            special_coverages = policy.get("specialCoverages", [])
            special_str = json.dumps(special_coverages).lower()
            
            # Search in add-ons
            addons = policy.get("addOns_OptionalBenefits", [])
            addons_str = json.dumps(addons).lower()
            
            if (coverage_lower in coverage_str or 
                coverage_lower in special_str or 
                coverage_lower in addons_str):
                matching_policies.append({
                    "policyName": policy.get("policyName"),
                    "insurer": policy.get("insurer"),
                    "code": policy.get("code")
                })
        
        return matching_policies


class PolicyPlugin:
    def __init__(self, mongo_service: MongoDBService):
        self.mongo_service = mongo_service
    
    @kernel_function(
        name="get_all_policies",
        description="Retrieves list of all active insurance policies with basic information (name, insurer, code)"
    )
    async def get_all_policies(self) -> str:
        policies = await self.mongo_service.get_all_policies()
        simplified = []
        for policy in policies:
            simplified.append({
                "policyName": policy.get("policyName"),
                "insurer": policy.get("insurer"),
                "code": policy.get("code"),
                "policyType": policy.get("policyType")
            })
        return json.dumps(simplified, indent=2)
    
    @kernel_function(
        name="get_policy_by_name",
        description="Retrieves complete details of a specific insurance policy by name"
    )
    async def get_policy_by_name(self, policy_name: str) -> str:
        policy = await self.mongo_service.get_policy_by_name(policy_name)
        if policy is None:
            return f"Policy '{policy_name}' not found"
        
        if "_id" in policy:
            policy["_id"] = str(policy["_id"])
        return json.dumps(policy, indent=2, default=str)
    
    @kernel_function(
        name="get_policy_coverage_details",
        description="Get detailed coverage information including base coverage, special coverages, and optional add-ons for a policy. Use this for questions about what is covered, benefits, treatments, diseases covered, etc."
    )
    async def get_policy_coverage_details(self, policy_name: str) -> str:
        coverage = await self.mongo_service.get_policy_coverage(policy_name)
        if coverage is None:
            return f"Coverage information not found for '{policy_name}'"
        return json.dumps(coverage, indent=2, default=str)
    
    @kernel_function(
        name="get_policy_premium_details",
        description="Get premium information for a policy including zone-wise pricing, sum insured options, and age brackets. Optionally filter by city and age."
    )
    async def get_policy_premium_details(self, policy_name: str, city: str = None, age: int = None) -> str:
        premiums = await self.mongo_service.get_policy_premiums(policy_name, city, age)
        if premiums is None:
            return f"Premium information not found for '{policy_name}'"
        return json.dumps(premiums, indent=2, default=str)
    
    @kernel_function(
        name="get_policy_waiting_periods",
        description="Get waiting period information for a policy including initial waiting period, pre-existing disease waiting period, and specific ailment waiting periods"
    )
    async def get_policy_waiting_periods(self, policy_name: str) -> str:
        waiting = await self.mongo_service.get_policy_waiting_periods(policy_name)
        if waiting is None:
            return f"Waiting period information not found for '{policy_name}'"
        return json.dumps(waiting, indent=2, default=str)
    
    @kernel_function(
        name="get_policy_exclusions",
        description="Get exclusion information for a policy - what is NOT covered, permanent exclusions, and specific exclusions"
    )
    async def get_policy_exclusions(self, policy_name: str) -> str:
        exclusions = await self.mongo_service.get_policy_exclusions(policy_name)
        if exclusions is None:
            return f"Exclusion information not found for '{policy_name}'"
        return json.dumps(exclusions, indent=2, default=str)
    
    @kernel_function(
        name="search_policies_by_coverage",
        description="Search for policies that offer a specific type of coverage (e.g., 'cancer', 'mental illness', 'maternity', 'HIV')"
    )
    async def search_policies_by_coverage(self, coverage_type: str) -> str:
        policies = await self.mongo_service.search_policies_by_coverage(coverage_type)
        if not policies:
            return f"No policies found with '{coverage_type}' coverage"
        return json.dumps(policies, indent=2)


class InsuranceAgent:
    def __init__(
        self,
        groq_api_key: str,
        mongodb_connection: str,
        database_name: str = "AskMyPolicy",
        model_id: str = "llama-3.3-70b-versatile"
    ):
        self.mongo_service = MongoDBService(mongodb_connection, database_name)
        self.kernel = Kernel()
        
        # Add Groq chat completion service
        self.chat_service = GroqChatCompletion(
            api_key=groq_api_key,
            model_id=model_id
        )
        
        # Add policy plugin
        policy_plugin = PolicyPlugin(self.mongo_service)
        self.kernel.add_plugin(policy_plugin, "PolicyPlugin")
        
        self.system_prompt = """You are an intelligent health insurance policy assistant. Your goal is to provide accurate, specific information from the policy database.

IMPORTANT GUIDELINES:
1. ALWAYS use the appropriate PolicyPlugin functions to retrieve accurate data.
2. For coverage questions: Use get_policy_coverage_details to get complete coverage information.
3. For premium questions: Use get_policy_premium_details.
4. For waiting period questions: Use get_policy_waiting_periods.
5. For exclusion questions: Use get_policy_exclusions.
6. For comparing coverage: Use search_policies_by_coverage.

UNDERSTANDING THE SCHEMA:
- coverage: Contains nested objects like inPatientHospitalization, preHospitalization, postHospitalization, dayCare, ambulanceCover, cumulativeBonus, advancedTechnology, etc.
- specialCoverages: Array of special disease coverages (HIV/AIDS, Mental Illness, Chronic conditions, etc.) with fields: diseaseName, isCovered, coverageType, limit, conditions.
- addOns_OptionalBenefits: Array of optional benefits (Cancer Booster, Critical Illness, Personal Accident, etc.) with fields: name, isAvailable, details, conditions.
- premiums: Array of zones, each containing cities and premiumChart with sumInsured options and age brackets.
- waitingPeriods: Contains initialDays, preExistingDiseaseMonths, and specificAilments array.
- exclusions: Contains permanent exclusions array and specific exclusions array.

RESPONSE FORMAT:
- Answer in 10-15 lines maximum.
- Be specific with numbers, limits, and conditions.
- Cite the policy name.
- If something is not covered in base policy, mention if it's available as an add-on.
- For coverage questions, clearly state YES/NO and provide details.

SYNONYMS TO UNDERSTAND:
- 'waiting period' = 'grace period' = 'initial waiting'
- 'coverage' = 'benefits' = 'what is covered'
- 'exclusions' = 'not covered' = 'what is excluded'
- 'premium' = 'cost' = 'price'
- 'add-ons' = 'riders' = 'optional benefits'

Always retrieve fresh data from the database using the functions."""

    async def ask(self, user_message: str) -> str:
        """
        Processes a single user query without maintaining chat history.
        """
        chat_history = ChatHistory()
        chat_history.add_system_message(self.system_prompt)
        chat_history.add_user_message(user_message)
        
        # Configure execution settings
        execution_settings = PromptExecutionSettings(
            service_id="groq_chat",
            max_tokens=800,
            temperature=0.3
        )
        execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        
        # Get response
        response = await self.chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=self.kernel
        )
        
        return str(response)


async def run():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MONGODB_CONNECTION = os.getenv("MONGODB_CONNECTION")
    
    agent = InsuranceAgent(
        groq_api_key=GROQ_API_KEY,
        mongodb_connection=MONGODB_CONNECTION,
        database_name="AskMyPolicy"
    )
    
    print("=" * 60)
    print("Health Insurance Policy AI Agent (Stateless Mode)")
    print("=" * 60)
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            response = await agent.ask(user_message=user_input)
            
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    asyncio.run(run())