from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class PolicyRecommender:
    def __init__(self, mongo_uri, db_name="AskMyPolicy", collection_name="compare"):
        """
        Initialize the Policy Recommender
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
            collection_name: Collection name for policies
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Load NLP model for text similarity
        print("Loading NLP model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    
    def flatten_dict(self, d, parent_key='', sep=' '):
        """Flatten nested dictionary into readable text"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if all(isinstance(item, dict) for item in v):
                    for item in v:
                        items.extend(self.flatten_dict(item, new_key, sep=sep).items())
                else:
                    items.append((new_key, ', '.join(str(x) for x in v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def extract_coverage_details(self, policy):
        """
        Extract structured coverage information from policy
        
        Returns:
            dict: Structured coverage data for analysis
        """
        coverage_data = {
            'text': '',
            'features': [],
            'special_coverages': [],
            'addons': [],
            'raw_coverage': policy.get('coverage', {}),
            'raw_special': policy.get('specialCoverages', []),
            'raw_addons': policy.get('addOns_OptionalBenefits', [])
        }
        
        # Extract from coverage object
        if 'coverage' in policy and isinstance(policy['coverage'], dict):
            flattened = self.flatten_dict(policy['coverage'])
            for key, value in flattened.items():
                feature = f"{key.replace('_', ' ')}: {value}"
                coverage_data['features'].append(feature)
        
        # Extract special coverages
        if 'specialCoverages' in policy and isinstance(policy['specialCoverages'], list):
            for special in policy['specialCoverages']:
                if isinstance(special, dict) and special.get('isCovered'):
                    coverage_data['special_coverages'].append(special)
        
        # Extract addons
        if 'addOns_OptionalBenefits' in policy and isinstance(policy['addOns_OptionalBenefits'], list):
            for addon in policy['addOns_OptionalBenefits']:
                if isinstance(addon, dict) and addon.get('isAvailable'):
                    coverage_data['addons'].append(addon)
        
        # Create combined text for NLP
        all_text = []
        all_text.extend(coverage_data['features'])
        
        for sc in coverage_data['special_coverages']:
            all_text.append(f"{sc.get('diseaseName', '')} {sc.get('details', '')} {sc.get('limit', '')}")
        
        for addon in coverage_data['addons']:
            all_text.append(f"{addon.get('name', '')} {addon.get('details', '')}")
        
        coverage_data['text'] = ' '.join(all_text)
        
        return coverage_data
    
    def keyword_matching(self, user_requirement, coverage_data):
        """
        Check for specific keyword matches in coverage
        
        Returns:
            dict: Matched features and score
        """
        # Normalize text
        user_req_lower = user_requirement.lower()
        coverage_text_lower = coverage_data['text'].lower()
        
        # Define keyword mappings
        keyword_map = {
            'hiv': ['hiv', 'aids', 'hiv/aids', 'std'],
            'cancer': ['cancer', 'oncology', 'chemotherapy', 'malignancy'],
            'mental': ['mental', 'psychiatric', 'psychology'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
            'hospitalization': ['hospitalization', 'inpatient', 'hospital'],
            'critical illness': ['critical illness', 'critical disease'],
            'accident': ['accident', 'accidental', 'injury'],
            'ambulance': ['ambulance', 'emergency transport'],
            'daycare': ['day care', 'daycare', 'day treatment'],
            'domiciliary': ['domiciliary', 'home treatment'],
            'cashless': ['cashless', 'network hospital'],
            'pre hospitalization': ['pre hospitalization', 'pre-hospitalization'],
            'post hospitalization': ['post hospitalization', 'post-hospitalization']
        }
        
        matched_features = []
        keyword_score = 0
        total_keywords = 0
        
        # Check each keyword category
        for category, keywords in keyword_map.items():
            # Check if user mentioned this category
            if any(kw in user_req_lower for kw in keywords):
                total_keywords += 1
                # Check if policy covers this
                if any(kw in coverage_text_lower for kw in keywords):
                    keyword_score += 1
                    matched_features.append(category)
        
        match_percentage = (keyword_score / total_keywords * 100) if total_keywords > 0 else 0
        
        return {
            'matched_features': matched_features,
            'keyword_score': keyword_score,
            'total_keywords': total_keywords,
            'match_percentage': match_percentage
        }
    
    def generate_coverage_summary(self, user_requirement, coverage_data, keyword_match):
        """
        Generate a human-like 4-5 line summary of coverage availability
        
        Returns:
            str: Natural language coverage summary
        """
        matched = keyword_match['matched_features']
        total = keyword_match['total_keywords']
        
        # Extract what user is specifically looking for
        user_req_lower = user_requirement.lower()
        
        # Check each requested coverage and provide clear response
        coverage_responses = []
        not_covered = []
        
        # Define coverage checks
        coverage_checks = {
            'HIV/AIDS': ['hiv', 'aids'],
            'Cancer': ['cancer', 'oncology', 'chemotherapy'],
            'Mental Illness': ['mental', 'psychiatric'],
            'Maternity': ['maternity', 'pregnancy'],
            'Critical Illness': ['critical illness'],
            'Accident': ['accident', 'accidental'],
            'Hospitalization': ['hospitalization', 'inpatient']
        }
        
        coverage_text_lower = coverage_data['text'].lower()
        
        for coverage_name, keywords in coverage_checks.items():
            # Check if user asked for this coverage
            if any(kw in user_req_lower for kw in keywords):
                # Check if policy covers it
                if any(kw in coverage_text_lower for kw in keywords):
                    # Find specific details
                    details = None
                    for sc in coverage_data['special_coverages']:
                        sc_name = sc.get('diseaseName', '').lower()
                        if any(kw in sc_name for kw in keywords):
                            limit = sc.get('limit', '')
                            if limit:
                                details = limit
                            break
                    
                    if details:
                        coverage_responses.append(f"✓ Yes, {coverage_name} is covered ({details})")
                    else:
                        coverage_responses.append(f"✓ Yes, {coverage_name} is covered")
                else:
                    not_covered.append(coverage_name)
        
        # Build summary
        if not coverage_responses and not not_covered:
            summary = "This policy provides standard health insurance coverage including in-patient hospitalization, pre and post hospitalization expenses, day care procedures, and ambulance services. "
            summary += "However, it may not specifically cover the exact features you mentioned. "
            summary += "Please review the policy details or consider exploring riders and optional add-ons to customize your coverage."
            return summary
        
        summary = ""
        
        # Add covered items
        if coverage_responses:
            if len(coverage_responses) == 1:
                summary += coverage_responses[0] + ". "
            elif len(coverage_responses) == 2:
                summary += coverage_responses[0] + " " + coverage_responses[1] + ". "
            else:
                summary += " ".join(coverage_responses[:3]) + ". "
        
        # Add not covered items
        if not_covered:
            if len(not_covered) == 1:
                summary += f"✗ However, {not_covered[0]} is not covered under the base policy. "
            else:
                summary += f"✗ However, {', '.join(not_covered[:-1])} and {not_covered[-1]} are not covered under the base policy. "
            
            summary += "You may explore optional riders and add-ons to enhance your coverage for these features."
        
        # Add general policy info
        if coverage_data['addons']:
            addon_names = [a.get('name', '') for a in coverage_data['addons'][:2]]
            if addon_names:
                summary += f" Available add-ons include {', '.join(addon_names)}."
        
        return summary
    
    def parse_age_bracket(self, age_string):
        """Parse age bracket string"""
        if '+' in age_string:
            min_age = int(age_string.replace('+', '').strip())
            return (min_age, 999)
        elif '-' in age_string:
            parts = age_string.split('-')
            return (int(parts[0].strip()), int(parts[1].strip()))
        else:
            age = int(age_string.strip())
            return (age, age)
    
    def find_applicable_premium(self, premiums, user_age, user_city, policy_category):
        """Find applicable premium based on criteria"""
        if not premiums or not isinstance(premiums, list):
            return None
        
        best_matches = []
        
        for zone_data in premiums:
            if not isinstance(zone_data, dict):
                continue
                
            cities = zone_data.get('cities', [])
            city_found = False
            for city in cities:
                if city.lower() == user_city.lower() or user_city.lower() in city.lower():
                    city_found = True
                    break
            
            if not city_found:
                continue
            
            zone_name = zone_data.get('zoneName', 'Unknown')
            premium_chart = zone_data.get('premiumChart', [])
            
            for chart_item in premium_chart:
                if not isinstance(chart_item, dict):
                    continue
                
                sum_insured = chart_item.get('sumInsured', 0)
                premium_options = chart_item.get('premiumOptions', [])
                
                for option in premium_options:
                    if not isinstance(option, dict):
                        continue
                    
                    option_type = option.get('type', '')
                    if option_type != policy_category:
                        continue
                    
                    composition = option.get('composition', '')
                    age_brackets = option.get('ageBrackets', [])
                    
                    for bracket in age_brackets:
                        if not isinstance(bracket, dict):
                            continue
                        
                        age_str = bracket.get('age', '')
                        premium_amount = bracket.get('premiumAmount', 0)
                        
                        try:
                            min_age, max_age = self.parse_age_bracket(age_str)
                            
                            if min_age <= user_age <= max_age:
                                best_matches.append({
                                    'premium_amount': premium_amount,
                                    'sum_insured': sum_insured,
                                    'zone': zone_name,
                                    'composition': composition,
                                    'age_bracket': age_str
                                })
                        except:
                            continue
        
        if best_matches:
            return min(best_matches, key=lambda x: x['premium_amount'])
        
        return None
    
    def get_recommendations(self, user_age, user_budget, user_city, policy_category, 
                          user_coverage_requirement, top_n=5):
        """Get policy recommendations"""
        print(f"\n{'='*70}")
        print(f"Fetching recommendations for:")
        print(f"  Age: {user_age}")
        print(f"  Budget: ₹{user_budget}")
        print(f"  City: {user_city}")
        print(f"  Category: {policy_category}")
        print(f"  Coverage Required: {user_coverage_requirement}")
        print(f"{'='*70}\n")
        
        # Fetch active policies
        active_policies = list(self.collection.find({"isActive": True}))
        print(f"Found {len(active_policies)} active policies in database")
        
        if not active_policies:
            print("No active policies found")
            return []
        
        # Filter by age, city, category, and budget
        filtered_policies = []
        for policy in active_policies:
            premiums = policy.get('premiums', [])
            applicable_premium_info = self.find_applicable_premium(
                premiums, user_age, user_city, policy_category
            )
            
            if applicable_premium_info is not None:
                premium_amount = applicable_premium_info['premium_amount']
                
                if premium_amount <= user_budget:
                    policy['premium_info'] = applicable_premium_info
                    filtered_policies.append(policy)
        
        print(f"After filtering (age, city, category, budget): {len(filtered_policies)} policies remain")
        
        if not filtered_policies:
            print("No policies found matching your criteria")
            return []
        
        # NLP + Keyword-based coverage matching
        print("\nPerforming coverage analysis (NLP + Keyword matching)...")
        
        user_embedding = self.model.encode([user_coverage_requirement])
        
        recommendations = []
        for policy in filtered_policies:
            coverage_data = self.extract_coverage_details(policy)
            
            # NLP similarity
            policy_embedding = self.model.encode([coverage_data['text']])
            nlp_score = cosine_similarity(user_embedding, policy_embedding)[0][0]
            
            # Keyword matching
            keyword_match = self.keyword_matching(user_coverage_requirement, coverage_data)
            
            # Combined score (70% keyword, 30% NLP)
            combined_score = (keyword_match['match_percentage'] * 0.007) + (nlp_score * 0.3)
            
            # Generate coverage summary
            coverage_summary = self.generate_coverage_summary(
                user_coverage_requirement, coverage_data, keyword_match
            )
            
            premium_info = policy['premium_info']
            
            recommendations.append({
                'policy_id': str(policy.get('_id', 'N/A')),
                'policy_name': policy.get('policyName', 'Unknown'),
                'insurer': policy.get('insurer', 'Unknown'),
                'policy_type': policy.get('policyType', 'N/A'),
                'code': policy.get('code', 'N/A'),
                'premium': premium_info['premium_amount'],
                'sum_insured': premium_info['sum_insured'],
                'zone': premium_info['zone'],
                'composition': premium_info['composition'],
                'age_bracket': premium_info['age_bracket'],
                'nlp_score': round(nlp_score, 4),
                'keyword_score': keyword_match['match_percentage'],
                'combined_score': round(combined_score, 4),
                'matched_features': keyword_match['matched_features'],
                'coverage_summary': coverage_summary
            })
        
        # Sort by combined score (descending) and then by premium (ascending)
        recommendations.sort(key=lambda x: (-x['combined_score'], x['premium']))
        
        top_recommendations = recommendations[:top_n]
        
        self._display_recommendations(top_recommendations)
        
        return top_recommendations
    
    def _display_recommendations(self, recommendations):
        """Display recommendations"""
        print(f"\n{'='*70}")
        print(f"TOP {len(recommendations)} POLICY RECOMMENDATIONS")
        print(f"{'='*70}\n")
        
        for idx, policy in enumerate(recommendations, 1):
            print(f"Rank #{idx}")
            print(f"  Policy Name: {policy['policy_name']}")
            print(f"  Insurer: {policy['insurer']}")
            print(f"  Code: {policy['code']}")
            print(f"  Premium: ₹{policy['premium']:,} | Sum Insured: ₹{policy['sum_insured']:,}")
            print(f"  Zone: {policy['zone']} | Composition: {policy['composition']}")
            print(f"\n  Coverage Analysis:")
            print(f"  {policy['coverage_summary']}")
            print(f"{'-'*70}\n")
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        print("MongoDB connection closed")


# Example usage
if __name__ == "__main__":
    MONGO_URI = "mongodb+srv://sanashaikh8804_db_user:Cy6SebZuF3BgeukJ@cluster0.akao778.mongodb.net/"
    
    recommender = PolicyRecommender(
        mongo_uri=MONGO_URI,
        db_name="AskMyPolicy",
        collection_name="compare"
    )
    
    
    user_age = 28
    user_budget = 15000
    user_city = "Mumbai"
    policy_category = "Individual"
    user_coverage = "I need HIV/AIDS coverage, cancer treatment, mental illness coverage, and hospitalization benefits"
    
    recommendations = recommender.get_recommendations(
        user_age=user_age,
        user_budget=user_budget,
        user_city=user_city,
        policy_category=policy_category,
        user_coverage_requirement=user_coverage,
        top_n=5
    )
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION SUMMARY")
    print(f"{'='*70}")
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            print(f"\n{idx}. {rec['policy_name']} ({rec['code']})")
            print(f"   Insurer: {rec['insurer']}")
            print(f"   Premium: ₹{rec['premium']:,} | Sum Insured: ₹{rec['sum_insured']:,}")
    
    recommender.close()