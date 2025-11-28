# GroqChatCompletion.py

import httpx
import json

class GroqChatCompletion:
    def __init__(self, api_key: str, model_id: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def get_chat_message_content(self, chat_history, settings=None, kernel=None):
        # This implementation is adapted to work without semantic-kernel's ChatHistory object
        messages = chat_history.get("messages", [])
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": getattr(settings, "temperature", 0.3) if settings else 0.3,
            "max_tokens": getattr(settings, "max_tokens", 800) if settings else 800,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, json=payload, headers=headers, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                # Handle potential variations in the response structure
                if data.get("choices") and data["choices"][0].get("message"):
                    return data["choices"][0]["message"].get("content", "No content found in response.")
                else:
                    return f"Unexpected API response format: {json.dumps(data)}"
            except httpx.HTTPStatusError as e:
                return f"API Error: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                return f"An unexpected error occurred: {e}"