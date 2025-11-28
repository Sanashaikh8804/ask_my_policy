import httpx
import asyncio

class GroqChatCompletion:
    def __init__(self, api_key: str, model_id: str = "llama-3-70b-8192"):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def get_chat_message_content(self, chat_history, settings=None, kernel=None):
        messages = []
        for msg in chat_history.messages:
            if msg.role == "system":
                messages.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": getattr(settings, "temperature", 0.3) if settings else 0.3,
            "max_tokens": getattr(settings, "max_tokens", 2000) if settings else 2000,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
