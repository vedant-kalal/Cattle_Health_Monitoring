
import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

class GeminiAI:
    def __init__(self, api_key: str = None, model: str = None):
        # Load from .env if not provided
        load_dotenv(find_dotenv())
        # Prefer neutral names for safe uploads; fallback to original names for compatibility
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("API_MODEL") or os.getenv("GEMINI_MODEL")
        if not self.api_key or not self.model:
            raise ValueError("API_KEY/API_MODEL (or GEMINI_API_KEY/GEMINI_MODEL) must be set.")
        genai.configure(api_key=self.api_key)
        self.model_obj = genai.GenerativeModel(model_name=self.model)

    def chat(self, prompt: str, history=None):
        """
        Chat with Gemini using the Python SDK, with a system prompt for cattle monitoring specialization.
        Optionally include chat history for multi-turn context.
        """
        system_prompt = (
            "You are an intelligent chatbot assistant for a cattle monitoring system. "
            "You help users with questions about cattle milk yield prediction, disease detection, animal health, and farm management. "
            "Always provide clear, helpful, and relevant answers based on the user's input. "
            "If the question is not related to cattle or farming, politely redirect the user."
        )
        # Compose conversation: system prompt, then history, then user prompt
        messages = [system_prompt]
        if history:
            messages.extend(history)
        messages.append(prompt)
        try:
            response = self.model_obj.generate_content(messages)
            return {"text": response.text}
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    print("--- Running Gemini SDK Individual Test ---")
    gemini_test = GeminiAI()
    print(f"Using API Key ending with: ...{gemini_test.api_key[-4:]}")
    print(f"Using Model: {gemini_test.model}")
    test_prompt = "Hello Gemini, this is a direct test. Are you receiving me?"
    response = gemini_test.chat(test_prompt)
    print("\n--- SDK Response ---")
    import json
    print(json.dumps(response, indent=2))
    print("--- End of Test ---")


