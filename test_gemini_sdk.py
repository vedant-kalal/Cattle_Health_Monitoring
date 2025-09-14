import google.generativeai as genai

# Configure with your Gemini API key
genai.configure(api_key="AIzaSyCf9PIY45-P6FQZD1vgG0dsVml0-bGtYes")

try:
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    # System prompt for chatbot specialization
    system_prompt = (
        "You are an intelligent chatbot assistant for a cattle monitoring system. "
        "You help users with questions about cattle milk yield prediction, disease detection, animal health, and farm management. "
        "Always provide clear, helpful, and relevant answers based on the user's input. "
        "If the question is not related to cattle or farming, politely redirect the user."
    )
    # Example user question
    user_question = "What are common symptoms of mastitis in cows?"
    # Gemini expects only 'user' and 'model' roles. Provide system prompt as first user message.
    response = model.generate_content([
        system_prompt,
        user_question
    ])
    print(response.text)
except Exception as e:
    print("Error:", e)