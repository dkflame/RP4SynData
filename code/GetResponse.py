import openai
import requests
import json

# Replace the following line with your API key 
openai.api_key = ""


def get_response(user_message):
    model = 'gpt-3.5-turbo'
    conversation_id = None
    response = openai.ChatCompletion.create(
                    model=model,
                    stream=True,
                    chatId=conversation_id,
                    messages=user_message
                )
    ai_message = ''
    for message in response:
        ai_message += message
    return ai_message