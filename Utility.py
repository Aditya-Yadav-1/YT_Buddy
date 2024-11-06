import os
from dotenv import load_dotenv
from groq import Groq
import requests

load_dotenv()

class Utility:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.api_url = f"{os.getenv('HF_API_ROUTE')}/{os.getenv('HF_EMBEDDINGS_MODEL')}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

class LLMService(Utility):
    def send_query(self, query, contexts=None):
        chat_completion = self.client.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": f'''The information provided below is the transcript and description of a youtube video. Try to answer the question based on this information.
                    If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
                    So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

                    Context: {contexts}
                    Do provide only helpful answers

                    Helpful answer:
                    '''
                } 
                for _ in [contexts] if contexts
            ] + [
                {
                    "role": "user",
                    "content": f"{query}"
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    
class EmbeddingService(Utility):
    def get_embedding(self, texts):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        return response.json()
    
class WhisperService(Utility):
    def transcribe_audio(self, audio_stream):
        transcription = self.client.audio.transcriptions.create(
            file=("audio.m4a", audio_stream),
            model="whisper-large-v3",
            prompt="Specify context or spelling",
            response_format="json",
            language="en",
            temperature=0.0
        )
        return transcription