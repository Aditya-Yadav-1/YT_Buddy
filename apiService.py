from Utility import LLMService, EmbeddingService, WhisperService

class APIService:
    def __init__(self):
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        self.whisper_service = WhisperService()

    def get_llm_response(self, query, contexts=None):
        return self.llm_service.send_query(query, contexts)

    def get_embedding(self, texts):
        return self.embedding_service.get_embedding(texts)

    def transcribe_audio(self, audio_stream):
        return self.whisper_service.transcribe_audio(audio_stream)