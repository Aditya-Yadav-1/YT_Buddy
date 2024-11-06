from Utility import LLMService, EmbeddingService, WhisperService

class APIService:
    def __init__(self):
        pass

    def get_llm_response(self, query, contexts=None):
        self.llm_service = LLMService()
        return self.llm_service.send_query(query, contexts)

    def get_embedding(self, texts):
        self.embedding_service = EmbeddingService()
        return self.embedding_service.get_embedding(texts)

    def transcribe_audio(self, audio_stream):
        self.whisper_service = WhisperService()
        return self.whisper_service.transcribe_audio(audio_stream)