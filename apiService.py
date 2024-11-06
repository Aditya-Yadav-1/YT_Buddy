from Utility import LLMService, EmbeddingService, WhisperService

class APIService:
    def __init__(self):
        pass

    def get_llm_response(self, query, contexts=None):
        llm_service = LLMService()
        return llm_service.send_query(query, contexts)

    def get_embedding(self, texts):
        embedding_service = EmbeddingService()
        return embedding_service.get_embedding(texts)

    def transcribe_audio(self, audio_stream):
        whisper_service = WhisperService()
        return whisper_service.transcribe_audio(audio_stream)