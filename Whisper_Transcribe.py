from pytubefix import YouTube
from dotenv import load_dotenv
from groq import Groq
from apiService import APIService
from io import BytesIO

load_dotenv()

class VideoToText:
    def __init__(self):
        pass

    def VideoDescription(self, url):
        video = YouTube(url)
        description = video.description
        return description

    def VideoToAudio(self, url):
        video = YouTube(url)
        audio = video.streams.filter(abr='160kbps').first()
        audio_stream = BytesIO()
        audio.stream_to_buffer(audio_stream)
        audio_stream.seek(0)
        return audio_stream

    def gen_transcript(self, video_url):
        audio_stream = self.VideoToAudio(video_url)
        videoDescription = self.VideoDescription(video_url)
        
        api_service = APIService()
        transcription = api_service.transcribe_audio(audio_stream)

        with open("data/transcription.txt", "w+", encoding='utf-8') as f:
            f.write(transcription.text + '\n' + '\n' + videoDescription)