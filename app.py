import streamlit as st
from Vector_DB import VectorDB
from YtBuddy import YtBuddyCreator
from streamlit_chat import message
from Whisper_Transcribe import VideoToText

st.set_page_config(page_title="YT Buddy", layout="wide")
st.title("YT Buddy - Chat with your :red[YouTube] videos")

def display_conversation(history):
    for i in range(len(history["assistant"])):
        message(history["user"][i], is_user=True, key=str(i) + "_user", avatar_style="avataaars")
        message(history["assistant"][i], key=str(i), avatar_style="bottts")

yt_url = st.text_input("🎬 Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

if "assistant" not in st.session_state:
    st.session_state["assistant"] = ["I am ready to help you"]
if "user" not in st.session_state:
    st.session_state["user"] = ["Hey there!"]
if "transcript_generated" not in st.session_state:
    st.session_state["transcript_generated"] = False
if "vector_store_loaded" not in st.session_state:
    st.session_state["vector_store_loaded"] = False

if yt_url and not st.session_state["transcript_generated"]:
    with st.spinner("🎙️ Generating transcript..."):
        transcriber = VideoToText()
        transcriber.gen_transcript(yt_url)
        st.session_state["transcript_generated"] = True

if st.session_state["transcript_generated"] and not st.session_state["vector_store_loaded"]:
    with st.spinner("🔍 Loading vector store..."):
        faiss_vector_db = VectorDB()
        vector_store = faiss_vector_db.createFaissVectorStore()
        st.session_state["vector_store_loaded"] = True


if st.session_state["vector_store_loaded"]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎥 Video")
        st.video(yt_url)

    with col2:
        st.subheader("💬 Chat with Video")

        with st.expander("📜 View Transcript", expanded=False):
            with open("data/transcription.txt", "r", encoding='utf-8') as f:
                transcript = f.read()
            st.success(transcript)

        user_query = st.text_input("🔎 Ask a question about the video", placeholder="What is the video about?")
        
        @st.cache_resource(show_spinner=True)
        def create_yt_buddy():
            yt_buddy_creator = YtBuddyCreator()
            return yt_buddy_creator
        
        yt_buddy = create_yt_buddy()

        if st.button("Get Answer"):
            answer = yt_buddy.generate_response(user_query)
            st.session_state["user"].append(user_query)
            st.session_state["assistant"].append(answer)

            if st.session_state["assistant"]:
                display_conversation(st.session_state)
