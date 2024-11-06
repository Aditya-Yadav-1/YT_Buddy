import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from apiService import APIService
import pickle

class VectorDB:
    def __init__(self):
        pass

    def createFaissVectorStore(self):
        with open("data/transcription.txt", "r", encoding='utf-8') as file:
            transcript = file.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size of 1000 characters
            chunk_overlap=300  # overlap size of 300 characters
        )

        chunks = splitter.split_text(transcript)
        # for chunk in chunks:
        #     print(f"{chunk}\n")

        api_service = APIService()
        embeddings = api_service.get_embedding(chunks)
        print(embeddings)

        embeddings_np = np.array(embeddings).astype('float32')
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)

        faiss.write_index(index, "faissVectorStore/VideoChat/faiss_index.index")
        with open("faissVectorStore/VideoChat/index.pkl", "wb") as f:
            pickle.dump(chunks, f)
        return "Vector Store Creation Completed"