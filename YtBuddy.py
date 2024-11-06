import pickle
from apiService import APIService
import numpy as np
import faiss

class YtBuddyCreator: 
    def __init__(self):
        pass

    def retrieval_qa_chain(self, question):
        api_service = APIService()
        question_embedding = api_service.get_embedding([question])[0]

        question_embedding_np = np.array(question_embedding).astype('float32').reshape(1, -1)

        index = faiss.read_index("faissVectorStore/VideoChat/faiss_index.index")
        with open("faissVectorStore/VideoChat/index.pkl", "rb") as f:
            chunks = pickle.load(f)

        k = 2
        distances, indices = index.search(question_embedding_np, k)

        all_contexts = []
        for query_idx, query_indices in enumerate(indices):
            contexts = [chunks[idx] for idx in query_indices]
            all_contexts.append(contexts)
            # print(f"Query {query_idx + 1} results:")
            # for i, context in enumerate(contexts, start=1):
            #     print(f"Context {i}: {context}")
        return contexts

    def generate_response(self, question):
        api_service = APIService()
        contexts = self.retrieval_qa_chain(question)
        answer = api_service.get_llm_response(question, contexts)
        return answer