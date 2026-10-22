import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SemanticCache:
    """
    Decision de Arquitectura: Para reducir costos de tokens en picos de demanda 
    (ej. días de partido), se implementa caché semántico.
    Si la similitud coseno > 0.95, se sirve desde caché sin llamar al LLM.
    """
    def __init__(self, threshold=0.95):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.cache_entries = [] # En producción: Redis + Vector DB

    def get_embedding(self, text):
        return self.model.encode([text])[0]

    def query(self, user_message: str):
        query_emb = self.get_embedding(user_message)
        if not self.cache_entries:
            return None
        
        cached_embs = np.array([e['embedding'] for e in self.cache_entries])
        similarities = cosine_similarity([query_emb], cached_embs)[0]
        max_idx = np.argmax(similarities)
        
        if similarities[max_idx] >= self.threshold:
            print(f"[CACHE HIT] Similitud: {similarities[max_idx]:.2f}")
            return self.cache_entries[max_idx]['response']
        
        print("[CACHE MISS] Llamando al LLM...")
        return None

    def store(self, question: str, answer: str):
        self.cache_entries.append({
            'question': question,
            'response': answer,
            'embedding': self.get_embedding(question)
        })
