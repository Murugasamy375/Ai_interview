import re
import math
from collections import Counter
from typing import List, Dict, Any
import numpy as np

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r'\b\w+\b', text.lower())

class BM25Scorer:
    """Standard BM25 keyword matching engine."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_lens = [len(doc) for doc in corpus]
        self.doc_term_freqs = [Counter(doc) for doc in corpus]
        
        # Calculate IDF (Inverse Document Frequency)
        self.idf = {}
        df = Counter()
        for doc in corpus:
            for term in set(doc):
                df[term] += 1
        for term, freq in df.items():
            # Standard BM25 IDF formulation
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: List[str]) -> List[float]:
        """Compute BM25 scores for all documents in the corpus against the query."""
        scores = [0.0] * self.corpus_size
        for i in range(self.corpus_size):
            doc_len = self.doc_lens[i]
            tf = self.doc_term_freqs[i]
            score = 0.0
            for term in query:
                if term not in self.idf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += self.idf[term] * (f * (self.k1 + 1.0)) / denom
            scores[i] = score
        return scores

    def get_self_score(self, query: List[str]) -> float:
        """Calculate the max possible score if the query matched itself as a document."""
        if not query:
            return 0.0
        q_len = len(query)
        tf = Counter(query)
        score = 0.0
        for term in query:
            idf_val = self.idf.get(term, math.log(self.corpus_size + 1.0))
            f = tf[term]
            denom = f + self.k1 * (1.0 - self.b + self.b * (q_len / max(1.0, self.avg_doc_len)))
            score += idf_val * (f * (self.k1 + 1.0)) / denom
        return score

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def compute_ats_score(resume_chunks: List[Dict[str, Any]], jd_chunks: List[Dict[str, Any]], alpha: float = 0.5) -> Dict[str, Any]:
    """
    Computes a hybrid ATS matching score between a Resume and Job Description.
    
    Combines:
    - Cosine Semantic Similarity of chunks
    - BM25 Keyword Search match scores
    
    Returns details on matches for each chunk of the job description.
    """
    if not resume_chunks or not jd_chunks:
        return {
            "semantic_score": 0.0,
            "bm25_score": 0.0,
            "ats_score": 0.0,
            "chunk_matches": []
        }

    # Initialize BM25 on the Resume chunks
    resume_tokenized = [tokenize(chunk["text"]) for chunk in resume_chunks]
    bm25 = BM25Scorer(resume_tokenized)

    chunk_matches = []
    semantic_similarities = []
    bm25_normalized_scores = []
    hybrid_scores = []

    for idx_jd, jd_chunk in enumerate(jd_chunks):
        jd_text = jd_chunk["text"]
        jd_embedding = jd_chunk["embedding"]
        jd_tokens = tokenize(jd_text)

        # 1. Cosine Semantic Similarity
        best_semantic_sim = -1.0
        best_resume_idx = 0
        for idx_res, res_chunk in enumerate(resume_chunks):
            sim = cosine_similarity(jd_embedding, res_chunk["embedding"])
            if sim > best_semantic_sim:
                best_semantic_sim = sim
                best_resume_idx = idx_res
        
        normalized_semantic = max(0.0, best_semantic_sim)

        # 2. BM25 score
        bm25_scores = bm25.get_scores(jd_tokens)
        max_bm25_raw = max(bm25_scores) if bm25_scores else 0.0
        
        self_score = bm25.get_self_score(jd_tokens)
        if self_score > 0:
            normalized_bm25 = min(1.0, max_bm25_raw / self_score)
        else:
            normalized_bm25 = 0.0

        # 3. Hybrid score calculation
        hybrid = alpha * normalized_semantic + (1.0 - alpha) * normalized_bm25

        semantic_similarities.append(normalized_semantic)
        bm25_normalized_scores.append(normalized_bm25)
        hybrid_scores.append(hybrid)

        chunk_matches.append({
            "jd_chunk_index": idx_jd,
            "jd_chunk_text": jd_text,
            "best_resume_chunk_text": resume_chunks[best_resume_idx]["text"],
            "semantic_similarity": round(normalized_semantic, 4),
            "bm25_score": round(normalized_bm25, 4),
            "hybrid_score": round(hybrid, 4)
        })

    avg_semantic = sum(semantic_similarities) / len(semantic_similarities)
    avg_bm25 = sum(bm25_normalized_scores) / len(bm25_normalized_scores)
    avg_hybrid = sum(hybrid_scores) / len(hybrid_scores)

    return {
        "semantic_score": round(avg_semantic, 4),
        "bm25_score": round(avg_bm25, 4),
        "ats_score": round(avg_hybrid * 100.0, 2),
        "chunk_matches": chunk_matches
    }
