from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

class LaBSE:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/LaBSE')

    def encode_pairs(self, sentence_pairs):
        flat_sentences = [sentence for pair in sentence_pairs for sentence in pair]
        original_shape = np.array(sentence_pairs).shape
        embeddings = torch.tensor(self.model.encode(flat_sentences)).reshape(original_shape[0],original_shape[1],-1)
        return embeddings, embeddings.mean(dim=1)

    def compute_similarities(self, sentence_pairs):
        individual_embeddings, average_embeddings = self.encode_pairs(sentence_pairs)
        similarities = []
        for emb1, emb2 in individual_embeddings:
            sim = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append(sim)
        return similarities
