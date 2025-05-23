from laserembeddings import Laser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

class LASER:
    def __init__(self):
        self.model = Laser()

    def encode_pairs(self, sentence_pairs, lang=''):
        flat_sentences = [sentence for pair in sentence_pairs for sentence in pair]

        embeddings_np = self.model.embed_sentences(flat_sentences, lang=lang)
        original_shape = np.array(sentence_pairs).shape
        embeddings = torch.tensor(embeddings_np).reshape(original_shape[0],original_shape[1],-1)

        return embeddings, embeddings.mean(dim=1)

    def compute_similarities(self, sentence_pairs, lang=''):

        individual_embeddings, _ = self.encode_pairs(sentence_pairs, lang=lang)
        similarities = []
        for emb1, emb2 in individual_embeddings:
            sim = cosine_similarity([emb1.numpy()], [emb2.numpy()])[0][0]
            similarities.append(sim)
        return similarities
