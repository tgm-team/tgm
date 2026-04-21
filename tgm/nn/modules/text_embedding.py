from typing import Optional, List
import torch
try:
    from sentence_transformers import SentenceTransformer

    class GloveTextEmbedding:
        r"""
        Simple Glove text embedder for text feature from RelBench datasets
        """
        def __init__(self, device: Optional[torch.device] = 'cpu'):
            self.model = SentenceTransformer(
                'sentence-transformers/average_word_embeddings_glove.6B.300d',
                device=device,
            )

        def __call__(self, sentences: List[str]) -> torch.Tensor:
            return torch.from_numpy(self.model.encode(sentences))

except ImportError:
    GloveTextEmbedding = None