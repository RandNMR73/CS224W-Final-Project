from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device
                                       ] = None):
        """
        Initializes the GloveTextEmbedding model with a specified device.

        Args:
            device (Optional[torch.device]): The device to run the model on (CPU or GPU).
        """
        # Load the pre-trained GloVe model for sentence embeddings
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        """
        Encodes a list of sentences into their corresponding embeddings.

        Args:
            sentences (List[str]): A list of sentences to encode.

        Returns:
            Tensor: A tensor containing the embeddings for the input sentences.
        """
        # Encode the sentences and convert the result to a PyTorch tensor
        return torch.from_numpy(self.model.encode(sentences))