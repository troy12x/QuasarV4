# c:\quasarv4\quasar\chunker.py

import torch
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the sentence tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' model for NLTK sentence tokenization...")
    nltk.download('punkt')

class SemanticChunker:
    """
    A multi-level semantic chunking system that segments documents into
    meaningful chunks like sentences or paragraphs.
    
    This implementation includes:
    - Splitting by sections and paragraphs.
    - A basic semantic segmentation fallback for long paragraphs.
    - Generation of sparse embeddings (TF-IDF) for each chunk.
    """
    def __init__(self, max_chunk_size=256, dense_embedding_model=None):
        self.max_chunk_size = max_chunk_size
        self.tfidf_vectorizer = TfidfVectorizer()
        self.dense_embedding_model = dense_embedding_model

    def _split_into_paragraphs(self, text):
        """Splits text into paragraphs based on double newlines."""
        return [p.strip() for p in text.split('\n\n') if p.strip()]

    def _semantic_segmentation(self, paragraph):
        """
        A basic semantic segmentation algorithm.
        This is a placeholder for a more sophisticated model. It splits a long
        paragraph into sentences and groups them to respect max_chunk_size.
        """
        sentences = nltk.sent_tokenize(paragraph)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len((current_chunk + " " + sentence).split()) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def chunk_document(self, document_text):
        """
        Processes a full document text into a list of semantic chunks.
        
        Args:
            document_text (str): The raw text of the document.
            
        Returns:
            list: A list of strings, where each string is a semantic chunk.
        """
        # For this implementation, we treat the whole document as one section.
        # A more advanced version would use headers or other markers.
        paragraphs = self._split_into_paragraphs(document_text)
        
        all_chunks = []
        for paragraph in paragraphs:
            # A simple length check based on word count
            if len(paragraph.split()) > self.max_chunk_size:
                chunks = self._semantic_segmentation(paragraph)
            else:
                chunks = [paragraph]
            all_chunks.extend(chunks)
            
        return all_chunks

    def fit_tfidf(self, corpus):
        """Fits the TF-IDF vectorizer on a corpus of documents."""
        self.tfidf_vectorizer.fit(corpus)

    def get_embeddings(self, chunks):
        """
        Generates the combined dense and sparse embeddings for a list of chunks.
        
        Args:
            chunks (list of str): The text chunks.
            
        Returns:
            list: A list of combined embeddings (as numpy arrays).
        """
        if not chunks:
            return []

        # Generate sparse embeddings (TF-IDF)
        sparse_embeddings = self.tfidf_vectorizer.transform(chunks).toarray()
        
        # Generate dense embeddings (using the provided encoder)
        if self.dense_embedding_model:
            # This assumes the model has an `encode` method like sentence-transformers
            dense_embeddings = self.dense_embedding_model.encode(chunks, convert_to_tensor=True)
            dense_embeddings = dense_embeddings.cpu().numpy()
        else:
            # If no dense model, use zeros as placeholders
            # A real application MUST provide a dense model.
            print("Warning: No dense embedding model provided. Using zero vectors.")
            # Assuming a common embedding size like 768 for placeholder shape
            dense_embeddings = np.zeros((len(chunks), 768))

        # For simplicity, we concatenate sparse and dense embeddings.
        # The lambdas from the formula can be seen as weighting during training.
        # Ensure dense and sparse embeddings can be concatenated.
        # This might require padding or truncation in a real scenario if dimensions vary.
        if dense_embeddings.shape[0] != sparse_embeddings.shape[0]:
            raise ValueError("Mismatch in number of dense and sparse embeddings.")

        # We will just concatenate them. The spec's `lambda` weights would be
        # applied in the model that consumes these embeddings.
        combined_embeddings = np.concatenate((dense_embeddings, sparse_embeddings), axis=1)
        
        return [torch.from_numpy(emb).float() for emb in combined_embeddings]
