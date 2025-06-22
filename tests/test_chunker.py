# c:\quasarv4\tests\test_chunker.py

import unittest
import torch
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.chunker import SemanticChunker

class TestChunker(unittest.TestCase):

    def setUp(self):
        """Set up a Chunker instance before each test."""
        self.chunker = SemanticChunker(max_chunk_size=20)

    def test_paragraph_splitting(self):
        """Test that the chunker correctly splits text by paragraphs."""
        doc = "This is the first paragraph.\n\nThis is the second one."
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "This is the first paragraph.")
        self.assertEqual(chunks[1], "This is the second one.")
        print("\nChunker paragraph splitting test successful.")

    def test_long_paragraph_segmentation(self):
        """Test that a long paragraph is broken down into smaller chunks."""
        long_para = "This is a very long paragraph that needs to be segmented. It talks about many things. The chunker should be smart enough to split it into multiple parts based on sentence boundaries, without breaking sentences midway. This ensures semantic coherence is maintained within each new chunk."
        chunks = self.chunker.chunk_document(long_para)
        self.assertTrue(len(chunks) > 1)
        # Check that the first chunk is a full sentence or more
        self.assertIn("segmented.", chunks[0])
        print("Chunker long paragraph segmentation test successful.")

    def test_embedding_generation(self):
        """Test the generation of combined sparse and dense embeddings."""
        corpus = ["the cat sat on the mat", "the dog ate my homework"]
        chunks = ["the cat sat on the mat", "the dog ate my homework"]
        
        # Fit the TF-IDF model
        self.chunker.fit_tfidf(corpus)
        
        # Get embeddings (with a mock dense model)
        embeddings = self.chunker.get_embeddings(chunks)
        
        self.assertEqual(len(embeddings), 2)
        # Check that embeddings are torch tensors
        self.assertIsInstance(embeddings[0], torch.Tensor)
        # Check the dimensionality (768 for placeholder dense + vocab size for sparse)
        expected_dim = 768 + len(self.chunker.tfidf_vectorizer.vocabulary_)
        self.assertEqual(embeddings[0].shape[0], expected_dim)
        print("Chunker embedding generation test successful.")

if __name__ == '__main__':
    unittest.main()
