# c:\quasarv4\tests\test_model.py

import torch
import unittest
import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.model import Quasar
from quasar.utils import SimpleTokenizer

class TestQuasarModel(unittest.TestCase):

    def setUp(self):
        """Set up a Quasar model instance before each test."""
        self.corpus = ["hello world", "quasar is a new model"]
        self.tokenizer = SimpleTokenizer(self.corpus)
        self.vocab_size = len(self.tokenizer)
        self.embedding_dim = 32
        self.hidden_dim = 64
        
        self.model = Quasar(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )

    def test_model_creation(self):
        """Test if the Quasar model is created without errors."""
        self.assertIsNotNone(self.model)
        print("\nQuasar model creation test successful.")

    def test_ingest_process(self):
        """Test the document ingestion process."""
        doc = "This is a test document. It has several sentences. The model should ingest it correctly."
        self.model.ingest(doc)
        self.assertTrue(len(self.model.pmb) > 0)
        print("Quasar ingest process test successful.")

    def test_forward_pass_no_memory(self):
        """Test the forward pass without memory retrieval."""
        prompt = "hello"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        
        logits = self.model(input_ids, memory_query=False)
        
        self.assertEqual(logits.shape, (1, self.vocab_size))
        print("Quasar forward pass (no memory) test successful.")

    def test_forward_pass_with_memory(self):
        """Test the forward pass with memory retrieval."""
        # First, ingest a document to populate memory
        doc = "The future of AI is here. It is called Quasar."
        self.model.ingest(doc)
        
        # Now, run a prompt
        prompt = "what is the future"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        
        logits = self.model(input_ids, memory_query=True)
        
        self.assertEqual(logits.shape, (1, self.vocab_size))
        print("Quasar forward pass (with memory) test successful.")

if __name__ == '__main__':
    unittest.main()
