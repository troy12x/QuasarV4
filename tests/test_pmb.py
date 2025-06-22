# c:\quasarv4\tests\test_pmb.py

import torch
import unittest
import uuid
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.pmb import ParameterMemoryBank

class TestPMB(unittest.TestCase):

    def setUp(self):
        """Set up a PMB instance before each test."""
        self.pmb = ParameterMemoryBank(num_blocks=16, slots_per_block=32)
        self.embedding_dim = 64

    def test_store_and_direct_retrieve(self):
        """Test storing an item and retrieving it directly by ID."""
        item_id = "test-id-123"
        key_emb = torch.randn(self.embedding_dim)
        value = "This is a test value."
        
        self.pmb.store(item_id, key_emb, value)
        
        retrieved_value = self.pmb.retrieve_direct(item_id)
        
        self.assertEqual(retrieved_value, value)
        self.assertEqual(len(self.pmb), 1)
        print("\nPMB store and direct retrieve test successful.")

    def test_semantic_retrieve(self):
        """Test retrieving items based on semantic similarity."""
        # Store a few items
        id1, val1 = uuid.uuid4(), "An article about dogs."
        key1 = torch.tensor([0.9, 0.1, 0.1]) # Strong 'dog' component
        
        id2, val2 = uuid.uuid4(), "A document on cats."
        key2 = torch.tensor([0.1, 0.9, 0.1]) # Strong 'cat' component
        
        id3, val3 = uuid.uuid4(), "Information on puppies."
        key3 = torch.tensor([0.8, 0.2, 0.1]) # Also strong 'dog' component
        
        self.pmb.store(id1, key1, val1)
        self.pmb.store(id2, key2, val2)
        self.pmb.store(id3, key3, val3)

        # Query for 'dog'
        query_vector = torch.tensor([1.0, 0.0, 0.0])
        results = self.pmb.retrieve_semantic(query_vector, top_k=2)
        
        self.assertEqual(len(results), 2)
        # The first result should be the one about dogs
        self.assertEqual(results[0][1], val1)
        # The second result should be the one about puppies
        self.assertEqual(results[1][1], val3)
        print("PMB semantic retrieve test successful.")

    def test_non_existent_item(self):
        """Test retrieving a non-existent item."""
        retrieved_value = self.pmb.retrieve_direct("non-existent-id")
        self.assertIsNone(retrieved_value)
        print("PMB non-existent item test successful.")

if __name__ == '__main__':
    unittest.main()
