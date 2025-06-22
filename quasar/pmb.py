# c:\quasarv4\quasar\pmb.py

import torch
import hashlib
import numpy as np

class ParameterMemoryBank:
    """
    Parameter Memory Bank (PMB) for infinite, queryable memory.
    
    This implementation uses a two-level hashing system for constant-time
    direct access and supports semantic similarity search.
    
    - Level 1: A list of 'blocks'.
    - Level 2: Each block is a dictionary-like structure mapping slots to items.
    
    For simplicity, we use Python lists and dictionaries. A production system
    would use a more optimized backend (e.g., Redis, custom memory store).
    """
    def __init__(self, num_blocks=1024, slots_per_block=4096):
        self.num_blocks = num_blocks
        self.slots_per_block = slots_per_block
        
        # PMB is a list of blocks, where each block is a list of slots.
        # Each slot can hold a tuple: (id, key_embedding, value)
        self.pmb = [ [None] * slots_per_block for _ in range(num_blocks) ]
        
        # For semantic search, we need a separate structure to hold all keys.
        # This is a trade-off for efficient similarity search.
        self.all_keys = []
        self.key_locations = [] # Stores (block_idx, slot_idx) for each key

    def _hash_fn(self, s, salt=""):
        """A simple, salted hash function."""
        return int(hashlib.sha256((str(s) + salt).encode()).hexdigest(), 16)

    def _get_hash_indices(self, item_id):
        """
        Calculates the block and slot indices for a given item ID using
        the two-level hashing scheme.
        """
        block_hash = self._hash_fn(item_id, salt="block")
        block_idx = block_hash % self.num_blocks
        
        slot_hash = self._hash_fn(item_id, salt=f"slot_{block_idx}")
        slot_idx = slot_hash % self.slots_per_block
        
        return block_idx, slot_idx

    def store(self, item_id, key_embedding, value):
        """
        Stores a key-value pair in the PMB using its ID.
        
        Args:
            item_id (str or int): A unique identifier for the data.
            key_embedding (torch.Tensor): The embedding vector (k_i,j).
            value (any): The data to store (v_i,j), e.g., text, metadata.
        """
        if not isinstance(key_embedding, torch.Tensor):
            raise TypeError("key_embedding must be a torch.Tensor")

        block_idx, slot_idx = self._get_hash_indices(item_id)
        
        # Store the item in the hash-based location.
        # Note: This simple implementation doesn't handle hash collisions.
        # A real system would need a collision resolution strategy (e.g., cuckoo hashing, chaining).
        if self.pmb[block_idx][slot_idx] is not None:
            print(f"Warning: Hash collision at ({block_idx}, {slot_idx}) for id {item_id}. Overwriting.")

        self.pmb[block_idx][slot_idx] = (item_id, key_embedding, value)
        
        # Also store the key for semantic search
        self.all_keys.append(key_embedding.detach().cpu().unsqueeze(0))
        self.key_locations.append((block_idx, slot_idx))

    def retrieve_direct(self, item_id):
        """
        Retrieves a value directly using its ID in O(1) time.
        
        Args:
            item_id (str or int): The unique identifier of the item.
            
        Returns:
            The stored value, or None if not found.
        """
        block_idx, slot_idx = self._get_hash_indices(item_id)
        item = self.pmb[block_idx][slot_idx]
        
        # Check if the found item ID matches, in case of no collision handling
        if item and item[0] == item_id:
            return item[2] # Return the value
        return None

    def retrieve_semantic(self, query_embedding, top_k=5):
        """
        Retrieves the top_k most semantically similar items to a query embedding.
        
        Args:
            query_embedding (torch.Tensor): The query vector.
            top_k (int): The number of similar items to return.
            
        Returns:
            A list of tuples, each containing (similarity_score, value).
        """
        if not self.all_keys:
            return []
            
        if not isinstance(query_embedding, torch.Tensor):
            raise TypeError("query_embedding must be a torch.Tensor")

        # Concatenate all keys into a single tensor for efficient computation
        all_keys_tensor = torch.cat(self.all_keys, dim=0).to(query_embedding.device)
        
        # Compute cosine similarity
        query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        keys_norm = torch.nn.functional.normalize(all_keys_tensor, p=2, dim=1)
        similarities = torch.mm(query_norm, keys_norm.T).squeeze(0)
        
        # Get top_k results
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(top_k, len(self.all_keys)))
        
        results = []
        for i in range(len(top_k_indices)):
            score = top_k_scores[i].item()
            original_idx = top_k_indices[i].item()
            
            block_idx, slot_idx = self.key_locations[original_idx]
            item = self.pmb[block_idx][slot_idx]
            
            if item:
                results.append((score, item[2])) # (similarity_score, value)
        
        return results

    def __len__(self):
        return len(self.all_keys)
