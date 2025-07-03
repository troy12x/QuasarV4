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
    def __init__(self, num_blocks=1024, slots_per_block=4096, embedding_dim=None):
        self.num_blocks = num_blocks
        self.slots_per_block = slots_per_block
        self.embedding_dim = embedding_dim
        
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
            # Handle collision by updating the existing entry or finding an empty slot
            pass  # For now, just overwrite

        self.pmb[block_idx][slot_idx] = (item_id, key_embedding.detach().cpu(), value.detach().cpu() if isinstance(value, torch.Tensor) else value)
        
        # Also store the key for semantic search
        self.all_keys.append(key_embedding.detach().cpu())
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

    def retrieve_by_indices(self, indices):
        """
        Retrieves items by their indices in the `all_keys` list.
        Args:
            indices (list or torch.Tensor): A list of indices.
        Returns:
            A list of the retrieved values.
        """
        results = []
        for idx in indices:
            if idx < len(self.key_locations):
                block_idx, slot_idx = self.key_locations[idx]
                item = self.pmb[block_idx][slot_idx]
                if item:
                    value = item[2]  # Get the value
                    # Convert back to tensor if it was stored as tensor
                    if isinstance(value, torch.Tensor):
                        results.append(value)
                    else:
                        # If value is not a tensor, create a zero tensor of appropriate size
                        if self.embedding_dim:
                            results.append(torch.zeros(self.embedding_dim))
                        else:
                            # Fallback: use the key embedding as value
                            results.append(item[1])  # Use key embedding
                else:
                    # No item found, append zero tensor
                    if self.embedding_dim:
                        results.append(torch.zeros(self.embedding_dim))
                    else:
                        results.append(torch.zeros_like(self.all_keys[0]) if self.all_keys else torch.zeros(1))
            else:
                # Index out of range
                if self.embedding_dim:
                    results.append(torch.zeros(self.embedding_dim))
                else:
                    results.append(torch.zeros_like(self.all_keys[0]) if self.all_keys else torch.zeros(1))
        return results

    def retrieve_semantic(self, query_embeddings, top_k=1):
        """
        Retrieves the top_k most semantically similar items for a batch of query embeddings.

        Args:
            query_embeddings (torch.Tensor): Query vectors (batch_size, embedding_dim) or (batch_size, seq_len, embedding_dim).
            top_k (int): The number of similar items to return for each query.

        Returns:
            A tensor of the aggregated retrieved values with the same shape as query_embeddings.
        """
        if not self.all_keys or top_k == 0:
            return torch.zeros_like(query_embeddings)

        if not isinstance(query_embeddings, torch.Tensor):
            raise TypeError("query_embeddings must be a torch.Tensor")

        # Store original shape and device
        original_shape = query_embeddings.shape
        device = query_embeddings.device
        
        # Flatten query embeddings to 2D for processing
        if query_embeddings.dim() > 2:
            query_flat = query_embeddings.view(-1, original_shape[-1])
        else:
            query_flat = query_embeddings

        # Handle empty memory bank
        if not self.all_keys:
            return torch.zeros_like(query_embeddings)

        try:
            # Stack all keys into a single tensor
            all_keys_tensor = torch.stack(self.all_keys, dim=0).to(device)
            
            # Compute cosine similarity
            query_norm = torch.nn.functional.normalize(query_flat, p=2, dim=-1)
            keys_norm = torch.nn.functional.normalize(all_keys_tensor, p=2, dim=-1)
            
            # Compute similarities: (batch_size, num_keys)
            similarities = torch.mm(query_norm, keys_norm.T)
            
            # Get top_k results for each query
            k = min(top_k, len(self.all_keys))
            if k > 0:
                top_k_scores, top_k_indices = torch.topk(similarities, k=k, dim=1)
                
                # Retrieve the corresponding values
                batch_results = []
                for i in range(query_flat.size(0)):
                    retrieved_values = self.retrieve_by_indices(top_k_indices[i].cpu().tolist())
                    
                    if retrieved_values:
                        # Stack and move to correct device
                        stacked_values = torch.stack(retrieved_values, dim=0).to(device)
                        # Average the top_k retrieved values
                        aggregated_value = torch.mean(stacked_values, dim=0)
                        batch_results.append(aggregated_value)
                    else:
                        # No valid retrievals, use zero tensor
                        batch_results.append(torch.zeros(original_shape[-1], device=device))
                
                # Stack all batch results
                if batch_results:
                    result = torch.stack(batch_results, dim=0)
                    # Reshape back to original shape
                    return result.view(original_shape)
                else:
                    return torch.zeros_like(query_embeddings)
            else:
                return torch.zeros_like(query_embeddings)
                
        except Exception as e:
            print(f"Error in PMB retrieve_semantic: {e}")
            return torch.zeros_like(query_embeddings)

    def __len__(self):
        return len(self.all_keys)