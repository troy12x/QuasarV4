# c:\quasarv4\quasar\model.py

import torch
import torch.nn as nn
import uuid

from .lnn import LNN
from .pmb import ParameterMemoryBank
from .chunker import SemanticChunker
from .moe import MoELayer

class Quasar(nn.Module):
    """
    The main Quasar model that integrates the LNN, PMB, and Chunker.
    
    This model can:
    1. Ingest documents into its memory bank.
    2. Perform retrieval-augmented generation for a given prompt.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pmb_config=None, chunker_config=None, lnn_config=None, num_experts=0, expert_dim=0, top_k=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.use_moe = num_experts > 0
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Core LNN for processing sequences
        # Note: input_size for LNN is embedding_dim, and it outputs hidden_dim states.
        self.lnn = LNN(input_size=embedding_dim, hidden_size=hidden_dim, **(lnn_config or {}))

        if self.use_moe:
            self.moe_layer = MoELayer(
                embedding_dim=hidden_dim, 
                num_experts=num_experts, 
                expert_dim=expert_dim,
                top_k=top_k
            )

        if pmb_config is None: pmb_config = {}
        self.pmb = ParameterMemoryBank(**pmb_config)

        # Semantic Chunker
        if chunker_config is None: chunker_config = {}
        # The chunker needs a model to generate dense embeddings. We will use
        # this Quasar model's own LNN encoder for that purpose.
        self.chunker = SemanticChunker(**chunker_config)

        # Output layer for next-token prediction
        # It takes the LNN hidden state and potentially a memory vector
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def get_dense_embedding(self, input_ids):
        """Generates a dense embedding for a tokenized chunk using the LNN."""
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            device = next(self.parameters()).device
            input_ids = input_ids.to(device)

            # Ensure input is 2D (batch_size, seq_len)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            embedded_input = self.embedding(input_ids)
            # LNN expects (seq, batch, dim)
            lnn_outputs = self.lnn(embedded_input.transpose(0, 1))
            # Use the last hidden state as the chunk's dense embedding
            dense_embedding = lnn_outputs[-1] # (batch, hidden_dim)
        self.train() # Set back to training mode
        return dense_embedding.squeeze(0) # Return (hidden_dim) if batch was 1

    def ingest(self, document_text):
        """
        Chunks a document and stores it in the Parameter Memory Bank.
        """
        print(f"Ingesting document...\n")
        # 1. Chunk the document
        chunks = self.chunker.chunk_document(document_text)
        print(f"Document split into {len(chunks)} chunks.")

        # 2. Generate embeddings for each chunk
        for chunk in chunks:
            # Generate a unique ID for the chunk
            chunk_id = uuid.uuid4()
            
            # Get dense embedding from the LNN encoder part of Quasar
            dense_embedding = self.get_dense_embedding(chunk)
            
            # For now, sparse embedding is handled abstractly inside the chunker.
            # We will just use the dense part for the key.
            key_embedding = dense_embedding

            # 3. Store in PMB
            self.pmb.store(chunk_id, key_embedding, chunk)
        
        print(f"Ingestion complete. PMB now contains {len(self.pmb)} items.")

    def forward(self, input_ids, memory_query=True, top_k=1):
        """
        Forward pass for generation.
        
        Args:
            input_ids (torch.Tensor): Tensor of token IDs for the prompt.
                                      Shape: (batch_size, seq_len)
            memory_query (bool): If True, performs retrieval from PMB.
            top_k (int): Number of memory items to retrieve.

        Returns:
            torch.Tensor: Logits for the next token. Shape: (batch_size, vocab_size)
        """
        load_balancing_loss = None

        
        # 1. Embed the input prompt
        embedded_input = self.embedding(input_ids)
        
        # 2. Process with LNN
        # LNN expects (seq_len, batch, dim), so we keep the current shape
        hidden_states = self.lnn(embedded_input)
        
        # Transpose to (batch, seq_len, dim) for MoE layer
        hidden_states = hidden_states.transpose(0, 1)

        # 3. Route through MoE layer if enabled
        if self.use_moe:
            hidden_states, routing_weights = self.moe_layer(hidden_states)
            # Flatten routing weights for loss calculation
            load_balancing_loss = self.moe_layer.get_load_balancing_loss(routing_weights.view(-1, routing_weights.shape[-1]))

        # 4. (Optional) Retrieve from memory
        # We'll use the last hidden state as the query vector for simplicity
        query_vector = hidden_states[:, -1, :]
        if memory_query and self.pmb.is_populated():
            retrieved_memory = self.pmb.retrieve_semantic(query_vector, k=1)
            if retrieved_memory:
                memory_tensor = retrieved_memory[0].unsqueeze(0).to(hidden_states.device)
                # Simple blending: add memory to the hidden states
                hidden_states = hidden_states + memory_tensor

        # 5. Generate output logits from the final hidden state
        logits = self.output_head(hidden_states[:, -1, :])
        
        if self.use_moe:
            return logits, load_balancing_loss
        return logits
