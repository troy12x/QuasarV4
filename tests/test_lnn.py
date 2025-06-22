# c:\quasarv4\tests\test_lnn.py

import torch
import unittest
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.lnn import LNN, LNNCell

class TestLNN(unittest.TestCase):

    def test_lnn_cell_forward_pass(self):
        """Test the forward pass of a single LNN cell."""
        input_size, hidden_size, batch_size = 10, 20, 4
        cell = LNNCell(input_size, hidden_size)
        
        t = torch.tensor(0.0)
        x = torch.randn(batch_size, hidden_size)
        u = torch.randn(batch_size, input_size)
        
        dx_dt = cell(t, x, u)
        
        self.assertEqual(dx_dt.shape, (batch_size, hidden_size))
        print("\nLNN Cell forward pass test successful.")

    def test_lnn_layer_forward_pass(self):
        """Test the forward pass of the full LNN layer with a sequence."""
        input_size, hidden_size, batch_size, seq_len = 10, 20, 4, 8
        lnn_layer = LNN(input_size, hidden_size)
        
        u_sequence = torch.randn(seq_len, batch_size, input_size)
        
        outputs = lnn_layer(u_sequence)
        
        self.assertEqual(outputs.shape, (seq_len, batch_size, hidden_size))
        print("LNN Layer forward pass test successful.")

    def test_lnn_with_residual_connection(self):
        """Test LNN with the residual connection enabled."""
        # Residual connection requires input_size == hidden_size
        size = 15
        batch_size, seq_len = 4, 8
        lnn_layer = LNN(size, size, lambda_res=0.5)
        
        u_sequence = torch.randn(seq_len, batch_size, size)
        
        outputs = lnn_layer(u_sequence)
        
        self.assertEqual(outputs.shape, (seq_len, batch_size, size))
        # A simple check to ensure it ran without errors
        self.assertIsNotNone(outputs)
        print("LNN with residual connection test successful.")

if __name__ == '__main__':
    unittest.main()
