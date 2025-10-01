"""
Attention Mechanism and Positional Encoding Implementation
=========================================================

This file contains all the key implementations from the notebook, organized 
into clear sections with detailed explanations for better understanding.

Author: Extracted from IBM Skills Network Lab
Purpose: Understanding self-attention and positional encoding concepts
"""

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Text Processing Libraries  
from Levenshtein import distance
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: BASIC SETUP AND HYPERPARAMETERS
# =============================================================================

print("🚀 Setting up Attention and Positional Encoding Implementation")
print("=" * 60)

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Training Parameters
learning_rate = 3e-4
batch_size = 64
max_iters = 5000
eval_interval = 200
eval_iters = 100

# Model Architecture Parameters
max_vocab_size = 256
vocab_size = max_vocab_size
block_size = 16        # Context length for predictions
n_embd = 32           # Embedding size
num_heads = 2         # Number of attention heads
n_layer = 2           # Number of transformer blocks
ff_scale_factor = 4   # Feed-forward scaling factor
dropout = 0.0         # Dropout rate

# Calculate head size and validate
head_size = n_embd // num_heads
assert (num_heads * head_size) == n_embd, "Embedding size must be divisible by number of heads"

print(f"Architecture: {n_embd}D embeddings, {num_heads} heads, {n_layer} layers")

# =============================================================================
# SECTION 2: SIMPLE TRANSLATION - FROM DICTIONARY TO NEURAL NETWORKS
# =============================================================================

class SimpleTranslator:
    """
    Demonstrates the evolution from simple dictionary lookup to neural network-based translation
    """
    
    def __init__(self):
        # French to English dictionary
        self.dictionary = {
            'le': 'the',
            'chat': 'cat', 
            'est': 'is',
            'sous': 'under',
            'la': 'the',
            'table': 'table'
        }
        
        # Create vocabularies
        self.vocabulary_in = sorted(list(set(self.dictionary.keys())))
        self.vocabulary_out = sorted(list(set(self.dictionary.values())))
        
        print(f"\n📚 TRANSLATION SETUP")
        print(f"Input vocabulary: {self.vocabulary_in}")
        print(f"Output vocabulary: {self.vocabulary_out}")
        
        # Create one-hot encodings
        self.one_hot_in = self._encode_one_hot(self.vocabulary_in)
        self.one_hot_out = self._encode_one_hot(self.vocabulary_out)
        
        # Create matrices for neural network approach
        self.K = torch.stack([self.one_hot_in[k] for k in self.dictionary.keys()])
        self.V = torch.stack([self.one_hot_out[k] for k in self.dictionary.values()])
        
        print(f"K matrix shape (Keys): {self.K.shape}")
        print(f"V matrix shape (Values): {self.V.shape}")
    
    def tokenize(self, text):
        """Split text into tokens (words)"""
        return text.split()
    
    def _encode_one_hot(self, vocabulary):
        """Convert vocabulary words to one-hot encoded vectors"""
        one_hot = {}
        vocab_size = len(vocabulary)
        
        for i, word in enumerate(vocabulary):
            vector = torch.zeros(vocab_size)
            vector[i] = 1
            one_hot[word] = vector
            
        return one_hot
    
    def find_closest_key(self, query):
        """Find closest dictionary key using Levenshtein distance"""
        closest_key, min_dist = None, float('inf')
        
        for key in self.dictionary.keys():
            dist = distance(query, key)
            if dist < min_dist:
                min_dist, closest_key = dist, key
                
        return closest_key
    
    def decode_one_hot(self, one_hot_dict, vector):
        """Decode one-hot vector back to token"""
        best_key, best_sim = None, 0
        
        for key, vec in one_hot_dict.items():
            similarity = torch.dot(vector, vec).item()
            if similarity > best_sim:
                best_sim, best_key = similarity, key
                
        return best_key
    
    def translate_basic(self, sentence):
        """Basic dictionary translation"""
        result = ""
        for token in self.tokenize(sentence):
            key = self.find_closest_key(token)
            result += self.dictionary[key] + " "
        return result.strip()
    
    def translate_matrix(self, sentence):
        """Matrix-based translation using Q @ K.T @ V"""
        result = ""
        for token in self.tokenize(sentence):
            if token in self.one_hot_in:
                q = self.one_hot_in[token]
                output = q @ self.K.T @ self.V
                translated = self.decode_one_hot(self.one_hot_out, output)
                result += translated + " "
        return result.strip()
    
    def translate_attention(self, sentence):
        """Translation using attention mechanism with softmax"""
        result = ""
        for token in self.tokenize(sentence):
            if token in self.one_hot_in:
                q = self.one_hot_in[token]
                # Apply attention: softmax(q @ K.T) @ V
                attention_weights = torch.softmax(q @ self.K.T, dim=0)
                output = attention_weights @ self.V
                translated = self.decode_one_hot(self.one_hot_out, output)
                result += translated + " "
        return result.strip()
    
    def translate_parallel(self, sentence):
        """Parallel translation using Q matrix"""
        tokens = self.tokenize(sentence)
        # Create Q matrix for all tokens
        Q = torch.stack([self.one_hot_in[token] for token in tokens if token in self.one_hot_in])
        
        # Parallel attention computation
        attention_weights = torch.softmax(Q @ self.K.T, dim=1)
        outputs = attention_weights @ self.V
        
        # Decode all outputs
        translated_tokens = [self.decode_one_hot(self.one_hot_out, output) for output in outputs]
        return " ".join(translated_tokens)
    
    def demonstrate_all_methods(self):
        """Demonstrate all translation approaches"""
        test_sentence = "le chat est sous la table"
        
        print(f"\n🔄 TRANSLATION DEMONSTRATION")
        print(f"Input: '{test_sentence}'")
        print("-" * 40)
        print(f"Basic:     {self.translate_basic(test_sentence)}")
        print(f"Matrix:    {self.translate_matrix(test_sentence)}")
        print(f"Attention: {self.translate_attention(test_sentence)}")
        print(f"Parallel:  {self.translate_parallel(test_sentence)}")

# =============================================================================
# SECTION 3: POSITIONAL ENCODING IMPLEMENTATIONS
# =============================================================================

class PositionalEncodingExamples:
    """
    Demonstrates different approaches to positional encoding
    """
    
    def __init__(self, max_length=100, embedding_dim=3):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
    def linear_encoding(self, scale=1.0):
        """Simple linear positional encoding"""
        pe = torch.zeros(self.max_length, self.embedding_dim)
        pe = torch.cat([scale * self.position] * self.embedding_dim, dim=1)
        return pe
    
    def scaled_linear_encoding(self):
        """Scaled linear encoding to reduce magnitude issues"""
        pe = torch.cat([
            0.1 * self.position,    # Dimension 1: slow increase
            -0.1 * self.position,   # Dimension 2: slow decrease  
            0 * self.position       # Dimension 3: constant
        ], dim=1)
        return pe
    
    def sinusoidal_encoding(self, frequency=6):
        """Sinusoidal positional encoding"""
        pe = torch.cat([
            torch.sin(2 * np.pi * self.position / frequency),  # Sine wave
            torch.ones_like(self.position),                    # Constant
            torch.ones_like(self.position)                     # Constant
        ], dim=1)
        return pe
    
    def mixed_sinusoidal_encoding(self):
        """Mixed frequency sinusoidal encoding (more realistic)"""
        pe = torch.cat([
            torch.cos(2 * np.pi * self.position / 25),  # Low frequency cosine
            torch.sin(2 * np.pi * self.position / 25),  # Low frequency sine
            torch.sin(2 * np.pi * self.position / 5)    # High frequency sine
        ], dim=1)
        return pe
    
    def visualize_encodings(self):
        """Visualize different positional encoding approaches"""
        encodings = {
            'Linear': self.linear_encoding(),
            'Scaled Linear': self.scaled_linear_encoding(),
            'Simple Sinusoidal': self.sinusoidal_encoding(),
            'Mixed Sinusoidal': self.mixed_sinusoidal_encoding()
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, encoding) in enumerate(encodings.items()):
            ax = axes[i]
            
            # Plot first 50 positions for clarity
            for dim in range(self.embedding_dim):
                ax.plot(encoding[:50, dim].numpy(), label=f'Dim {dim+1}')
            
            ax.set_title(f'{name} Positional Encoding')
            ax.set_xlabel('Position')
            ax.set_ylabel('Encoding Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("📊 Positional Encoding Comparison:")
        for name, encoding in encodings.items():
            print(f"{name:20}: Range [{encoding.min():.2f}, {encoding.max():.2f}]")

# =============================================================================
# SECTION 4: SELF-ATTENTION IMPLEMENTATION
# =============================================================================

class SelfAttentionHead(nn.Module):
    """
    Self-attention head implementation from scratch
    """
    
    def __init__(self, vocab_size, n_embd, head_size, dropout=0.0):
        super().__init__()
        self.head_size = head_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        # Linear projections for Q, K, V
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Create causal mask for decoder-style attention
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T = x.shape  # Batch size, Sequence length
        
        # Get embeddings
        embeddings = self.embedding(x)  # (B, T, n_embd)
        
        # Compute Q, K, V
        k = self.key(embeddings)        # (B, T, head_size)
        q = self.query(embeddings)      # (B, T, head_size)
        v = self.value(embeddings)      # (B, T, head_size)
        
        # Compute attention scores
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)
        
        # Apply causal mask (for decoder-style attention)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = attention_weights @ v  # (B, T, head_size)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implementation
    """
    
    def __init__(self, vocab_size, n_embd, num_heads, dropout=0.0):
        super().__init__()
        assert n_embd % num_heads == 0
        
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        
        # Multiple attention heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(vocab_size, n_embd, self.head_size, dropout) 
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Run all heads in parallel and concatenate results
        head_outputs = [head(x)[0] for head in self.heads]  # Get outputs, ignore attention weights
        
        # Concatenate all heads
        out = torch.cat(head_outputs, dim=-1)  # (B, T, n_embd)
        
        # Final projection
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

# =============================================================================
# SECTION 5: COMPLETE TRANSFORMER COMPONENTS
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers
    """
    
    def __init__(self, n_embd, max_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_length, n_embd)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * 
                           (-np.log(10000.0) / n_embd))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        if n_embd % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_length, 1, n_embd)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """
    Complete transformer block with self-attention and feed-forward layers
    """
    
    def __init__(self, vocab_size, n_embd, num_heads, dropout=0.0):
        super().__init__()
        
        # Multi-head self attention
        self.attention = MultiHeadAttention(vocab_size, n_embd, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, ff_scale_factor * n_embd),
            nn.ReLU(),
            nn.Linear(ff_scale_factor * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.ln1(x))
        
        # Feed-forward with residual connection  
        x = x + self.feed_forward(self.ln2(x))
        
        return x

class SimpleTransformer(nn.Module):
    """
    Complete transformer model combining all components
    """
    
    def __init__(self, vocab_size, n_embd, num_heads, n_layers, max_length=1000, dropout=0.0):
        super().__init__()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(n_embd, max_length, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(vocab_size, n_embd, num_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output head
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        
        # Token embeddings
        tok_emb = self.token_embedding(x)  # (B, T, n_embd)
        
        # Add positional encoding
        x = self.pos_encoding(tok_emb.transpose(0, 1)).transpose(0, 1)  # (B, T, n_embd)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits

# =============================================================================
# SECTION 6: DEMONSTRATION AND TESTING
# =============================================================================

class AttentionDemo:
    """
    Comprehensive demonstration of all attention and positional encoding concepts
    """
    
    def __init__(self):
        print("\n🎯 ATTENTION AND POSITIONAL ENCODING DEMO")
        print("=" * 50)
    
    def run_translation_demo(self):
        """Demonstrate translation evolution"""
        print("\n1. TRANSLATION EVOLUTION DEMO")
        translator = SimpleTranslator()
        translator.demonstrate_all_methods()
    
    def run_positional_encoding_demo(self):
        """Demonstrate positional encoding approaches"""
        print("\n2. POSITIONAL ENCODING DEMO")
        pos_encoder = PositionalEncodingExamples()
        pos_encoder.visualize_encodings()
    
    def run_attention_demo(self):
        """Demonstrate self-attention mechanisms"""
        print("\n3. SELF-ATTENTION DEMO")
        
        # Create sample data
        vocab_size_demo = 50
        seq_length = 8
        batch_size = 2
        
        # Random token indices
        sample_data = torch.randint(0, vocab_size_demo, (batch_size, seq_length))
        
        print(f"Input shape: {sample_data.shape}")
        print(f"Sample tokens: {sample_data[0].tolist()}")
        
        # Single attention head
        single_head = SelfAttentionHead(vocab_size_demo, n_embd, head_size)
        output, attention_weights = single_head(sample_data)
        
        print(f"\nSingle Head Output shape: {output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Attention weights sample:\n{attention_weights[0, :3, :3]}")
        
        # Multi-head attention
        multi_head = MultiHeadAttention(vocab_size_demo, n_embd, num_heads)
        multi_output = multi_head(sample_data)
        
        print(f"\nMulti-Head Output shape: {multi_output.shape}")
    
    def run_transformer_demo(self):
        """Demonstrate complete transformer"""
        print("\n4. COMPLETE TRANSFORMER DEMO")
        
        # Create transformer model
        model = SimpleTransformer(
            vocab_size=100,
            n_embd=n_embd,
            num_heads=num_heads,
            n_layers=n_layer,
            dropout=dropout
        )
        
        # Sample input
        sample_input = torch.randint(0, 100, (2, 10))  # (batch_size, seq_len)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Input shape: {sample_input.shape}")
        
        # Forward pass
        output = model(sample_input)
        print(f"Output shape: {output.shape}")
        print(f"Output represents logits for {output.shape[-1]} vocabulary items")
    
    def run_all_demos(self):
        """Run all demonstrations"""
        self.run_translation_demo()
        self.run_positional_encoding_demo() 
        self.run_attention_demo()
        self.run_transformer_demo()
        
        print("\n✅ ALL DEMONSTRATIONS COMPLETED!")
        print("=" * 50)

# =============================================================================
# SECTION 7: PYTORCH BUILT-IN TRANSFORMERS EXAMPLES  
# =============================================================================

def pytorch_transformer_examples():
    """
    Examples using PyTorch's built-in transformer components
    """
    
    print("\n🔧 PYTORCH BUILT-IN TRANSFORMERS")
    print("=" * 40)
    
    # MultiheadAttention example
    embed_dim = 64
    num_heads = 8
    
    multihead_attn = nn.MultiheadAttention(
        embed_dim=embed_dim, 
        num_heads=num_heads,
        batch_first=True
    )
    
    # Sample data
    seq_len, batch_size = 10, 3
    query = torch.rand(batch_size, seq_len, embed_dim)
    key = torch.rand(batch_size, seq_len, embed_dim)
    value = torch.rand(batch_size, seq_len, embed_dim)
    
    attn_output, attn_weights = multihead_attn(query, key, value)
    
    print(f"MultiheadAttention:")
    print(f"  Input shape: {query.shape}")
    print(f"  Output shape: {attn_output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    
    # TransformerEncoder example
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        batch_first=True
    )
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    src = torch.rand(batch_size, seq_len, embed_dim)
    encoded = transformer_encoder(src)
    
    print(f"\nTransformerEncoder:")
    print(f"  Input shape: {src.shape}")
    print(f"  Output shape: {encoded.shape}")
    
    # Full Transformer example
    transformer_model = nn.Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        batch_first=True
    )
    
    src = torch.rand(2, 10, 512)
    tgt = torch.rand(2, 8, 512)
    
    out = transformer_model(src, tgt)
    
    print(f"\nFull Transformer:")
    print(f"  Src shape: {src.shape}")
    print(f"  Tgt shape: {tgt.shape}")
    print(f"  Output shape: {out.shape}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 ATTENTION AND POSITIONAL ENCODING IMPLEMENTATION")
    print("=" * 60)
    print("This file demonstrates the key concepts from the notebook:")
    print("1. Evolution from dictionary to neural network translation")
    print("2. Different positional encoding approaches")
    print("3. Self-attention mechanism implementation")
    print("4. Complete transformer architecture")
    print("5. PyTorch built-in transformer examples")
    print("=" * 60)
    
    try:
        # Run comprehensive demo
        demo = AttentionDemo()
        demo.run_all_demos()
        
        # Show PyTorch built-in examples
        pytorch_transformer_examples()
        
        print("\n🎉 SUCCESS: All implementations completed successfully!")
        print("You can now explore and modify any of the classes/functions above.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Make sure all required libraries are installed:")
        print("pip install torch torchtext python-Levenshtein matplotlib numpy")

    print("\n" + "=" * 60)
    print("📚 KEY CONCEPTS DEMONSTRATED:")
    print("✅ One-hot encoding and matrix operations")
    print("✅ Query-Key-Value attention mechanism")  
    print("✅ Softmax attention weights")
    print("✅ Positional encoding (linear vs sinusoidal)")
    print("✅ Multi-head self-attention")
    print("✅ Complete transformer architecture")
    print("✅ Residual connections and layer normalization")
    print("=" * 60)