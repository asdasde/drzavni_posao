"""Neural language models over tokens plus a role embedding on every token."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import Vocabulary, roles_for
from .language_model import LanguageModel


CELLS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
ARCHITECTURES = (*CELLS, "transformer")


class Dropout(nn.Module):
    """Standard dropout, or with *locked* one mask per sequence for every time step."""

    def __init__(self, p: float, locked: bool = False) -> None:
        super().__init__()
        self.p = p
        self.locked = locked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        if not self.locked:
            return F.dropout(x, self.p, training=True)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)


class CausalSelfAttention(nn.Module):
    """Pre-norm multi-head attention over the current and earlier positions.

    The output projection starts at zero: a fresh attention layer adds a near
    constant average about 20x larger than the LSTM outputs, which drowns them
    and leaves training stuck at unigram loss (for good under fp16).
    """

    def __init__(self, size: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.attention = nn.MultiheadAttention(size, heads, dropout=dropout, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.attention(y, y, y, attn_mask=mask, need_weights=False, is_causal=True)[0]


class _Network(nn.Module):
    """Token and role embeddings in, next-token logits out.

    Right padding never reaches earlier positions in these causal models, so
    padded batches need no packing; logits at padding positions are garbage.
    """

    carries_state = False

    def __init__(
        self,
        vocab_size: int,
        num_roles: int,
        embedding_dim: int,
        role_dim: int,
        width: int,
        tie_weights: bool,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.role_embedding = nn.Embedding(num_roles, role_dim, padding_idx=0)
        if tie_weights:
            self.output_projection = nn.Linear(width, embedding_dim)
            self.output = nn.Linear(embedding_dim, vocab_size)
            self.output.weight = self.token_embedding.weight
        else:
            self.output_projection = nn.Identity()
            self.output = nn.Linear(width, vocab_size)

    def embed(self, input_ids: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.token_embedding(input_ids), self.role_embedding(roles)], dim=-1)

    def head(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.output_projection(x))


class RecurrentLM(_Network):
    """Stacked RNN, GRU or LSTM layers.

    *residual* adds pre-norm residual connections around every layer; with
    *attention_heads*, a causal self-attention layer reads the top layer's outputs.
    """

    def __init__(
        self,
        vocab_size: int,
        num_roles: int,
        *,
        cell: str = "lstm",
        embedding_dim: int = 64,
        role_dim: int = 16,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.2,
        residual: bool = False,
        attention_heads: int = 0,
        tie_weights: bool = False,
        locked_dropout: bool = False,
    ) -> None:
        super().__init__(vocab_size, num_roles, embedding_dim, role_dim, hidden_size, tie_weights)
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {tuple(CELLS)}")
        self.carries_state = attention_heads == 0
        self.residual = residual
        width = embedding_dim + role_dim
        self.input_projection = nn.Linear(width, hidden_size) if residual else nn.Identity()
        sizes = [hidden_size if residual else width] + [hidden_size] * (num_layers - 1)
        self.layers = nn.ModuleList(
            CELLS[cell](size, hidden_size, batch_first=True) for size in sizes
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in sizes) if residual else None
        self.attention = (
            CausalSelfAttention(hidden_size, attention_heads, dropout) if attention_heads else None
        )
        self.final_norm = (
            nn.LayerNorm(hidden_size) if residual or attention_heads else nn.Identity()
        )
        self.dropout = Dropout(dropout, locked_dropout)

    def forward(self, input_ids: torch.Tensor, roles: torch.Tensor, state=None):
        """Logits for the next token at every position, and every layer's final state."""
        x = self.input_projection(self.dropout(self.embed(input_ids, roles)))
        states = []
        for index, layer in enumerate(self.layers):
            layer_state = None if state is None else state[index]
            if self.residual:
                output, layer_state = layer(self.norms[index](x), layer_state)
                x = x + self.dropout(output)
            else:
                x, layer_state = layer(x, layer_state)
                x = self.dropout(x)
            states.append(layer_state)
        if self.attention is not None:
            x = x + self.dropout(self.attention(x))
        return self.head(self.final_norm(x)), states


class TransformerLM(_Network):
    """Pre-norm causal Transformer with learned positions."""

    def __init__(
        self,
        vocab_size: int,
        num_roles: int,
        *,
        embedding_dim: int = 64,
        role_dim: int = 16,
        hidden_size: int = 256,
        num_layers: int = 4,
        attention_heads: int = 4,
        dropout: float = 0.1,
        tie_weights: bool = False,
        max_positions: int = 2048,
    ) -> None:
        super().__init__(vocab_size, num_roles, embedding_dim, role_dim, hidden_size, tie_weights)
        self.input_projection = nn.Linear(embedding_dim + role_dim, hidden_size)
        self.positions = nn.Embedding(max_positions, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            attention_heads,
            4 * hidden_size,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, roles: torch.Tensor, state=None):
        length = input_ids.size(1)
        positions = self.positions(torch.arange(length, device=input_ids.device))
        x = self.dropout(self.input_projection(self.embed(input_ids, roles)) + positions)
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=input_ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.head(self.final_norm(x)), None


def build_network(
    vocab_size: int,
    num_roles: int,
    *,
    cell: str,
    embedding_dim: int,
    role_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    residual: bool,
    attention_heads: int,
    tie_weights: bool,
    locked_dropout: bool,
) -> _Network:
    if attention_heads and (hidden_size % attention_heads or hidden_size // attention_heads % 8):
        raise ValueError(
            "the attention head size must be a multiple of 8, or CUDA falls back to "
            "attention that materialises every score matrix"
        )
    if cell == "transformer":
        if residual or locked_dropout:
            raise ValueError("the transformer is always residual and has no locked dropout")
        return TransformerLM(
            vocab_size,
            num_roles,
            embedding_dim=embedding_dim,
            role_dim=role_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_heads=attention_heads,
            dropout=dropout,
            tie_weights=tie_weights,
        )
    return RecurrentLM(
        vocab_size,
        num_roles,
        cell=cell,
        embedding_dim=embedding_dim,
        role_dim=role_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        residual=residual,
        attention_heads=attention_heads,
        tie_weights=tie_weights,
        locked_dropout=locked_dropout,
    )


class NeuralLM(LanguageModel):
    """Generation for a trained network.

    For networks that carry state, the state is reused while the history only
    grows, as it does during sampling, so each step feeds just the new tokens.
    Other networks rerun the whole history every step.
    """

    def __init__(self, network: _Network, vocab: Vocabulary, device: str = "cpu") -> None:
        self.network = network.eval()
        self.vocab = vocab
        self.device = device
        self._history: list[int] = []
        self._state = None
        self._logits: torch.Tensor | None = None

    @torch.no_grad()
    def next_token_probability(self, history: Sequence[int]) -> np.ndarray:
        history = list(history)
        known = len(self._history)
        reusable = 0 < known <= len(history) and history[:known] == self._history
        if not (self.network.carries_state and reusable):
            known, self._state = 0, None
        if known < len(history):
            roles = roles_for(history, self.vocab)[known:]
            logits, self._state = self.network(
                torch.tensor([history[known:]], device=self.device),
                torch.tensor([roles], device=self.device),
                state=self._state,
            )
            self._logits = logits[0, -1]
        self._history = history
        return torch.softmax(self._logits.double(), dim=-1).cpu().numpy()
