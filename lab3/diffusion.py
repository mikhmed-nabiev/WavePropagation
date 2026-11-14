"""Learned reverse transitions for an ordered absorbing-mask process."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .grammar import COUNTS, INDEX, MAX_SLOTS, TOKENS, Derivation


# Shared structural prior for exploratory transitions and the random baseline.
PRIOR = {**dict(zip(COUNTS, (3.0, 3.0, 1.5, 0.8), strict=True)),
         'u': 5.0, 'u_t': 2.0, 'q': 2.0, 'f': 2.0,
         'D_x': 2.6, 'D_y': 2.6, 'D_z': 2.6, 'lap': 2.8,
         'vec': 0.75, 'grad': 1.8, 'scale': 0.6, 'dot': 0.65, 'div': 0.85}


def prior(legal):
    p = np.array([PRIOR.get(token, 1.0) for token in legal])
    return p / p.sum()


class Denoiser(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(len(TOKENS), 12, padding_idx=0)
            self.network = nn.Sequential(nn.Linear(12 * MAX_SLOTS + 4, 80), nn.GELU(),
                                         nn.Linear(80, len(TOKENS)))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.003)

    def forward(self, masked, context):
        return self.network(torch.cat((self.embedding(masked).flatten(1), context), dim=1))

    def update(self, pool, leaves, weights, rng, steps):
        examples = [training_rows(tokens, leaves) for tokens in pool]
        # Give each graph its own weight independently of derivation length.
        p = np.repeat(weights / [len(rows) for rows in examples], [len(rows) for rows in examples])
        x, context, allowed, target = (torch.as_tensor(np.stack(values))
                                       for values in zip(*(row for rows in examples for row in rows)))
        for _ in range(steps):
            batch = rng.choice(len(target), size=96, p=p / p.sum())
            logits = self(x[batch], context[batch]).masked_fill(~allowed[batch], -1e9)
            loss = F.cross_entropy(logits, target[batch])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()


def masked_input(state):
    mask = np.zeros(MAX_SLOTS, dtype=np.int64)
    mask[:len(state.tokens)] = [INDEX[token] for token in state.tokens]
    slot = state.pending[0]
    context = np.array([len(state.tokens) / MAX_SLOTS, slot.order / 2,
                        slot.kind == 'v', (slot.branch + 1) / 4], dtype=np.float32)
    return mask, context


def training_rows(tokens, leaves):
    # q_t deterministically masks the suffix. Sampling a row samples a noise level.
    state = Derivation(leaves)
    rows = []
    for token in tokens:
        masked, context = masked_input(state)
        allowed = np.isin(TOKENS, state.legal())
        rows.append((masked, context, allowed, INDEX[token]))
        state.add(token)
    return rows


@torch.inference_mode()
def sample_rule(state, legal, rng, denoiser):
    p = prior(legal)
    if denoiser is not None and len(legal) > 1:
        mask, context = masked_input(state)
        logits = denoiser(torch.from_numpy(mask[None]), torch.from_numpy(context[None]))[0]
        learned = (logits[[INDEX[token] for token in legal]] / 1.15).softmax(0).numpy()
        p = 0.20 * p + 0.80 * learned
    return rng.choice(legal, p=p / p.sum()).item()


def sample(leaves, rng, denoiser=None):
    state = Derivation(leaves)
    while state.pending:
        state.add(sample_rule(state, state.legal(), rng, denoiser))
    return state.equation(), state.tokens
