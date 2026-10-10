"""Train a neural language model on rolling turn windows and report BPC."""

import argparse
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ...transcription.constants import CORE_CHARACTERS
from .bpe import BpeVocabulary, build_vocabulary
from .dataset import ROLES, CharVocabulary, TurnWindowDataset, collate_windows
from .metrics import bits_per_character, scored_characters
from .model import NeuralLM, build_network
from .prepare import group_by_split, read_jsonl


NETWORK_FIELDS = (
    "cell",
    "embedding_dim",
    "role_dim",
    "hidden_size",
    "num_layers",
    "dropout",
    "residual",
    "attention_heads",
    "tie_weights",
    "locked_dropout",
)


@dataclass(frozen=True)
class TrainConfig:
    cell: str = "lstm"
    embedding_dim: int = 64
    role_dim: int = 16
    hidden_size: int = 512
    num_layers: int = 3
    dropout: float = 0.2
    residual: bool = False
    attention_heads: int = 0
    tie_weights: bool = False
    locked_dropout: bool = False
    context_turns: int = 5
    max_context_chars: int = 1000
    premise_mode: str = "none"
    bpe_vocab_size: int | None = None
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 20
    patience: int = 3
    seed: int = 0


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def masked_nll(network: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Summed negative log-likelihood, in nats, over the loss-masked targets."""
    logits, _ = network(batch["input_ids"], batch["roles"])
    nll = F.cross_entropy(logits.transpose(1, 2), batch["target_ids"], reduction="none")
    return (nll * batch["loss_mask"]).sum()


def length_batches(
    lengths: list[int], batch_size: int, rng: random.Random, pool: int = 50
) -> list[list[int]]:
    """Shuffled batches of similar-length examples, so little compute goes to padding.

    Examples are shuffled, then sorted by length within pools of *pool* batches.
    """
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size * pool):
        chunk = sorted(order[start : start + batch_size * pool], key=lengths.__getitem__)
        batches += [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(
    network: torch.nn.Module,
    dataset: TurnWindowDataset,
    characters: int,
    batch_size: int,
    device: str,
) -> float:
    network.eval()
    lengths = _lengths(dataset)
    order = sorted(range(len(dataset)), key=lengths.__getitem__)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate_windows)
    total = sum(masked_nll(network, _to(batch, device)).item() for batch in loader)
    return bits_per_character(total, characters)


def train(
    config: TrainConfig,
    episodes: list[dict],
    device: str = "cpu",
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[NeuralLM, dict[str, float]]:
    """Train on the train split, stop early on validation BPC, report test BPC.

    The returned model holds the weights of the best validation epoch.
    """
    if config.premise_mode == "encoder":
        raise NotImplementedError("the premise encoder is not implemented yet")
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    by_split = group_by_split(episodes)
    vocab = build_vocabulary(by_split["train"], config.bpe_vocab_size)
    datasets = {
        split: TurnWindowDataset(
            group,
            vocab,
            context_turns=config.context_turns,
            max_context_chars=config.max_context_chars,
            premise_mode=config.premise_mode,
        )
        for split, group in by_split.items()
    }
    characters = {split: scored_characters(group) for split, group in by_split.items()}
    network = build_network(
        len(vocab), len(ROLES), **{name: getattr(config, name) for name in NETWORK_FIELDS}
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    train_lengths = _lengths(datasets["train"])

    best_bpc, best_state, stale = math.inf, None, 0
    for epoch in range(1, config.epochs + 1):
        network.train()
        train_nll = 0.0
        loader = DataLoader(
            datasets["train"],
            batch_sampler=length_batches(train_lengths, config.batch_size, rng),
            collate_fn=collate_windows,
        )
        for batch in loader:
            batch = _to(batch, device)
            nll = masked_nll(network, batch)
            optimizer.zero_grad()
            (nll / batch["loss_mask"].sum()).backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
            optimizer.step()
            train_nll += nll.item()
        validation_bpc = evaluate(
            network, datasets["validation"], characters["validation"], config.batch_size, device
        )
        scheduler.step(validation_bpc)
        if on_epoch is not None:
            on_epoch(
                epoch,
                {
                    "train_bpc": bits_per_character(train_nll, characters["train"]),
                    "validation_bpc": validation_bpc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )
        if validation_bpc < best_bpc:
            best_bpc, stale = validation_bpc, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break

    network.load_state_dict(best_state)
    results = {
        "validation_bpc": best_bpc,
        "test_bpc": evaluate(
            network, datasets["test"], characters["test"], config.batch_size, device
        ),
        "epochs": epoch,
        "parameters": sum(p.numel() for p in network.parameters()),
    }
    return NeuralLM(network, vocab, device), results


def save_checkpoint(model: NeuralLM, config: TrainConfig, path: Path) -> None:
    """Weights, configuration and, for BPE, the tokenizer."""
    tokenizer = model.vocab.tokenizer if isinstance(model.vocab, BpeVocabulary) else None
    torch.save(
        {
            "config": asdict(config),
            "state_dict": model.network.state_dict(),
            "tokenizer": None if tokenizer is None else tokenizer.to_str(),
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[NeuralLM, TrainConfig]:
    saved = torch.load(path, map_location=device)
    config = TrainConfig(**saved["config"])
    if saved["tokenizer"] is None:
        vocab = CharVocabulary()
    else:
        vocab = BpeVocabulary(Tokenizer.from_str(saved["tokenizer"]))
    network = build_network(
        len(vocab), len(ROLES), **{name: getattr(config, name) for name in NETWORK_FIELDS}
    )
    network.load_state_dict(saved["state_dict"])
    return NeuralLM(network.to(device), vocab, device), config


def _lengths(dataset: TurnWindowDataset) -> list[int]:
    return [len(dataset.example(index)["input_ids"]) for index in range(len(dataset))]


def _to(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _argument_type(field) -> Callable[[str], object]:
    if isinstance(field.default, bool):
        return lambda text: text.lower() in ("1", "true", "yes")
    return int if field.default is None else type(field.default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=Path("data/lstm/episodes.jsonl"))
    parser.add_argument("--limit", type=int, help="Keep the first N episodes of every split.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--samples", type=int, default=1, help="Turns to sample per speaker.")
    for field in fields(TrainConfig):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}", type=_argument_type(field), default=field.default
        )
    args = parser.parse_args()
    config = TrainConfig(**{field.name: getattr(args, field.name) for field in fields(TrainConfig)})

    episodes = [
        e for group in group_by_split(read_jsonl(args.episodes)).values() for e in group[: args.limit]
    ]
    model, results = train(
        config,
        episodes,
        args.device,
        on_epoch=lambda epoch, metrics: print(
            f"epoch {epoch}: " + ", ".join(f"{k} {v:.4g}" for k, v in metrics.items())
        ),
    )
    print(results)

    rng = np.random.default_rng(config.seed)
    newline = model.vocab.ids["\n"]
    for speaker in CORE_CHARACTERS:
        tag = f"[{speaker}]"
        for _ in range(args.samples):
            ids = model.generate([model.vocab.ids[tag]], newline, rng)
            print(f"  {tag} {model.vocab.decode(ids).strip()}")


if __name__ == "__main__":
    main()
