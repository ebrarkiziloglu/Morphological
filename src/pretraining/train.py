import argparse
import logging
import math
import os
import sys
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
    set_seed,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class LineByLineTextDataset(Dataset):
    """
    Very simple line-by-line text dataset for causal LM pretraining.
    Each non-empty line is tokenized independently.
    """

    def __init__(self, tokenizer, file_path: str, max_seq_len: int):
        self.tokenizer = tokenizer
        self.examples = []

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Training file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                encoded = tokenizer(
                    line,
                    truncation=True,
                    max_length=max_seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.examples.append(
                    {
                        "input_ids": encoded["input_ids"].squeeze(0),
                        "attention_mask": encoded["attention_mask"].squeeze(0),
                    }
                )

        if len(self.examples) == 0:
            raise ValueError(f"No non-empty lines found in {file_path}")

        logger.info(
            f"Loaded {len(self.examples)} examples from {file_path} "
            f"(max_seq_len={max_seq_len})"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_model_and_tokenizer(
    tokenizer_path: str,
    lower: bool = False,
    hidden_size: int = 768,
    num_layers: int = 6,
    num_heads: int = 12,
    max_position_embeddings: int = 512,
) -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """Build a decoder-only GPT-2 style model and tokenizer.

    Uses a Hugging Face-compatible tokenizer from ``tokenizer_path``.
    This should point to ``src/tokenizer/tokenizer_char``.
    """
    logger.info(f"[Init] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _ensure_usable_tokenizer(tokenizer, tokenizer_path)

    # Build GPT-2 style config, using your desired dimensions
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=max_position_embeddings,
        n_ctx=max_position_embeddings,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    logger.info(
        f"[Init] Decoder-only model created with hidden_size={hidden_size}, "
        f"num_layers={num_layers}, num_heads={num_heads}, "
        f"max_position_embeddings={max_position_embeddings}, "
        f"vocab_size={tokenizer.vocab_size}"
    )
    logger.info(
        f"[Init] Tokenizer pad_token_id={tokenizer.pad_token_id}, "
        f"bos_token_id={getattr(tokenizer, 'bos_token_id', None)}, "
        f"eos_token_id={getattr(tokenizer, 'eos_token_id', None)}"
    )
    return model, tokenizer


def _ensure_usable_tokenizer(tokenizer, source: str):
    if tokenizer.pad_token is None:
        pad = getattr(tokenizer, "eos_token", None) or getattr(
            tokenizer, "cls_token", None
        )
        if pad is None:
            raise ValueError("Tokenizer has no pad/eos/cls token defined.")
        tokenizer.add_special_tokens({"pad_token": pad})
    vocab_len = len(tokenizer)
    if tokenizer.vocab_size == 0 or vocab_len <= 1:
        raise ValueError(
            f"Tokenizer from {source} is empty "
            f"(vocab_size={tokenizer.vocab_size}, len={vocab_len}, "
            f"pad_token_id={tokenizer.pad_token_id}). Refusing to train; "
            "this produces all-padding batches and NaN loss."
        )


def load_model_and_tokenizer_from_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
) -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """Load a saved GPT-2 LM checkpoint; tokenizer always comes from tokenizer_path."""
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    logger.info(f"[Init] Loading model from {checkpoint_path}")
    logger.info(f"[Init] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _ensure_usable_tokenizer(tokenizer, tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    logger.info(
        f"[Init] Loaded checkpoint: vocab_size={tokenizer.vocab_size}, "
        f"len(tokenizer)={len(tokenizer)}, pad_token_id={tokenizer.pad_token_id}, "
        f"model.config.vocab_size={model.config.vocab_size}"
    )
    return model, tokenizer


def reset_embedding_weights(model: GPT2LMHeadModel) -> None:
    """Re-initialise the word-token embedding layer (wte) and the tied
    output projection (lm_head) to random weights, destroying lexical
    knowledge while keeping positional embeddings and transformer layers
    intact.  Uses the same Normal(0, initializer_range) scheme that GPT-2
    applies at construction time."""
    init_range = model.config.initializer_range  # default 0.02

    # Reset input embeddings
    model.transformer.wte.weight.data.normal_(mean=0.0, std=init_range)
    logger.info(
        f"[ResetEmb] Re-initialised wte embeddings "
        f"(shape={list(model.transformer.wte.weight.shape)}, std={init_range})"
    )

    # If lm_head is weight-tied to wte the above already covers it;
    # if untied, reset it separately.
    if not model.config.tie_word_embeddings:
        model.lm_head.weight.data.normal_(mean=0.0, std=init_range)
        logger.info("[ResetEmb] Re-initialised untied lm_head weights")
    else:
        logger.info("[ResetEmb] lm_head is tied to wte; no separate reset needed")


def create_dataloader(
    file_path: str,
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, int]:
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        max_seq_len=max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    return dataloader, steps_per_epoch


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _parse_epoch_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated epoch list. Empty / None -> save every epoch."""
    if value is None or str(value).strip() == "":
        return None
    epochs = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        epochs.append(int(part))
    return epochs or None


def optimizer_steps_per_epoch(num_batches: int, grad_acc: int) -> int:
    """Number of optimizer updates per epoch (one update every ``grad_acc`` batches,
    plus a final flush if the epoch length is not divisible by ``grad_acc``)."""
    grad_acc = max(int(grad_acc), 1)
    return math.ceil(num_batches / grad_acc)


def train_one_phase(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    grad_acc: int,
    logging_steps: int,
    device: torch.device,
    phase_name: str,
    save_dir: str = None,
    tokenizer=None,
    save_epochs: Optional[List[int]] = None,
    seed: int = 0,
):
    """
    Train for exactly ``num_epochs`` passes over ``dataloader``.

    ``global_step`` / the LR schedule count **optimizer** steps, not micro-batches.
    With ``grad_acc > 1``, there are fewer optimizer steps than dataloader batches.

    If ``save_dir`` and ``tokenizer`` are set, checkpoints are written at epoch
    boundaries as ``{save_dir}/ep{E}/seed_{seed}``. When ``save_epochs`` is None,
    every epoch is saved; otherwise only the listed 1-based epoch numbers are
    saved (e.g. [5, 10]).
    """
    if num_epochs <= 0:
        logger.info(f"[{phase_name}] num_epochs={num_epochs}; skipping training.")
        return

    grad_acc = max(int(grad_acc), 1)
    batches_per_epoch = len(dataloader)
    opt_steps_epoch = optimizer_steps_per_epoch(batches_per_epoch, grad_acc)
    total_steps = num_epochs * opt_steps_epoch
    if total_steps <= 0:
        logger.info(f"[{phase_name}] total_steps={total_steps}; skipping training.")
        return

    logger.info(f"[{phase_name}] Moving model to device: {device}")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.1,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 100, 1),
        num_training_steps=total_steps,
    )

    global_step = 0
    running_loss = 0.0
    save_set = set(save_epochs) if save_epochs is not None else None

    logger.info(
        f"[{phase_name}] Starting training: {num_epochs} epochs, "
        f"{batches_per_epoch} batches/epoch, grad_acc={grad_acc} -> "
        f"{opt_steps_epoch} optimizer steps/epoch, "
        f"{total_steps} optimizer steps total, lr={lr:.2e}."
    )
    if save_dir is not None:
        layout_hint = os.path.join(save_dir, "ep{E}", f"seed_{seed}")
        if save_set is None:
            logger.info(
                f"[{phase_name}] Will save a checkpoint after every epoch as {layout_hint}"
            )
        else:
            logger.info(
                f"[{phase_name}] Will save checkpoints after epochs {sorted(save_set)} as {layout_hint}"
            )

    def _optimizer_update():
        nonlocal global_step, running_loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
        if global_step % logging_steps == 0:
            avg_loss = running_loss / logging_steps
            logger.info(
                f"[{phase_name}] step {global_step}/{total_steps} "
                f"- loss: {avg_loss:.4f} "
                f"- lr: {scheduler.get_last_lr()[0]:.2e}"
            )
            running_loss = 0.0

    for epoch in range(num_epochs):
        logger.info(f"[{phase_name}] Epoch {epoch + 1}/{num_epochs} started.")
        micro_in_acc = 0
        for step, batch in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / grad_acc
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[{phase_name}] Non-finite loss at optimizer_step={global_step}, "
                    f"batch={step}: {loss.item()!r}. This is usually an empty tokenizer "
                    "(vocab_size=0) so every token is padding and all labels are -100."
                )
            loss.backward()
            running_loss += loss.item()
            micro_in_acc += 1

            if micro_in_acc == grad_acc:
                _optimizer_update()
                micro_in_acc = 0

        # Flush leftover micro-batches that did not fill a full grad_acc window.
        if micro_in_acc > 0:
            _optimizer_update()

        # End of epoch: optionally save a checkpoint
        epoch_num = epoch + 1
        should_save = (
            save_dir is not None
            and tokenizer is not None
            and (save_set is None or epoch_num in save_set)
        )
        if should_save:
            epoch_dir = os.path.join(save_dir, f"ep{epoch_num}", f"seed_{seed}")
            os.makedirs(epoch_dir, exist_ok=True)
            logger.info(f"[{phase_name}] Saving epoch {epoch_num} checkpoint to {epoch_dir}")
            model.save_pretrained(epoch_dir)

    logger.info(
        f"[{phase_name}] Finished {num_epochs} epochs "
        f"({global_step}/{total_steps} optimizer steps)."
    )



def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-phase decoder-only pretraining for morphology experiments. "
        "Use --resume_from to skip Phase 1 and continue from a saved checkpoint."
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Directory of a Hugging Face saved model (e.g. a Phase-1 checkpoint). "
        "When set, Phase 1 is skipped and only Phase 2 is trained; "
        "architecture flags are ignored (loaded from checkpoint). "
        "The tokenizer is always loaded from --tokenizer_path, not the checkpoint.",
    )

    parser.add_argument(
        "--phase1_path",
        type=str,
        default=None,
        help="Text file for Phase 1 pretraining (required unless --resume_from is set).",
    )
    parser.add_argument(
        "--phase2_path",
        type=str,
        default=None,
        help="Text file for Phase 2 pretraining (e.g., German). "
        "Required unless --epochs_phase2 is 0.",
    )

    parser.add_argument(
        "--l1",
        type=str,
        required=True,
        help="Identifier for Phase 1 language (e.g., 'tur' or 'eng').",
    )
    parser.add_argument(
        "--l2",
        type=str,
        default="deu",
        help="Identifier for Phase 2 language (default: 'deu').",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="src/tokenizer/tokenizer_char",
        help="Path to the character tokenizer. Always used, including with "
        "--resume_from; epoch checkpoints store model weights only.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../models",
        help="Base output directory. Checkpoints are saved as "
        "{out}/ep{E}/seed_{seed}/.",
    )

    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--epochs_phase1",
        type=int,
        default=10,
        help="Number of Phase-1 epochs (default: 10).",
    )
    parser.add_argument(
        "--epochs_phase2",
        type=int,
        default=5,
        help="Number of Phase-2 epochs (default: 5). Set 0 to skip Phase 2.",
    )
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Phase 1 (default: 1e-4).",
    )
    parser.add_argument(
        "--lr_phase2",
        type=float,
        default=5e-5,
        help="Learning rate for Phase 2 (default: 5e-5).",
    )
    parser.add_argument(
        "--save_epochs_phase1",
        type=str,
        default="5,10",
        help="Comma-separated 1-based Phase-1 epochs to checkpoint "
        "(default: '5,10'). Empty string saves every epoch.",
    )
    parser.add_argument(
        "--save_epochs_phase2",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated 1-based Phase-2 epochs to checkpoint "
        "(default: '1,2,3,4,5'). Empty string saves every epoch.",
    )
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument(
        "--reset_embeddings",
        action="store_true",
        help="Re-initialise the word-token embedding layer before Phase 2 "
        "to remove lexical knowledge acquired during Phase 1.",
    )

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--max_position_embeddings", type=int, default=512)

    args = parser.parse_args()
    if args.resume_from is None and args.phase1_path is None:
        parser.error(
            "Either --phase1_path (two-phase / Phase-1 run) or --resume_from (Phase 2 only) is required."
        )
    if args.epochs_phase2 > 0 and not args.phase2_path:
        parser.error("--phase2_path is required when --epochs_phase2 > 0.")
    return args


def main():
    args = parse_args()
    logger.info(f"[Init] Parsed arguments: {args}")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Init] Using device: {device}")

    save_epochs_p1 = _parse_epoch_list(args.save_epochs_phase1)
    save_epochs_p2 = _parse_epoch_list(args.save_epochs_phase2)

    if args.resume_from:
        model, tokenizer = load_model_and_tokenizer_from_checkpoint(
            args.resume_from, tokenizer_path=args.tokenizer_path
        )
    else:
        model, tokenizer = build_model_and_tokenizer(
            tokenizer_path=args.tokenizer_path,
            lower=args.lower,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_position_embeddings=args.max_position_embeddings,
        )

    os.makedirs(args.out, exist_ok=True)

    if not args.resume_from:
        if args.epochs_phase1 <= 0:
            logger.info("[Phase1] epochs_phase1 <= 0; skipping Phase 1.")
            model.save_pretrained(args.out)
        else:
            # Phase 1: pretraining on L1 (e.g., Turkish or English)
            logger.info(
                f"[Phase1] Building dataloader from {args.phase1_path} for language {args.l1}"
            )
            train_loader_p1, batches_per_epoch_p1 = create_dataloader(
                file_path=args.phase1_path,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                num_workers=args.cpus,
            )
            opt_steps_epoch_p1 = optimizer_steps_per_epoch(
                batches_per_epoch_p1, args.grad_acc
            )
            total_steps_p1 = args.epochs_phase1 * opt_steps_epoch_p1
            logger.info(
                f"[Phase1] Training on {args.l1}: {args.epochs_phase1} epochs "
                f"x {batches_per_epoch_p1} batches/epoch, grad_acc={args.grad_acc} "
                f"-> {opt_steps_epoch_p1} optimizer steps/epoch "
                f"= {total_steps_p1} optimizer steps total "
                f"(lr={args.lr:.2e}, seed={args.seed})"
            )
            train_one_phase(
                model=model,
                dataloader=train_loader_p1,
                num_epochs=args.epochs_phase1,
                lr=args.lr,
                grad_acc=args.grad_acc,
                logging_steps=args.logging_steps,
                device=device,
                phase_name="Phase1",
                save_dir=args.out,
                tokenizer=tokenizer,
                save_epochs=save_epochs_p1,
                seed=args.seed,
            )

            # Convenience alias for the final Phase-1 state
            # phase1_dir = os.path.join(args.out, f"phase1_{args.l1}")
            # os.makedirs(phase1_dir, exist_ok=True)
            # logger.info(f"[Phase1] Saving final Phase-1 checkpoint to {phase1_dir}")
            # model.save_pretrained(phase1_dir)
            # tokenizer.save_pretrained(phase1_dir)
    else:
        logger.info(
            "[Init] Skipping Phase 1 (--resume_from set); training Phase 2 only."
        )

    if args.epochs_phase2 <= 0:
        logger.info("[Phase2] epochs_phase2 <= 0; skipping Phase 2.")
        return

    # Optionally reset embedding weights before Phase 2
    if args.reset_embeddings:
        logger.info(
            "[Phase2] --reset_embeddings is set: re-initialising embedding layer "
            "to remove lexical knowledge from Phase 1."
        )
        reset_embedding_weights(model)

    # Phase 2: continue pretraining on L2 (typically German)
    logger.info(
        f"[Phase2] Building dataloader from {args.phase2_path} for language {args.l2}"
    )
    train_loader_p2, batches_per_epoch_p2 = create_dataloader(
        file_path=args.phase2_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.cpus,
    )
    opt_steps_epoch_p2 = optimizer_steps_per_epoch(
        batches_per_epoch_p2, args.grad_acc
    )
    total_steps_p2 = args.epochs_phase2 * opt_steps_epoch_p2
    lr_phase2 = args.lr_phase2 if args.lr_phase2 is not None else args.lr
    logger.info(
        f"[Phase2] Training on {args.l2}: {args.epochs_phase2} epochs "
        f"x {batches_per_epoch_p2} batches/epoch, grad_acc={args.grad_acc} "
        f"-> {opt_steps_epoch_p2} optimizer steps/epoch "
        f"= {total_steps_p2} optimizer steps total "
        f"(lr={lr_phase2:.2e}, seed={args.seed})"
    )
    train_one_phase(
        model=model,
        dataloader=train_loader_p2,
        num_epochs=args.epochs_phase2,
        lr=lr_phase2,
        grad_acc=args.grad_acc,
        logging_steps=args.logging_steps,
        device=device,
        phase_name="Phase2",
        save_dir=args.out,
        tokenizer=tokenizer,
        save_epochs=save_epochs_p2,
        seed=args.seed,
    )

    # Final checkpoint
    # final_dir = os.path.join(args.out, f"phase2_{args.l1}_{args.l2}")
    # os.makedirs(final_dir, exist_ok=True)
    # logger.info(f"[Final] Saving final checkpoint to {final_dir}")
    # model.save_pretrained(final_dir)
    # tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
