#!/usr/bin/env python3
"""
Evaluate Hub checkpoints on gold vs. corrupted sentence pairs.

Same PPL-preference protocol as ``evaluate_ppl.py``, but loads weights from a
Hugging Face model repo (``--repo_id`` + ``--subdir``) instead of a local tree.

If ``--json_path`` is omitted, testsets are chosen from the Hub directory path:
  * path contains ``shifted`` -> ``*_pairs_shifted.json``
  * otherwise -> regular ``*_pairs.json``

CSV outputs go under ``eval_results/{repo}/{subdir_underscored}/`` and
include ``repo`` and ``subdir`` on every row.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Compute nodes cannot use Xet CAS; default Hub download timeout is 10s.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HOME", "/home/vault/b279bb/b279bb15/hf_home")

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import evaluate_ppl as ppl  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "src", "eval_dataset")
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "results")
DEFAULT_TOKENIZER_PATH = os.path.join(REPO_ROOT, "models", "v3")
DEFAULT_STAGE_DIR = "/home/vault/b279bb/b279bb15/hf_stage"
DEFAULT_HF_USER = "ebrarkiziloglu"

REGULAR_TESTSETS = (
    "dative_accusative_pairs.json",
    "accusative_dative_pairs.json",
)
SHIFTED_TESTSETS = (
    "dative_accusative_pairs_shifted.json",
    "accusative_dative_pairs_shifted.json",
)
CHECKPOINT_MARKERS = ("config.json", "model.safetensors", "pytorch_model.bin")


def normalize_repo_id(repo_id: str, hf_user: str) -> str:
    repo_id = repo_id.strip()
    if "/" not in repo_id:
        return f"{hf_user}/{repo_id}"
    return repo_id


def path_is_shifted(directory_path: str) -> bool:
    return "shifted" in directory_path.lower()


def run_name_for(repo_id: str, directory_path: str) -> str:
    repo_short = repo_id.split("/")[-1]
    slug = directory_path.strip("/").replace("/", "_") or "root"
    return f"{repo_short}_{slug}"


def resolve_testsets(
    directory_path: str,
    dataset_dir: str,
    json_path_override: Optional[str],
) -> List[str]:
    if json_path_override:
        return [
            os.path.expanduser(p.strip())
            for p in json_path_override.split(",")
            if p.strip()
        ]
    names = SHIFTED_TESTSETS if path_is_shifted(directory_path) else REGULAR_TESTSETS
    paths = [os.path.join(dataset_dir, name) for name in names]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing testset files: {missing}")
    return paths


def list_repo_files(
    api: HfApi,
    repo_id: str,
    revision: str,
    token: Optional[str],
) -> Set[str]:
    items = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        recursive=True,
        token=token,
    )
    return {item.path for item in items if getattr(item, "path", None)}


def subdir_exists(files: Set[str], directory_path: str) -> bool:
    """True if *directory_path* is a file or prefix of any path in the repo listing."""
    directory_path = directory_path.strip("/")
    if not directory_path:
        return bool(files)
    prefix = f"{directory_path}/"
    return any(path == directory_path or path.startswith(prefix) for path in files)


def is_hf_checkpoint(files: Set[str], folder: str) -> bool:
    folder = folder.strip("/")
    prefix = f"{folder}/" if folder else ""
    return any(f"{prefix}{name}" in files for name in CHECKPOINT_MARKERS)


def parse_epoch_seed(folder: str) -> Tuple[Optional[int], Optional[int]]:
    epoch: Optional[int] = None
    seed: Optional[int] = None
    for part in folder.strip("/").split("/"):
        if part.startswith("ep") and part[2:].isdigit():
            epoch = int(part[2:])
        elif part.startswith("seed_") and part[5:].isdigit():
            seed = int(part[5:])
    return epoch, seed


def _checkpoint_folders(files: Set[str], directory_path: str) -> Set[str]:
    """Hub folders under *directory_path* that look like model checkpoints."""
    directory_path = directory_path.strip("/")
    prefix = f"{directory_path}/" if directory_path else ""
    folders: Set[str] = set()
    for path in files:
        if not any(path == name or path.endswith(f"/{name}") for name in CHECKPOINT_MARKERS):
            continue
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        if directory_path:
            if folder != directory_path and not folder.startswith(prefix):
                continue
        folders.add(folder)
    return folders


def discover_hf_checkpoints(
    files: Set[str],
    directory_path: str,
) -> List[Tuple[int, int, str]]:
    """Find ``ep{{k}}/seed_{{s}}`` (or seedless ``ep{{k}}``) checkpoints on the Hub."""
    directory_path = directory_path.strip("/")
    prefix = f"{directory_path}/" if directory_path else ""
    found: List[Tuple[int, int, str]] = []
    seen: Set[str] = set()

    for folder in sorted(_checkpoint_folders(files, directory_path)):
        rel = folder[len(prefix):] if prefix and folder.startswith(prefix) else folder
        if directory_path and folder == directory_path:
            continue
        parts = [p for p in rel.split("/") if p]
        if not parts or not (parts[0].startswith("ep") and parts[0][2:].isdigit()):
            continue
        if len(parts) == 1:
            pass
        elif len(parts) == 2 and parts[1].startswith("seed_"):
            pass
        else:
            continue
        if folder in seen:
            continue
        seen.add(folder)
        epoch, seed = parse_epoch_seed(folder)
        if epoch is None:
            continue
        found.append((epoch, seed if seed is not None else -1, folder))
    return found


def resolve_hf_checkpoints(
    files: Set[str],
    directory_path: str,
    epochs: Sequence[int],
    seeds: Sequence[int],
) -> List[Tuple[int, int, str]]:
    """Return ``(epoch, seed, subfolder)`` for each Hub checkpoint to evaluate.

    Missing epochs / seeds under the subdir are skipped. Only combinations
    that actually exist on the Hub (and are in *epochs* / *seeds*) are kept.
    """
    directory_path = directory_path.strip("/")
    if is_hf_checkpoint(files, directory_path):
        epoch, seed = parse_epoch_seed(directory_path)
        return [(epoch if epoch is not None else -1, seed if seed is not None else -1, directory_path)]

    discovered = discover_hf_checkpoints(files, directory_path)
    available_epochs = sorted({ep for ep, _, _ in discovered})
    requested_epochs = set(epochs)
    requested_seeds = set(seeds)
    missing_epochs = [ep for ep in epochs if ep not in set(available_epochs)]
    if missing_epochs:
        logger.info(
            f"Skipping missing epochs under {directory_path or '/'}: "
            f"{missing_epochs} (available={available_epochs or 'none'})"
        )

    checkpoints: List[Tuple[int, int, str]] = []
    seen: Set[str] = set()
    for epoch, seed, folder in discovered:
        if epoch not in requested_epochs:
            continue
        if seed != -1 and seed not in requested_seeds:
            continue
        if folder in seen:
            continue
        seen.add(folder)
        checkpoints.append((epoch, seed, folder))

    available_for_requested = sorted({ep for ep, _, _ in checkpoints})
    logger.info(
        f"Evaluating {len(checkpoints)} checkpoint(s) under {directory_path or '/'}: "
        f"epochs={available_for_requested}"
    )
    return checkpoints


def load_local_tokenizer(tokenizer_path: str):
    tokenizer_path = os.path.expanduser(tokenizer_path)
    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    return ppl.load_tokenizer(tokenizer_path)


def _exception_chain(exc: BaseException) -> str:
    parts: List[str] = [f"{type(exc).__name__}: {exc}"]
    cur: Optional[BaseException] = exc.__cause__ or exc.__context__
    seen: Set[int] = {id(exc)}
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " <- ".join(parts)


def _is_not_found(exc: BaseException) -> bool:
    text = _exception_chain(exc)
    return "404" in text or "Entry Not Found" in text or "RemoteEntryNotFound" in text


def staged_checkpoint_dir(stage_dir: str, repo_id: str, subfolder: str) -> str:
    return os.path.join(os.path.expanduser(stage_dir), repo_id, subfolder.strip("/"))


def is_local_checkpoint(path: str) -> bool:
    return any(
        os.path.isfile(os.path.join(path, name)) for name in CHECKPOINT_MARKERS
    )


def list_local_files(stage_dir: str, repo_id: str) -> Set[str]:
    root = os.path.join(os.path.expanduser(stage_dir), repo_id)
    if not os.path.isdir(root):
        return set()
    files: Set[str] = set()
    for dirpath, _, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        for name in filenames:
            files.add(name if rel_dir == "." else os.path.join(rel_dir, name))
    return files


def delete_staged_checkpoint(path: str, stage_dir: str) -> None:
    path = os.path.abspath(path)
    root = os.path.abspath(os.path.expanduser(stage_dir))
    if not path.startswith(root + os.sep):
        logger.warning(f"Refusing to delete outside stage dir: {path}")
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
        logger.info(f"Deleted staged checkpoint {path}")
    parent = os.path.dirname(path)
    while parent.startswith(root + os.sep) and parent != root:
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)


def _hub_download(
    repo_id: str,
    filename: str,
    local_dir: str,
    revision: str,
    token: Optional[str],
    retries: int = 5,
) -> str:
    last_err: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token,
                local_dir=local_dir,
                etag_timeout=60,
            )
        except Exception as exc:
            last_err = exc
            if _is_not_found(exc):
                raise
            logger.warning(
                f"Download failed {repo_id}/{filename} "
                f"(attempt {attempt}/{retries}): {_exception_chain(exc)}"
            )
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download {repo_id}/{filename}") from last_err


def download_checkpoint_to_stage(
    repo_id: str,
    subfolder: str,
    stage_dir: str,
    revision: str,
    token: Optional[str],
) -> str:
    """Download one checkpoint into ``{stage_dir}/{repo_id}/{subfolder}``."""
    dest = staged_checkpoint_dir(stage_dir, repo_id, subfolder)
    if is_local_checkpoint(dest):
        logger.info(f"Already staged: {dest}")
        return dest

    repo_root = os.path.join(os.path.expanduser(stage_dir), repo_id)
    os.makedirs(dest, exist_ok=True)
    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    logger.info(f"Downloading {repo_id}/{subfolder} -> {dest}")
    _hub_download(repo_id, f"{prefix}config.json", repo_root, revision, token)

    try:
        _hub_download(repo_id, f"{prefix}model.safetensors", repo_root, revision, token)
    except Exception as exc:
        if not _is_not_found(exc):
            raise
        logger.info(f"No model.safetensors; trying pytorch_model.bin")
        _hub_download(repo_id, f"{prefix}pytorch_model.bin", repo_root, revision, token)

    try:
        _hub_download(
            repo_id, f"{prefix}generation_config.json", repo_root, revision, token
        )
    except Exception:
        pass

    if not is_local_checkpoint(dest):
        raise FileNotFoundError(f"Download finished but no weights in {dest}")
    return dest


def load_local_model(model_path: str, device: torch.device) -> Any:
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    model.to(device)
    return model


def hf_fingerprint(repo_id: str, revision: str, subfolder: str) -> str:
    return f"hf:{repo_id}@{revision}:{subfolder}"


def run_one_directory(
    args: argparse.Namespace,
    repo_id: str,
    directory_path: str,
    files: Set[str],
) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = args.token
    revision = args.revision
    directory_path = directory_path.strip("/")
    if not subdir_exists(files, directory_path):
        logger.warning(
            f"Skipping missing subdir {repo_id}/{directory_path}"
        )
        return False

    seeds = ppl.parse_int_list(args.seeds)
    epochs = ppl.parse_int_list(args.epochs)
    checkpoints = resolve_hf_checkpoints(files, directory_path, epochs, seeds)
    if not checkpoints:
        logger.warning(
            f"Skipping {repo_id}/{directory_path}: no checkpoints for "
            f"epochs={epochs} seeds={seeds}"
        )
        return False

    if args.download_only:
        n_ok = 0
        for epoch, seed, subfolder in checkpoints:
            try:
                dest = download_checkpoint_to_stage(
                    repo_id, subfolder, args.stage_dir, revision, token
                )
                logger.info(f"Staged seed={seed} epoch={epoch} -> {dest}")
                n_ok += 1
            except Exception as exc:
                logger.warning(
                    f"Failed to stage {repo_id}/{subfolder}: {_exception_chain(exc)}"
                )
        return n_ok > 0

    run_name = args.run_name or run_name_for(repo_id, directory_path)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(
        DEFAULT_OUT_ROOT, run_name
    )
    os.makedirs(out_dir, exist_ok=True)

    json_paths = resolve_testsets(directory_path, args.dataset_dir, args.json_path)
    logger.info(
        f"Testsets for {repo_id}/{directory_path} "
        f"({'shifted' if path_is_shifted(directory_path) else 'regular'}): {json_paths}"
    )

    datasets: Dict[str, List[Dict[str, Any]]] = {}
    for jp in json_paths:
        examples = ppl.load_json_dataset(jp)
        if args.split:
            examples = [e for e in examples if e.get("split") == args.split]
        datasets[jp] = examples
        cats = sorted({ppl.normalize_category(e.get(args.category_key)) for e in examples})
        logger.info(f"Loaded {jp}: {len(examples)} examples; categories={cats}")

    tokenizer = load_local_tokenizer(args.tokenizer_path)

    overall_rows: List[Dict[str, Any]] = []
    category_rows: List[Dict[str, Any]] = []

    for epoch, seed, subfolder in checkpoints:
        ckpt_id = f"{repo_id}/{subfolder}" if subfolder else repo_id
        staged = staged_checkpoint_dir(args.stage_dir, repo_id, subfolder)
        logger.info(
            f"=== {run_name} | {repo_id} | {directory_path} | "
            f"seed={seed} | epoch={epoch} | {ckpt_id} ==="
        )
        model = None
        try:
            if not is_local_checkpoint(staged):
                if args.local_only:
                    logger.warning(f"Skipping missing local checkpoint: {staged}")
                    continue
                download_checkpoint_to_stage(
                    repo_id, subfolder, args.stage_dir, revision, token
                )
            model = load_local_model(staged, device)
        except Exception as exc:
            logger.warning(f"Skipping checkpoint {ckpt_id}: {_exception_chain(exc)}")
            if not args.keep_models:
                delete_staged_checkpoint(staged, args.stage_dir)
            continue
        fp = hf_fingerprint(repo_id, revision, subfolder)

        try:
            for jp, examples in datasets.items():
                testset = ppl._testset_name(jp)
                y_pred, ppl_g, ppl_c, nll_g, nll_c, cats, skipped = ppl.score_examples(
                    model,
                    tokenizer,
                    examples,
                    args.gold_key,
                    args.corrupt_key,
                    device,
                    args.max_length,
                    category_key=args.category_key,
                )
                results = ppl.summarize_predictions(
                    y_pred,
                    ppl_g,
                    ppl_c,
                    nll_g,
                    nll_c,
                    cats,
                    skipped,
                    fp,
                    bootstrap_n=args.bootstrap_n,
                    bootstrap_seed=args.bootstrap_seed,
                    bootstrap_alpha=args.bootstrap_alpha,
                )

                logger.info(
                    f"[{testset}] seed={seed} epoch={epoch} "
                    f"acc={100 * results['accuracy']:.2f}% "
                    f"n={results['n_examples']} "
                    f"mean_ppl_gold={results['mean_ppl_gold']:.4f}"
                )
                if results["by_category"]:
                    logger.info(
                        f"[{testset}] by category:\n"
                        f"{ppl.format_category_report(results['by_category'])}"
                    )

                overall_rows.append(
                    {
                        "run_name": run_name,
                        "hf_repo": repo_id,
                        "directory_path": directory_path,
                        "seed": seed,
                        "epoch": epoch,
                        "checkpoint": ckpt_id,
                        "testset": testset,
                        "json_path": jp,
                        "n_examples": results["n_examples"],
                        "n_skipped": results["n_skipped"],
                        "accuracy": results["accuracy"],
                        "precision_prefer_gold": results["precision_prefer_gold"],
                        "recall_prefer_gold": results["recall_prefer_gold"],
                        "f1_prefer_gold": results["f1_prefer_gold"],
                        "mean_ppl_gold": results["mean_ppl_gold"],
                        "mean_ppl_corrupt": results["mean_ppl_corrupt"],
                        "margin_nll_mean": results["margin_nll_mean"],
                        "margin_nll_median": results["margin_nll_median"],
                        "margin_nll_std": results["margin_nll_std"],
                        "fingerprint": fp,
                    }
                )

                for cat, stats in results["by_category"].items():
                    category_rows.append(
                        {
                            "run_name": run_name,
                            "hf_repo": repo_id,
                            "directory_path": directory_path,
                            "seed": seed,
                            "epoch": epoch,
                            "checkpoint": ckpt_id,
                            "testset": testset,
                            "json_path": jp,
                            "category": cat,
                            "n_examples": int(stats["n_examples"]),
                            "n_correct": int(stats["n_correct"]),
                            "accuracy": stats["accuracy"],
                            "mean_ppl_gold": stats["mean_ppl_gold"],
                            "mean_ppl_corrupt": stats["mean_ppl_corrupt"],
                            "mean_margin_nll": stats["mean_margin_nll"],
                            "fingerprint": fp,
                        }
                    )
        finally:
            ppl.unload_model(model)
            if not args.keep_models:
                delete_staged_checkpoint(staged, args.stage_dir)

    if not overall_rows:
        logger.warning(
            f"Skipping {repo_id}/{directory_path}: every checkpoint failed to load"
        )
        return False

    overall_fields = [
        "run_name",
        "hf_repo",
        "directory_path",
        "seed",
        "epoch",
        "checkpoint",
        "testset",
        "json_path",
        "n_examples",
        "n_skipped",
        "accuracy",
        "precision_prefer_gold",
        "recall_prefer_gold",
        "f1_prefer_gold",
        "mean_ppl_gold",
        "mean_ppl_corrupt",
        "margin_nll_mean",
        "margin_nll_median",
        "margin_nll_std",
        "fingerprint",
    ]
    category_fields = [
        "run_name",
        "hf_repo",
        "directory_path",
        "seed",
        "epoch",
        "checkpoint",
        "testset",
        "json_path",
        "category",
        "n_examples",
        "n_correct",
        "accuracy",
        "mean_ppl_gold",
        "mean_ppl_corrupt",
        "mean_margin_nll",
        "fingerprint",
    ]

    ppl.write_csv(
        os.path.join(out_dir, f"{run_name}_per_seed_overall.csv"),
        overall_rows,
        overall_fields,
    )
    ppl.write_csv(
        os.path.join(out_dir, f"{run_name}_per_seed_by_category.csv"),
        category_rows,
        category_fields,
    )

    agg_overall = ppl.aggregate_rows(
        overall_rows,
        group_keys=("run_name", "hf_repo", "directory_path", "epoch", "testset"),
        metric_keys=ppl.OVERALL_METRIC_KEYS,
    )
    agg_category = ppl.aggregate_rows(
        category_rows,
        group_keys=("run_name", "hf_repo", "directory_path", "epoch", "testset", "category"),
        metric_keys=ppl.CATEGORY_METRIC_KEYS,
    )

    agg_overall_fields = [
        "run_name",
        "hf_repo",
        "directory_path",
        "epoch",
        "testset",
        "n_seeds",
        "seeds",
        "n_examples",
        "accuracy_mean",
        "accuracy_std",
        "mean_ppl_gold_mean",
        "mean_ppl_gold_std",
        "mean_ppl_corrupt_mean",
        "mean_ppl_corrupt_std",
        "margin_nll_mean_mean",
        "margin_nll_mean_std",
        "f1_prefer_gold_mean",
        "f1_prefer_gold_std",
    ]
    agg_category_fields = [
        "run_name",
        "hf_repo",
        "directory_path",
        "epoch",
        "testset",
        "category",
        "n_seeds",
        "seeds",
        "n_examples",
        "accuracy_mean",
        "accuracy_std",
        "mean_ppl_gold_mean",
        "mean_ppl_gold_std",
        "mean_ppl_corrupt_mean",
        "mean_ppl_corrupt_std",
        "mean_margin_nll_mean",
        "mean_margin_nll_std",
    ]

    ppl.write_csv(
        os.path.join(out_dir, f"{run_name}_aggregate_overall.csv"),
        agg_overall,
        agg_overall_fields,
    )
    ppl.write_csv(
        os.path.join(out_dir, f"{run_name}_aggregate_by_category.csv"),
        agg_category,
        agg_category_fields,
    )
    logger.info(f"Done. CSV outputs in {out_dir}")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare PPL of gold vs corrupted sentences for Hugging Face "
            "checkpoints. Testsets default from whether --subdir contains "
            "'shifted'."
        )
    )
    p.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hub repo id, or comma-separated ids. Bare names get --hf_user prefix "
        "(e.g. morph-char-en-de -> ebrarkiziloglu/morph-char-en-de).",
    )
    p.add_argument(
        "--subdir",
        type=str,
        required=True,
        help="Directory path inside the repo, or comma-separated paths "
        "(e.g. phase_2/regular/de_babylm/shifted). "
        "If the path itself is a checkpoint, evaluate only that folder.",
    )
    p.add_argument(
        "--hf_user",
        type=str,
        default=DEFAULT_HF_USER,
        help="Prefix used when --repo_id has no slash.",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Git revision / branch / commit on the Hub.",
    )
    p.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hub token (default: HF_TOKEN / HUGGING_FACE_HUB_TOKEN env).",
    )
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help="Local tokenizer directory (must contain tokenizer.json / "
        "tokenizer_config.json). Default: models/v3.",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing the regular / shifted JSON testsets.",
    )
    p.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Override auto testset selection with comma-separated JSON paths.",
    )
    p.add_argument("--gold_key", type=str, default="gold_text")
    p.add_argument("--corrupt_key", type=str, default="corrupted")
    p.add_argument(
        "--category_key",
        type=str,
        default="category",
        help="JSON field used for per-category breakdown (missing -> NaN).",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        help="If set, only rows with this `split` value.",
    )
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="CSV / folder name prefix (default: {repo}/{subdir_underscored}).",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds under ep{{k}}/seed_{{s}}.",
    )
    p.add_argument(
        "--epochs",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated epoch indices matching ep{k} directories. "
        "Epochs that are absent under --subdir are skipped.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="CSV output directory (default: eval_results/{run_name}).",
    )
    p.add_argument(
        "--stage_dir",
        type=str,
        default=DEFAULT_STAGE_DIR,
        help="Local directory for downloaded checkpoints "
        "(default: /home/vault/b279bb/b279bb15/hf_stage).",
    )
    p.add_argument(
        "--download_only",
        action="store_true",
        help="Download checkpoints to --stage_dir and exit (run on a login node).",
    )
    p.add_argument(
        "--local_only",
        action="store_true",
        help="Do not download. Evaluate only checkpoints already in --stage_dir.",
    )
    p.add_argument(
        "--keep_models",
        action="store_true",
        help="Keep staged checkpoints after eval (default: delete each one).",
    )
    p.add_argument(
        "--bootstrap_n",
        type=int,
        default=0,
        help="Bootstrap resamples for CI (0 disables; recommended 0 for sweeps).",
    )
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--bootstrap_alpha", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.stage_dir = os.path.expanduser(args.stage_dir)
    os.makedirs(args.stage_dir, exist_ok=True)

    if args.download_only and args.local_only:
        raise SystemExit("Use only one of --download_only or --local_only")

    repo_ids = [
        normalize_repo_id(r, args.hf_user)
        for r in args.repo_id.split(",")
        if r.strip()
    ]
    subdirs = [s.strip() for s in args.subdir.split(",") if s.strip()]
    if not repo_ids or not subdirs:
        raise SystemExit("Need at least one --repo_id and one --subdir")

    n_ok = 0
    n_skip = 0
    api = None if args.local_only else HfApi(token=args.token)
    for repo_id in repo_ids:
        try:
            if args.local_only:
                files = list_local_files(args.stage_dir, repo_id)
                if not files:
                    logger.warning(
                        f"Skipping repo {repo_id}: nothing staged under "
                        f"{args.stage_dir}/{repo_id}"
                    )
                    n_skip += len(subdirs)
                    continue
            else:
                files = list_repo_files(api, repo_id, args.revision, args.token)
        except Exception as exc:
            logger.warning(
                f"Skipping repo {repo_id}: failed to list files "
                f"(revision={args.revision}): {exc}"
            )
            n_skip += len(subdirs)
            continue
        for directory_path in subdirs:
            if run_one_directory(args, repo_id, directory_path, files):
                n_ok += 1
            else:
                n_skip += 1
    action = "download" if args.download_only else "eval"
    logger.info(f"Finished {n_ok} repo/subdir {action}(s); skipped {n_skip}.")


if __name__ == "__main__":
    main()
