"""Create an artwork ImageFolder dataset from local <style>/<image> directories.

Local preparation requires only the standard library. For uploading install
huggingface_hub and authenticate with `hf auth login` or HF_TOKEN.

Examples:
    python src/prepare_hf_dataset.py --source data/dataset --output data/hf_dataset
    python src/prepare_hf_dataset.py --styles Impressionism Surrealism --max-per-class 400
    python src/prepare_hf_dataset.py --output data/hf_upload --upload --repo user/artworks

Load the result with:
    from datasets import load_dataset
    dataset = load_dataset("imagefolder", data_dir="data/hf_dataset")

Existing output is never overwritten. Images must already be downloaded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_STYLES = ["Impressionism", "Surrealism", "Minimalism", "Pop Art"]


def prepare_dataset(source, output, styles, max_per_class, seed, train_ratio, val_ratio):
    """Prepare splits atomically; reject exact duplicates with conflicting labels."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directories")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and train + validation must be < 1")
    if max_per_class < 0:
        raise ValueError("max-per-class must be >= 0 (0 means all images)")
    if not styles or len(set(styles)) != len(styles):
        raise ValueError("Supply unique style names")

    rng = random.Random(seed)
    seen = {}
    selections = {}
    for style in styles:
        if Path(style).name != style or style in {".", ".."}:
            raise ValueError(f"Invalid style name: {style}")
        directory = source / style
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing style directory: {directory}")
        images = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                if seen[digest] != style:
                    raise ValueError(f"Identical image has conflicting labels: {path}")
                continue
            seen[digest] = style
            images.append((path, digest))
        rng.shuffle(images)
        selected = images[:max_per_class] if max_per_class else images
        n_train, n_val = int(len(selected) * train_ratio), int(len(selected) * val_ratio)
        if min(n_train, n_val, len(selected) - n_train - n_val) < 1:
            raise ValueError(f"Not enough distinct images for three nonempty splits: {style}")
        selections[style] = {
            "train": selected[:n_train],
            "validation": selected[n_train:n_train + n_val],
            "test": selected[n_train + n_val:],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hf-dataset-", dir=output.parent))
    counts, manifest = {}, []
    try:
        for style, splits in selections.items():
            counts[style] = {}
            for split, images in splits.items():
                destination = staging / split / style
                destination.mkdir(parents=True)
                counts[style][split] = len(images)
                for path, digest in images:
                    shutil.copy2(path, destination / path.name)
                    manifest.append({"source": str(path.relative_to(source)),
                                     "file": f"{split}/{style}/{path.name}", "sha256": digest})
        metadata = {"seed": seed, "train_ratio": train_ratio, "validation_ratio": val_ratio,
                    "max_per_class": max_per_class, "counts": counts, "files": manifest}
        (staging / "split_manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        card = (
            "---\ntask_categories:\n- image-classification\ntags:\n- art\n- painting\n---\n\n"
            "# Artwork style classification\n\n"
            f"{len(styles)} styles: {', '.join(styles)}.\n\n"
            f"Split seed: {seed}. Train/validation/test ratios: "
            f"{train_ratio:g}/{val_ratio:g}/{1 - train_ratio - val_ratio:g}.\n\n"
            "Counts after sampling and exact-file deduplication:\n\n"
            "| Style | Train | Validation | Test |\n| --- | --- | --- | --- |\n"
            + "".join(f"| {style} | {c['train']} | {c['validation']} | {c['test']} |\n"
                      for style, c in counts.items())
            + "\n## Loading\n\n```python\nfrom datasets import load_dataset\n"
            'dataset = load_dataset("imagefolder", data_dir="path/to/dataset")\n```\n\n'
            "## Provenance and limitations\n\n"
            "Images were supplied locally. Record their source and applicable usage rights "
            "before publication; no image license is inferred by this script.\n\n"
            "Splits are random by image, not grouped by artist. Exact byte duplicates are "
            "removed, but resized copies and related works can still cross splits. "
            "This split does not measure generalization to unseen artists. "
            "Image extensions are filtered; image decoding is not validated.\n"
        )
        (staging / "README.md").write_text(card, encoding="utf-8")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/hf_dataset"))
    parser.add_argument("--styles", nargs="+", default=DEFAULT_STYLES)
    parser.add_argument("--max-per-class", type=int, default=400, help="0 = all images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--upload", action="store_true", help="Upload after local preparation")
    parser.add_argument("--repo", default=os.getenv("HF_DATASET_REPO"))
    parser.add_argument("--public", action="store_true", help="Create a public repository (default: private)")
    args = parser.parse_args()
    if args.upload:
        if not args.repo:
            parser.error("--upload requires --repo or HF_DATASET_REPO")
        try:
            from huggingface_hub import HfApi
        except ImportError:
            parser.error("Install upload dependency: pip install huggingface_hub")
    try:
        counts = prepare_dataset(args.source, args.output, args.styles, args.max_per_class,
                                 args.seed, args.train_ratio, args.val_ratio)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"Dataset prepared: {args.output}")
    if args.upload:
        api = HfApi()
        api.create_repo(repo_id=args.repo, repo_type="dataset", private=not args.public,
                        exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(args.output))
        print(f"Uploaded: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
