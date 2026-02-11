#!/usr/bin/env python3
"""
Download script for all datasets used in the Uncertainty-Aware Medical LLMs project.

Usage:
    python download_datasets.py --all                    # Download everything
    python download_datasets.py --dataset medqa           # Download one dataset
    python download_datasets.py --tier 1                  # Download tier 1 (critical)
    python download_datasets.py --group uncertainty       # Download by experiment group
    python download_datasets.py --list                    # List all available datasets
    python download_datasets.py --dataset medqa --preview # Preview without downloading

Requirements:
    pip install datasets pandas requests
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------------------------

DATASETS = {
    # --- TIER 1: Critical ---
    "medqa": {
        "tier": 1,
        "hf_id": "GBaker/MedQA-USMLE-4-options-hf",
        "hf_config": None,
        "description": "MedQA USMLE 4-option MCQ (12,723 questions)",
        "groups": ["uncertainty", "multimedqa"],
    },
    "pubmedqa_labeled": {
        "tier": 1,
        "hf_id": "qiaojin/PubMedQA",
        "hf_config": "pqa_labeled",
        "description": "PubMedQA expert-labeled subset (1,000 questions)",
        "groups": ["uncertainty", "hallucination", "multimedqa"],
    },
    "pubmedqa_artificial": {
        "tier": 1,
        "hf_id": "qiaojin/PubMedQA",
        "hf_config": "pqa_artificial",
        "description": "PubMedQA auto-generated subset (211K questions)",
        "groups": ["uncertainty", "multimedqa"],
    },
    "medmcqa": {
        "tier": 1,
        "hf_id": "openlifescienceai/medmcqa",
        "hf_config": None,
        "description": "MedMCQA Indian medical exam MCQ (193K questions)",
        "groups": ["uncertainty", "multimedqa"],
    },
    "medhallu_labeled": {
        "tier": 1,
        "hf_id": "UTAustin-AIHealth/MedHallu",
        "hf_config": "pqa_labeled",
        "description": "MedHallu expert-labeled hallucination detection (1,000 pairs)",
        "groups": ["hallucination"],
    },
    "medhallu_artificial": {
        "tier": 1,
        "hf_id": "UTAustin-AIHealth/MedHallu",
        "hf_config": "pqa_artificial",
        "description": "MedHallu auto-generated hallucination detection (9,000 pairs)",
        "groups": ["hallucination"],
    },
    "medhalt_reasoning_fct": {
        "tier": 1,
        "hf_id": "openlifescienceai/Med-HALT",
        "hf_config": "reasoning_FCT",
        "description": "Med-HALT fact-checking reasoning (18.9K examples)",
        "groups": ["hallucination", "counterfactual"],
    },
    "medhalt_reasoning_nota": {
        "tier": 1,
        "hf_id": "openlifescienceai/Med-HALT",
        "hf_config": "reasoning_nota",
        "description": "Med-HALT none-of-the-above reasoning (18.9K examples)",
        "groups": ["hallucination", "counterfactual"],
    },
    "medhalt_reasoning_fake": {
        "tier": 1,
        "hf_id": "openlifescienceai/Med-HALT",
        "hf_config": "reasoning_fake",
        "description": "Med-HALT fake question reasoning (1.86K examples)",
        "groups": ["hallucination", "counterfactual"],
    },

    # --- TIER 2: High Priority ---
    "covid_qa": {
        "tier": 2,
        "hf_id": "deepset/covid_qa_deepset",
        "hf_config": None,
        "description": "COVID-QA extractive QA (2,019 pairs)",
        "groups": ["domain_specific"],
    },
    "healthsearchqa": {
        "tier": 2,
        "hf_id": "katielink/healthsearchqa",
        "hf_config": None,
        "description": "HealthSearchQA consumer health queries (3,173 questions)",
        "groups": ["open_ended", "multimedqa"],
    },
    "liveqa": {
        "tier": 2,
        "hf_id": "katielink/liveqa_trec2017",
        "hf_config": None,
        "description": "LiveQA TREC 2017 consumer health QA (738 pairs)",
        "groups": ["consumer_health", "multimedqa"],
    },

    # --- TIER 3: Control / Supporting ---
    "triviaqa": {
        "tier": 3,
        "hf_id": "mandarjoshi/trivia_qa",
        "hf_config": "rc.nocontext",
        "description": "TriviaQA general knowledge QA (174K questions, non-medical control)",
        "groups": ["control", "uncertainty"],
    },
}

# MedicationQA is GitHub-only (no HuggingFace dataset)
GITHUB_ONLY_DATASETS = {
    "medicationqa": {
        "tier": 2,
        "github_url": "https://github.com/abachaa/Medication_QA_MedInfo2019",
        "description": "MedicationQA medication QA (674 pairs, GitHub only)",
        "groups": ["medication_safety", "multimedqa"],
    },
}


def list_datasets():
    """Print a formatted table of all available datasets."""
    print("\n" + "=" * 90)
    print("AVAILABLE DATASETS")
    print("=" * 90)
    print(f"{'Name':<28} {'Tier':<6} {'Groups':<30} {'Description'}")
    print("-" * 90)
    for name, info in sorted(DATASETS.items(), key=lambda x: (x[1]["tier"], x[0])):
        groups = ", ".join(info["groups"])
        print(f"{name:<28} {info['tier']:<6} {groups:<30} {info['description']}")
    for name, info in sorted(GITHUB_ONLY_DATASETS.items()):
        groups = ", ".join(info["groups"])
        print(f"{name:<28} {info['tier']:<6} {groups:<30} {info['description']}")
    print("=" * 90)
    print(f"\nTotal: {len(DATASETS) + len(GITHUB_ONLY_DATASETS)} datasets")
    print("  HuggingFace: ", len(DATASETS))
    print("  GitHub-only: ", len(GITHUB_ONLY_DATASETS))


def download_hf_dataset(name, info, output_dir, preview=False):
    """Download a single HuggingFace dataset."""
    hf_id = info["hf_id"]
    hf_config = info.get("hf_config")

    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"HuggingFace ID: {hf_id}")
    if hf_config:
        print(f"Config: {hf_config}")
    print(f"Description: {info['description']}")
    print(f"{'=' * 60}")

    if preview:
        print("[PREVIEW MODE] Would download from HuggingFace.")
        print(f"  load_dataset(\"{hf_id}\"{', \"' + hf_config + '\"' if hf_config else ''})")
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    try:
        print(f"Loading dataset...")
        if hf_config:
            dataset = load_dataset(hf_id, hf_config)
        else:
            dataset = load_dataset(hf_id)

        print(f"Dataset loaded successfully!")
        print(f"Splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} examples")

        # Save to disk
        save_path = Path(output_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_path))
        print(f"Saved to: {save_path}")

        # Also save a small sample as JSON for inspection
        sample_path = save_path / "sample.json"
        for split_name, split_data in dataset.items():
            sample = [split_data[i] for i in range(min(3, len(split_data)))]
            with open(sample_path, "w") as f:
                json.dump({"split": split_name, "sample": sample}, f, indent=2, default=str)
            break  # Just save sample from first split
        print(f"Sample saved to: {sample_path}")

        return True

    except Exception as e:
        print(f"ERROR downloading {name}: {e}")
        return False


def download_github_dataset(name, info, output_dir, preview=False):
    """Download a GitHub-only dataset."""
    github_url = info["github_url"]
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name} (GitHub-only)")
    print(f"GitHub URL: {github_url}")
    print(f"Description: {info['description']}")
    print(f"{'=' * 60}")

    if preview:
        print(f"[PREVIEW MODE] Would clone from: {github_url}")
        return True

    save_path = Path(output_dir) / name
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        import subprocess
        clone_path = save_path / "repo"
        if clone_path.exists():
            print(f"Already cloned at: {clone_path}")
        else:
            print(f"Cloning repository...")
            subprocess.run(
                ["git", "clone", github_url, str(clone_path)],
                check=True, capture_output=True, text=True
            )
            print(f"Cloned to: {clone_path}")
        return True

    except Exception as e:
        print(f"ERROR cloning {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Uncertainty-Aware Medical LLMs project"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str, help="Download a specific dataset by name")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Download all datasets of a tier")
    parser.add_argument("--group", type=str, help="Download by experiment group")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--preview", action="store_true", help="Preview without downloading")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent / "downloaded"),
        help="Output directory for downloaded datasets"
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not any([args.all, args.dataset, args.tier, args.group]):
        parser.print_help()
        return

    # Determine which datasets to download
    to_download = {}
    to_download_github = {}

    if args.all:
        to_download = dict(DATASETS)
        to_download_github = dict(GITHUB_ONLY_DATASETS)
    elif args.dataset:
        name = args.dataset
        if name in DATASETS:
            to_download[name] = DATASETS[name]
        elif name in GITHUB_ONLY_DATASETS:
            to_download_github[name] = GITHUB_ONLY_DATASETS[name]
        else:
            print(f"ERROR: Unknown dataset '{name}'. Use --list to see available datasets.")
            sys.exit(1)
    elif args.tier:
        to_download = {k: v for k, v in DATASETS.items() if v["tier"] == args.tier}
        to_download_github = {k: v for k, v in GITHUB_ONLY_DATASETS.items() if v["tier"] == args.tier}
    elif args.group:
        to_download = {k: v for k, v in DATASETS.items() if args.group in v["groups"]}
        to_download_github = {k: v for k, v in GITHUB_ONLY_DATASETS.items() if args.group in v["groups"]}

    total = len(to_download) + len(to_download_github)
    if total == 0:
        print("No datasets matched the criteria.")
        return

    print(f"\nWill {'preview' if args.preview else 'download'} {total} dataset(s):")
    for name in sorted(to_download.keys()):
        print(f"  [HuggingFace] {name}")
    for name in sorted(to_download_github.keys()):
        print(f"  [GitHub]      {name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    successes = 0
    failures = 0

    for name, info in sorted(to_download.items()):
        if download_hf_dataset(name, info, output_dir, preview=args.preview):
            successes += 1
        else:
            failures += 1

    for name, info in sorted(to_download_github.items()):
        if download_github_dataset(name, info, output_dir, preview=args.preview):
            successes += 1
        else:
            failures += 1

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {successes} succeeded, {failures} failed out of {total}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
