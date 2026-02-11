# Resources Catalog

**Project:** Uncertainty-Aware Medical LLMs: Quantifying Doubt in the Face of Counterfactuals
**Last Updated:** 2026-02-11

---

## Overview

This document catalogs all resources gathered during the resource-finding phase for the research project on uncertainty quantification in medical LLMs. The collection spans three categories:

- **25 papers** covering semantic entropy, conformal prediction, calibration, hallucination detection, and adversarial robustness in medical QA (stored in `papers/`)
- **9 tier-1 datasets** already downloaded for medical QA, hallucination detection, and counterfactual reasoning, with additional tier-2 and tier-3 datasets available via download script (stored in `datasets/downloaded/`)
- **8 code repositories** providing reference implementations for uncertainty estimation, conformal prediction, and medical hallucination benchmarks (cloned into `code/`)

Together these resources cover the full pipeline: uncertainty estimation methods, medical domain evaluation benchmarks, counterfactual/adversarial test sets, and calibration techniques.

---

## Papers

All papers are stored in `papers/`. Relevance tiers are defined as follows:

| Tier | Meaning |
|------|---------|
| **CORE** | Foundational uncertainty estimation methods central to the project |
| **MEDICAL** | Medical domain QA, hallucination, or clinical NLP |
| **COUNTERFACTUAL** | Adversarial robustness, counterfactual reasoning, or fragility analysis |
| **METHODOLOGY** | General calibration, confidence elicitation, or UQ techniques |

### Paper Inventory

| # | Filename | Title | Venue / Year | Citations | Tier | Notes |
|---|---------|-------|-------------|-----------|------|-------|
| 1 | `farquhar2024_semantic_entropy.pdf` | Detecting Hallucinations in LLMs Using Semantic Entropy | Nature, 2024 | 889 | CORE | **WARNING: PDF contains wrong paper** (STDP spiking neural network). Needs re-download. |
| 2 | `kuhn2023_semantic_uncertainty.pdf` | Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation | ICLR, 2023 | 525 | CORE | Core method paper. Defines semantic entropy over meaning clusters. |
| 3 | `wu2024_uncertainty_medical_qa.pdf` | Uncertainty Estimation for Medical QA | 2024 | -- | MEDICAL | Directly relevant to our topic. Applies UE to medical question answering. |
| 4 | `ness2024_medfuzz.pdf` | MedFuzz: Exploring the Robustness of LLMs in Medical QA | Microsoft Research, 2024 | -- | COUNTERFACTUAL | Adversarial fuzzing of medical LLM inputs. No public code available. |
| 5 | `wang2024_conu.pdf` | ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees | EMNLP, 2024 | -- | CORE | Conformal prediction framework for LLMs. Formal coverage guarantees. |
| 6 | `wang2024_word_sequence_entropy.pdf` | Word-Sequence Entropy: Towards Uncertainty Estimation in Free-Form Medical Question Answering Applications and Beyond | EAAI, 2025 | -- | MEDICAL | Medical QA uncertainty via word-sequence entropy. **WARNING: PDF may contain wrong paper.** No public code. |
| 7 | `zhao2025_afice.pdf` | AFICE: Robust Uncertainty Estimation via Bilateral Confidence Estimation and DPO Alignment | AAAI, 2025 | -- | COUNTERFACTUAL | Counterfactual resistance through bilateral confidence estimation and DPO. **WARNING: PDF may contain wrong paper.** |
| 8 | `kadavath2022_know_what_they_know.pdf` | Language Models (Mostly) Know What They Know | Anthropic, 2022 | -- | CORE | Foundational self-evaluation and P(True) methodology. |
| 9 | `lin2022_verbalized_uncertainty.pdf` | Teaching Models to Express Uncertainty in Words | 2022 | -- | METHODOLOGY | Verbalized confidence and linguistic uncertainty expressions. |
| 10 | `geng2023_calibration_survey.pdf` | A Survey of Confidence Estimation and Calibration in Large Language Models | 2023 | -- | METHODOLOGY | Comprehensive survey of calibration techniques for LLMs. |
| 11 | `tian2023_calibration.pdf` | Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models | 2023 | -- | METHODOLOGY | Prompting strategies for better-calibrated confidence outputs. |
| 12 | `xiong2023_confidence_elicitation.pdf` | Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs | 2023 | -- | METHODOLOGY | Empirical study of confidence elicitation approaches. |
| 13 | `fadeeva2023_lm_polygraph.pdf` | LM-Polygraph: Uncertainty Estimation in the Wild for LLM Applications | 2023 | -- | CORE | Framework paper for lm-polygraph library. 40+ UE methods. |
| 14 | `quach2023_conformal_lm.pdf` | Conformal Language Modeling | 2023 | -- | METHODOLOGY | Conformal prediction applied to language model outputs. |
| 15 | `kumar2023_cp_mcqa.pdf` | Conformal Prediction with Large Language Models for Multi-Choice Question Answering | 2023 | -- | METHODOLOGY | Conformal prediction specifically for MCQ format. Directly relevant to MedQA/MedMCQA evaluation. |
| 16 | `ren2023_knowno.pdf` | KnowNo: Knowing When You Don't Know -- LLM Calibration via Conformal Prediction | 2023 | -- | CORE | Conformal prediction for abstention. Knows when to say "I don't know." |
| 17 | `chen2024_inside.pdf` | INSIDE: LLM's Internal States for Hallucination Detection | 2024 | -- | CORE | Uses internal model states (hidden representations) for hallucination detection. |
| 18 | `kapoor2024_taught_uncertainty.pdf` | Large Language Models Can Be Taught to Express Uncertainty | 2024 | -- | METHODOLOGY | Training/fine-tuning LLMs to output calibrated uncertainty. |
| 19 | `ye2024_benchmarking_uq.pdf` | Benchmarking LLM Uncertainty Quantification Methods | 2024 | -- | METHODOLOGY | Systematic comparison of UQ methods across tasks. |
| 20 | `shorinwa2024_uq_survey.pdf` | A Survey on Uncertainty Quantification for Large Language Models | 2024 | -- | METHODOLOGY | Recent comprehensive survey of UQ for LLMs. |
| 21 | `su2024_api_conformal.pdf` | API-Based Conformal Prediction for Black-Box LLMs | 2024 | -- | METHODOLOGY | Conformal prediction without access to model internals. Relevant for API-only models. |
| 22 | `zeng2024_uncertainty_fragile.pdf` | Uncertainty in LLMs is Fragile | 2024 | -- | COUNTERFACTUAL | Demonstrates fragility of uncertainty estimates under perturbation. Directly relevant to counterfactual angle. |
| 23 | `hou2023_input_clarification.pdf` | Input Clarification for Better Understanding | 2023 | -- | METHODOLOGY | Clarification-seeking behavior in ambiguous inputs. |
| 24 | `umapathi2023_medhalt.pdf` | Med-HALT: Medical Domain Hallucination Test for LLMs | 2023 | -- | MEDICAL | Benchmark for medical hallucination detection. Provides our Med-HALT datasets. |
| 25 | `pandit2025_medhallu.pdf` | MedHallu: Hallucinations in Medical Reasoning of LLMs | 2025 | -- | MEDICAL | Latest hallucination detection benchmark for medical LLMs. |

### Papers by Relevance Tier

**CORE (6 papers):** Foundational methods -- semantic entropy, conformal prediction, internal state probing, lm-polygraph framework.
- farquhar2024, kuhn2023, wang2024_conu, kadavath2022, fadeeva2023, ren2023, chen2024

**MEDICAL (4 papers):** Domain-specific medical QA and hallucination detection.
- wu2024, wang2024_word_sequence_entropy, umapathi2023, pandit2025

**COUNTERFACTUAL (3 papers):** Adversarial robustness and uncertainty fragility.
- ness2024, zhao2025, zeng2024

**METHODOLOGY (12 papers):** Calibration, confidence elicitation, conformal prediction techniques, surveys.
- lin2022, geng2023, tian2023, xiong2023, quach2023, kumar2023, kapoor2024, ye2024, shorinwa2024, su2024, hou2023

---

## Datasets

### Downloaded Datasets (Tier 1)

All tier-1 datasets are pre-downloaded to `datasets/downloaded/`. Each directory contains the HuggingFace `datasets` library format plus a `sample.json` for quick inspection.

| # | Directory Name | Source | HuggingFace ID | Splits / Size | Description | Experiment Groups |
|---|---------------|--------|----------------|---------------|-------------|-------------------|
| 1 | `medqa` | MedQA USMLE | `GBaker/MedQA-USMLE-4-options-hf` | 10,178 train / 1,272 val / 1,273 test | 4-option MCQ from USMLE Steps 1-3. Gold standard medical QA benchmark. | uncertainty, multimedqa |
| 2 | `pubmedqa_labeled` | PubMedQA | `qiaojin/PubMedQA` (config: `pqa_labeled`) | 1,000 train | Expert-labeled yes/no/maybe questions from PubMed abstracts. | uncertainty, hallucination, multimedqa |
| 3 | `pubmedqa_artificial` | PubMedQA | `qiaojin/PubMedQA` (config: `pqa_artificial`) | 211,269 train | Auto-generated yes/no/maybe labels. Large-scale training set. | uncertainty, multimedqa |
| 4 | `medmcqa` | MedMCQA | `openlifescienceai/medmcqa` | 182,822 train / 4,183 val / 6,150 test | Indian medical entrance exam (AIIMS/NEET) MCQ. Four subjects. | uncertainty, multimedqa |
| 5 | `medhallu_labeled` | MedHallu | `UTAustin-AIHealth/MedHallu` (config: `pqa_labeled`) | 1,000 train | Expert-labeled medical hallucination detection pairs. | hallucination |
| 6 | `medhallu_artificial` | MedHallu | `UTAustin-AIHealth/MedHallu` (config: `pqa_artificial`) | 9,000 train | Auto-generated medical hallucination detection pairs. | hallucination |
| 7 | `medhalt_reasoning_fct` | Med-HALT | `openlifescienceai/Med-HALT` (config: `reasoning_FCT`) | 18,866 train | Fact-checking reasoning tasks for hallucination testing. | hallucination, counterfactual |
| 8 | `medhalt_reasoning_nota` | Med-HALT | `openlifescienceai/Med-HALT` (config: `reasoning_nota`) | 18,866 train | None-of-the-above reasoning tasks. Tests refusal under unanswerable questions. | hallucination, counterfactual |
| 9 | `medhalt_reasoning_fake` | Med-HALT | `openlifescienceai/Med-HALT` (config: `reasoning_fake`) | 1,858 train | Fake/nonsensical question detection. Tests counterfactual robustness. | hallucination, counterfactual |

### Additional Datasets (Available via Download Script)

These datasets are not pre-downloaded but can be obtained using `datasets/download_datasets.py`.

| # | Name | Tier | HuggingFace ID | Description | Experiment Groups |
|---|------|------|----------------|-------------|-------------------|
| 1 | `covid_qa` | 2 | `deepset/covid_qa_deepset` | COVID-QA extractive QA (2,019 pairs) | domain_specific |
| 2 | `healthsearchqa` | 2 | `katielink/healthsearchqa` | HealthSearchQA consumer health queries (3,173 questions) | open_ended, multimedqa |
| 3 | `liveqa` | 2 | `katielink/liveqa_trec2017` | LiveQA TREC 2017 consumer health QA (738 pairs) | consumer_health, multimedqa |
| 4 | `medicationqa` | 2 | GitHub only: [Medication_QA_MedInfo2019](https://github.com/abachaa/Medication_QA_MedInfo2019) | MedicationQA medication QA (674 pairs) | medication_safety, multimedqa |
| 5 | `triviaqa` | 3 | `mandarjoshi/trivia_qa` (config: `rc.nocontext`) | TriviaQA general knowledge QA (174K questions). Non-medical control dataset. | control, uncertainty |

---

## Code Repositories

All repositories are cloned into `code/`. Each provides reference implementations of methods or benchmarks used in this project.

| # | Directory | GitHub URL | Stars | License | Description | Key Components |
|---|-----------|-----------|-------|---------|-------------|----------------|
| 1 | `code/lm-polygraph/` | [IINemo/lm-polygraph](https://github.com/IINemo/lm-polygraph) | ~421 | MIT | **Primary UE framework.** 40+ uncertainty estimation methods including token entropy, semantic entropy, P(True), verbalized confidence, and more. | `pip install lm-polygraph`; supports HuggingFace models; extensible estimator API |
| 2 | `code/semantic_uncertainty/` | [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty) | ~403 | BSD-3 | Reference implementation of semantic entropy from Kuhn et al. (ICLR 2023). Clusters generations by semantic equivalence. | Supports BioASQ medical dataset; generation + clustering pipeline |
| 3 | `code/semantic-entropy-probes/` | [OATML/semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes) | -- | -- | Efficient single-generation alternative to full semantic entropy. Uses learned probes on hidden states. | Avoids costly multi-sample generation; probe training scripts |
| 4 | `code/Conformal-Uncertainty-Criterion/` | [Zhiyuan-GG/Conformal-Uncertainty-Criterion](https://github.com/Zhiyuan-GG/Conformal-Uncertainty-Criterion) | -- | -- | ConU implementation. Conformal prediction for LLMs with correctness coverage guarantees (EMNLP 2024). | Conformal calibration; prediction set construction |
| 5 | `code/afice/` | [zhaoy777/afice](https://github.com/zhaoy777/afice) | ~6 | -- | AFICE: bilateral confidence estimation + DPO alignment (AAAI 2025). Targets counterfactual robustness. | DPO training scripts; bilateral confidence scoring |
| 6 | `code/medhalt/` | [medhalt/medhalt](https://github.com/medhalt/medhalt) | ~29 | -- | Med-HALT benchmark evaluation code. Provides evaluation pipelines for medical hallucination testing. | Evaluation scripts for reasoning_FCT, reasoning_nota, reasoning_fake tasks |
| 7 | `code/MedHallu/` | [MedHallu/MedHallu](https://github.com/MedHallu/MedHallu) | ~14 | MIT | MedHallu hallucination detection benchmark. Pairs medical questions with hallucinated/factual answers. | Evaluation scripts; dataset loaders |
| 8 | `code/LLM-Uncertainty-Bench/` | [smartyfh/LLM-Uncertainty-Bench](https://github.com/smartyfh/LLM-Uncertainty-Bench) | ~259 | MIT | NeurIPS 2024 benchmark for LLM uncertainty quantification. Includes conformal prediction baselines. | UQ method implementations; benchmarking harness; conformal prediction |

### Notable Repositories Without Public Code

| Paper | Reason |
|-------|--------|
| MedFuzz (Microsoft Research, Ness et al. 2024) | No public code released. The MedFuzz adversarial fuzzing approach will need to be reimplemented based on the paper. |
| Word-Sequence Entropy (Wang et al. 2024) | No public code released. The word-sequence entropy method will need to be reimplemented. |

---

## Dataset Download Instructions

The download script at `datasets/download_datasets.py` supports flexible dataset retrieval from HuggingFace and GitHub.

### Prerequisites

```bash
pip install datasets pandas requests
```

### Usage

```bash
# List all available datasets with tiers and descriptions
python datasets/download_datasets.py --list

# Download all datasets (tier 1, 2, and 3)
python datasets/download_datasets.py --all

# Download a specific dataset by name
python datasets/download_datasets.py --dataset medqa

# Download all datasets of a given tier
python datasets/download_datasets.py --tier 1    # Critical (already downloaded)
python datasets/download_datasets.py --tier 2    # High priority
python datasets/download_datasets.py --tier 3    # Control/supporting

# Download by experiment group
python datasets/download_datasets.py --group uncertainty
python datasets/download_datasets.py --group hallucination
python datasets/download_datasets.py --group counterfactual
python datasets/download_datasets.py --group multimedqa

# Preview what would be downloaded without actually downloading
python datasets/download_datasets.py --dataset triviaqa --preview

# Specify a custom output directory
python datasets/download_datasets.py --tier 2 --output-dir /path/to/output
```

### Output Format

Each downloaded dataset is saved in HuggingFace `datasets` library format under `datasets/downloaded/<name>/`. A `sample.json` file is also created in each directory containing 3 example records from the first split for quick inspection.

### Loading a Downloaded Dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/downloaded/medqa")
print(dataset)
# DatasetDict({
#     train: Dataset({...}),
#     validation: Dataset({...}),
#     test: Dataset({...})
# })
```

---

## Notes

### Papers Requiring Re-download

Three PDFs may contain incorrect content and need to be verified and potentially re-downloaded:

1. **`farquhar2024_semantic_entropy.pdf`** (11MB) -- Confirmed to contain the wrong paper. The PDF contains a paper on STDP spiking neural networks instead of the Farquhar et al. Nature 2024 paper on semantic entropy for hallucination detection. This is a high-priority re-download since it is a core method paper.

2. **`wang2024_word_sequence_entropy.pdf`** (783KB) -- May contain the wrong paper. Needs verification. This paper describes the word-sequence entropy method for medical QA uncertainty.

3. **`zhao2025_afice.pdf`** (2.5MB) -- May contain the wrong paper. Needs verification. This paper describes the AFICE bilateral confidence estimation approach.

### Missing Public Code

- **MedFuzz** (Ness et al. 2024, Microsoft Research): No public repository. The adversarial fuzzing methodology for medical QA will need to be reimplemented from the paper description.
- **Word-Sequence Entropy** (Wang et al. 2024): No public repository. The entropy computation method will need to be reimplemented.

### Dataset Coverage by Experiment

| Experiment Area | Primary Datasets | Supporting Datasets |
|----------------|-----------------|---------------------|
| Uncertainty estimation on medical MCQ | medqa, medmcqa, pubmedqa_labeled | pubmedqa_artificial, triviaqa (control) |
| Hallucination detection | medhallu_labeled, medhallu_artificial | pubmedqa_labeled |
| Counterfactual robustness | medhalt_reasoning_fct, medhalt_reasoning_nota, medhalt_reasoning_fake | medhallu_labeled |
| Open-ended medical QA | pubmedqa_labeled | healthsearchqa, liveqa, covid_qa, medicationqa |

### Repository Dependency Notes

- **lm-polygraph** is installable via pip (`pip install lm-polygraph`) and supports HuggingFace model integration. This is the recommended primary framework for running uncertainty estimation experiments.
- **semantic_uncertainty** requires specific BioASQ data setup for medical experiments. See the repository README for configuration.
- **LLM-Uncertainty-Bench** includes conformal prediction baselines that may be directly usable for our evaluation pipeline.
- **afice** includes DPO training scripts that could be adapted for counterfactual robustness fine-tuning.
