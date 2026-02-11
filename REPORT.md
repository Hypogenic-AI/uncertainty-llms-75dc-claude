# Research Report: Uncertainty-Aware Medical LLMs — Quantifying Doubt in the Face of Counterfactuals

## Abstract

We investigate whether prompting large language models (LLMs) to express uncertainty can reduce confident, unsafe completions when presented with counterfactual or implausible medical evidence. Using two frontier models (GPT-4.1, Claude Sonnet 4.5) on three medical datasets (MedQA, Med-HALT reasoning_fake, Med-HALT reasoning_fct), we compare baseline prompting against uncertainty-augmented prompting across 1,200 API calls. Our core finding is striking: **uncertainty-augmented prompting dramatically reduces unsafe completions on counterfactual medical questions** — from 97% to 9% for GPT-4.1 and from 71% to 2% for Claude Sonnet 4.5 — while simultaneously raising cautiousness rates from near-zero to ~90%. Confidence-based AUROC for counterfactual detection improves from ~0.66 (baseline) to ~0.89 (uncertainty-augmented), substantially exceeding the prior best medical uncertainty estimation result of AUROC 0.58 (Wu et al., 2024). However, the intervention is not without cost: Claude Sonnet 4.5 shows a significant accuracy degradation on legitimate medical questions (93% → 66%), partly attributable to a 34% response format compliance failure under the uncertainty prompt. GPT-4.1 preserves accuracy perfectly (92% → 92%), demonstrating that the safety-accuracy tradeoff is model-dependent rather than inherent. Neither model shows meaningful improvement on the fact-checking dataset (Med-HALT FCT), suggesting that uncertainty prompting is specifically effective for detecting *implausible premises* rather than *subtle factual errors*.

---

## 1. Introduction

### 1.1 Motivation

Large language models are increasingly deployed in medical decision support, yet they exhibit a dangerous tendency to answer confidently regardless of input quality. When presented with absurd, counterfactual, or dangerous medical claims, current frontier models typically provide confident responses that engage with the false premise rather than flagging it as suspicious. This behavior poses a direct patient safety risk: a model that confidently explains a treatment for a fictional disease, or recommends a medication based on fabricated pharmacology, could cause real harm if its output is acted upon.

The practical question motivating this research is: **Can we teach LLMs to say "I'm not sure" when the evidence looks weird or unsafe?** We operationalize this through a simple, deployment-ready intervention: prompt engineering that explicitly instructs models to assess evidence quality before answering.

### 1.2 Research Gap

Our literature review (see `literature_review.md`) identifies three isolated research streams:

1. **Uncertainty estimation methods** (semantic entropy, verbalized confidence, conformal prediction) — validated primarily on general-domain QA, achieving AUROC 0.75–0.85 on tasks like TriviaQA but collapsing to ~0.50–0.58 on medical benchmarks (Wu et al., 2024).
2. **Adversarial medical robustness testing** (MedFuzz, Ness et al., 2024) — demonstrates accuracy drops under adversarial perturbation but does not measure uncertainty.
3. **Counterfactual resistance** (AFICE, Zhao et al., 2025) — trains models to resist opposing arguments via DPO alignment but is not tested in medical contexts.

**No existing work combines uncertainty-augmented prompting with counterfactual medical inputs to measure whether LLMs become appropriately cautious.** This is the gap we fill.

### 1.3 Research Question

Can prompting LLMs to express uncertainty reduce confident, unsafe completions when presented with counterfactual or implausible medical evidence?

### 1.4 Hypotheses

- **H1 (Cautiousness):** Uncertainty-augmented prompts will increase the rate of cautious responses ("I don't know" or low confidence) on counterfactual medical questions.
- **H2 (Unsafe Reduction):** Uncertainty-augmented prompts will reduce the rate of confident, incorrect answers on counterfactual inputs.
- **H3 (Calibration):** Uncertainty-augmented prompts will improve AUROC for distinguishing counterfactual from legitimate questions via self-reported confidence.
- **H4 (Accuracy Preservation):** Uncertainty prompts will not significantly reduce accuracy on legitimate medical questions.

---

## 2. Methods

### 2.1 Models

| Model | Provider | Access Method | Notes |
|-------|----------|---------------|-------|
| GPT-4.1 | OpenAI | Direct API | Frontier model, strong instruction following |
| Claude Sonnet 4.5 | Anthropic (via OpenRouter) | OpenAI-compatible API | Frontier model, strong reasoning |

### 2.2 Datasets

| Dataset | Source | N (sampled) | Type | Description |
|---------|--------|-------------|------|-------------|
| MedQA | USMLE-style MCQ | 100 | Legitimate | Standard medical board exam questions; establishes baseline accuracy and confidence |
| Med-HALT reasoning_fake | Umapathi et al. (2023) | 100 | Counterfactual | Absurd/counterfactual medical questions with fabricated premises; primary test condition |
| Med-HALT reasoning_fct | Umapathi et al. (2023) | 100 | Fact-check | Questions where a student's answer is incorrect and model must identify the correct answer |

All samples were randomly selected with seed=42 for reproducibility. Sample IDs are preserved in `results/samples.json`.

### 2.3 Experimental Conditions

**Baseline Prompt:**
Standard medical expert persona requesting a JSON response with answer selection and confidence rating (1–10 scale).

**Uncertainty-Augmented Prompt:**
Extended persona that explicitly instructs the model to:
1. **Assess evidence quality** — check for implausible, absurd, counterfactual, or dangerous claims
2. **Express appropriate uncertainty** — select "I do not know" when available, give low confidence (1–3) on suspicious inputs
3. **Only answer confidently** when the question is medically sound
4. Additionally report an `evidence_quality` field: normal | suspicious | implausible | dangerous

Both prompts request structured JSON output with answer_index, answer_text, confidence (1–10), and reasoning.

### 2.4 Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **Cautiousness Rate** | % of responses where model says "I don't know" OR confidence ≤ 3 | Measures willingness to express uncertainty |
| **Unsafe Completion Rate** | % of responses with confidence ≥ 7 AND incorrect answer (or confident non-IDK on counterfactual) | Measures dangerous overconfidence |
| **Accuracy** | % correct on MedQA and FCT (not applicable for fake questions) | Measures knowledge preservation |
| **AUROC** | Area under ROC curve using inverted confidence to discriminate legitimate vs. counterfactual questions | Measures confidence calibration |
| **Evidence Flag Rate** | % of uncertainty-condition responses flagging evidence as suspicious/implausible/dangerous | Measures explicit evidence quality assessment |

### 2.5 Statistical Tests

- **Chi-squared test** for comparing cautiousness and unsafe rates between conditions
- **Independent t-test** for confidence score comparisons
- **Cohen's d** for effect size on confidence differences
- **Odds ratio** for cautiousness rate effect size
- **Significance level:** α = 0.05

### 2.6 Implementation Details

- Temperature: 0.3 (low, for reproducibility)
- Max tokens: 500
- Retry logic: exponential backoff, max 5 retries per call
- Response caching: SHA-256 hash-based to enable re-running without duplicate API calls
- Total API calls: 1,200 (2 models × 2 conditions × 3 datasets × 100 samples)
- Parse success rate: 94.0% overall (see Section 3.6 for per-condition breakdown)

---

## 3. Results

### 3.1 Primary Finding: Dramatic Reduction in Unsafe Completions on Counterfactual Questions

The most striking result is on the Med-HALT fake (counterfactual) dataset:

| Model | Condition | Cautiousness | Unsafe Rate | Mean Confidence | Evidence Flagged |
|-------|-----------|-------------|-------------|-----------------|------------------|
| GPT-4.1 | Baseline | 3.0% | **97.0%** | 9.5 ± 1.0 | — |
| GPT-4.1 | Uncertainty | 91.0% | **9.0%** | 3.5 ± 3.5 | 93.0% |
| Claude Sonnet 4.5 | Baseline | 22.0% | **71.0%** | 8.0 ± 1.8 | — |
| Claude Sonnet 4.5 | Uncertainty | 88.0% | **2.0%** | 1.8 ± 1.1 | 90.0% |

**Key observations:**
- **GPT-4.1 baseline is alarming:** 97% of responses to counterfactual medical questions were *confident and non-IDK* — the model confidently engaged with absurd premises in virtually every case.
- **Uncertainty prompting is transformative:** Both models shift from predominantly unsafe to predominantly cautious. GPT-4.1's unsafe rate drops 88 percentage points (97% → 9%); Claude's drops 69 percentage points (71% → 2%).
- **Both models successfully flag evidence quality:** ~90% of uncertainty-condition responses correctly identify the counterfactual questions as suspicious/implausible/dangerous.
- **Claude has higher baseline cautiousness:** Even without the uncertainty prompt, Claude flags 22% of counterfactual questions as suspicious, vs. only 3% for GPT-4.1. This suggests some inherent caution in Claude's default behavior.

Statistical significance for all counterfactual results: p < 0.0001 (chi-squared tests for both cautiousness and unsafe rate comparisons, both models).

### 3.2 AUROC: Counterfactual Detection via Confidence

Using self-reported confidence as a counterfactual detector (lower confidence → more likely counterfactual):

| Model | Condition | AUROC |
|-------|-----------|-------|
| GPT-4.1 | Baseline | 0.667 |
| GPT-4.1 | Uncertainty | **0.879** |
| Claude Sonnet 4.5 | Baseline | 0.660 |
| Claude Sonnet 4.5 | Uncertainty | **0.892** |

**Context:** Wu et al. (2024) reported the best prior AUROC for medical uncertainty estimation at 0.58 (Two-Phase Verification method). Our uncertainty-augmented prompting achieves AUROC of 0.88–0.89, a substantial improvement. However, a direct comparison is imperfect: our task (distinguishing completely fabricated questions from legitimate ones) is arguably easier than the fine-grained uncertainty estimation evaluated by Wu et al. (distinguishing correct from incorrect answers on legitimate medical questions).

The baseline AUROC of ~0.66 indicates that even without explicit uncertainty instructions, both models give *slightly* lower confidence on counterfactual questions — but not nearly enough to be useful as a safety mechanism.

### 3.3 Accuracy Preservation (H4): Model-Dependent Results

| Model | Condition | MedQA Accuracy | FCT Accuracy |
|-------|-----------|----------------|-------------|
| GPT-4.1 | Baseline | 92.0% | 89.0% |
| GPT-4.1 | Uncertainty | **92.0%** | **88.0%** |
| Claude Sonnet 4.5 | Baseline | 93.0% | 82.0% |
| Claude Sonnet 4.5 | Uncertainty | **66.0%** | **75.0%** |

**GPT-4.1** preserves accuracy perfectly: 92% → 92% on MedQA, 89% → 88% on FCT. The uncertainty prompt does not cause GPT-4.1 to second-guess valid medical knowledge.

**Claude Sonnet 4.5** shows a significant accuracy drop: 93% → 66% on MedQA (p = 0.001 for cautiousness increase). This 27-percentage-point drop is problematic and exceeds the 5% threshold we pre-specified for acceptable degradation.

**Critical confound:** Claude's accuracy drop is substantially attributable to a **34% parse failure rate** on MedQA under the uncertainty prompt (vs. 4% at baseline). When Claude receives the more complex uncertainty prompt, it frequently produces verbose narrative responses instead of the requested JSON format, particularly on legitimate questions where it has a strong answer but the prompt's instructions about evidence assessment create conflicting generation pressures. Parse failures are counted as incorrect, inflating the apparent accuracy drop. GPT-4.1 maintains 0% parse error across all conditions.

### 3.4 Fact-Checking Dataset: Limited Impact

On the Med-HALT FCT (fact-checking) dataset, uncertainty prompting shows **no meaningful improvement**:

| Model | Condition | Cautiousness | Unsafe Rate | Accuracy |
|-------|-----------|-------------|-------------|----------|
| GPT-4.1 | Baseline | 0.0% | 11.0% | 89.0% |
| GPT-4.1 | Uncertainty | 1.0% | 12.0% | 88.0% |
| Claude Sonnet 4.5 | Baseline | 3.0% | 15.0% | 82.0% |
| Claude Sonnet 4.5 | Uncertainty | 9.0% | 18.0% | 75.0% |

None of the FCT differences reach statistical significance. This is interpretable: FCT questions present *real* medical questions where a student gave a wrong answer — the premise is legitimate, the task is to identify the correct answer. The uncertainty prompt is designed to detect *implausible premises*, not *subtle factual errors*. Models appropriately maintain high confidence on these questions because the evidence *is* medically sound; the challenge is identifying which answer is correct, not whether the question makes sense.

### 3.5 Confidence Distributions

The violin plots reveal the mechanism of the intervention clearly:

- **Baseline condition:** Both models cluster at high confidence (8–10) across all datasets, including counterfactual questions. GPT-4.1 is especially extreme, with mean confidence of 9.96/10 on MedQA and 9.47/10 even on counterfactual questions.
- **Uncertainty condition on counterfactual questions:** Confidence distributions shift dramatically downward (mean 1.8 for Claude, 3.5 for GPT-4.1), creating clear bimodal separation from legitimate questions.
- **Uncertainty condition on legitimate/FCT questions:** Confidence remains high (8.0–9.6), confirming that the prompt doesn't indiscriminately suppress confidence.

### 3.6 Parse Success Rates

| Model | Condition | MedQA | Med-HALT Fake | Med-HALT FCT |
|-------|-----------|-------|---------------|-------------|
| GPT-4.1 | Baseline | 100% | 100% | 100% |
| GPT-4.1 | Uncertainty | 100% | 100% | 100% |
| Claude Sonnet 4.5 | Baseline | 96% | 91% | 97% |
| Claude Sonnet 4.5 | Uncertainty | **66%** | 90% | 88% |

GPT-4.1 achieves perfect JSON format compliance across all conditions. Claude Sonnet 4.5 shows substantially degraded format compliance under the uncertainty prompt, particularly on MedQA (66%). This is an important practical finding: more complex system prompts with multi-step reasoning instructions can degrade structured output compliance in some models.

---

## 4. Statistical Analysis

### 4.1 Hypothesis Testing Summary

| Hypothesis | GPT-4.1 | Claude Sonnet 4.5 | Verdict |
|-----------|---------|-------------------|---------|
| **H1 (Cautiousness on counterfactual)** | 3% → 91%, p < 0.0001, OR = 327 | 22% → 88%, p < 0.0001, OR = 26 | **Strongly supported** |
| **H2 (Unsafe reduction on counterfactual)** | 97% → 9%, p < 0.0001 | 71% → 2%, p < 0.0001 | **Strongly supported** |
| **H3 (AUROC improvement)** | 0.667 → 0.879 | 0.660 → 0.892 | **Strongly supported** |
| **H4 (Accuracy preservation)** | 92% → 92% (preserved) | 93% → 66% (degraded, partly due to parse errors) | **Mixed: model-dependent** |

### 4.2 Effect Sizes

The effect sizes on the primary outcome (counterfactual questions) are extraordinarily large:

- **GPT-4.1 confidence on counterfactual:** Cohen's d = 2.31 (very large; > 0.8 is conventionally "large")
- **Claude Sonnet 4.5 confidence on counterfactual:** Cohen's d = 4.19 (extremely large)
- **GPT-4.1 cautiousness odds ratio on counterfactual:** OR = 326.93
- **Claude Sonnet 4.5 cautiousness odds ratio on counterfactual:** OR = 26.00

These are among the largest effect sizes one could observe in a prompting intervention study, indicating that the uncertainty prompt fundamentally changes model behavior on counterfactual inputs.

### 4.3 Non-Significant Results

- **FCT dataset:** No significant differences for either model on cautiousness, unsafe rate, or accuracy (all p > 0.05)
- **GPT-4.1 on MedQA:** No significant change in cautiousness (p = 1.0) or unsafe rate (p = 1.0); small confidence decrease (d = 0.50, p = 0.0005) that does not affect accuracy

---

## 5. Discussion

### 5.1 The Core Insight: Prompt-Based Safety Guardrails Work for Absurd Inputs

The central finding is both simple and powerful: **explicitly asking LLMs to assess evidence quality before answering causes them to correctly flag implausible medical questions ~90% of the time**, compared to ~3-22% at baseline. This is a zero-cost intervention (no fine-tuning, no additional infrastructure) that could be deployed immediately in medical LLM applications.

The mechanism is clear from the data: the uncertainty prompt activates the model's existing ability to recognize implausible premises — an ability that is suppressed by standard task-focused prompts. Both models already "know" that counterfactual questions are absurd (as evidenced by slightly lower baseline confidence), but they default to answering confidently because that's what the standard prompt asks for.

### 5.2 The Specificity of the Effect: Implausible Premises vs. Subtle Errors

A critical nuance is that the intervention works specifically for **obviously implausible premises** (Med-HALT fake) but not for **subtle factual errors** (Med-HALT FCT). This makes sense: the uncertainty prompt instructs models to look for "implausible, absurd, counterfactual" claims, which is exactly what fake questions contain. FCT questions have legitimate premises but require careful reasoning to identify the correct answer — a task that benefits from knowledge, not from uncertainty.

This specificity is both a strength and a limitation. It means the intervention can be a reliable guardrail against grossly implausible inputs (e.g., questions about fictional diseases or impossible physiological claims) but should not be expected to improve performance on tasks requiring nuanced medical reasoning.

### 5.3 The Accuracy-Safety Tradeoff Is Model-Dependent

The divergent accuracy results between GPT-4.1 (no degradation) and Claude Sonnet 4.5 (significant degradation) reveal that the accuracy-safety tradeoff is **not inherent to the intervention but depends on model characteristics**:

- **GPT-4.1** exhibits strong instruction following: it applies the uncertainty assessment to evidence quality as instructed, determines that legitimate MedQA questions are sound, and proceeds to answer with high confidence and maintained accuracy. It perfectly compartmentalizes the two tasks (evidence assessment + answer selection).
- **Claude Sonnet 4.5** appears to struggle with the multi-faceted prompt, producing verbose narrative responses instead of structured JSON on legitimate questions (34% parse failure). This suggests that the more complex prompt creates competing generation pressures that degrade output format compliance, which in turn manifests as apparent accuracy loss.

**Practical implication:** Uncertainty-augmented prompts should be designed with attention to model-specific instruction-following characteristics. For models prone to format degradation under complex prompts, techniques like structured output constraints (e.g., JSON mode, function calling) should be used alongside the uncertainty instructions.

### 5.4 Comparison with Prior Work

| Method | Context | AUROC |
|--------|---------|-------|
| Token probability methods (Wu et al., 2024) | Medical QA, correct vs. incorrect | ~0.50 |
| Two-Phase Verification (Wu et al., 2024) | Medical QA, correct vs. incorrect | 0.58 |
| Semantic Entropy (Farquhar et al., 2024) | General QA, confabulation detection | 0.75–0.85 |
| **Our baseline prompting** | Medical, legitimate vs. counterfactual | **0.66** |
| **Our uncertainty prompting** | Medical, legitimate vs. counterfactual | **0.88–0.89** |

Our AUROC results exceed prior medical UE methods by a wide margin, but the comparison is not fully apples-to-apples. Our task (distinguishing fabricated questions from real ones) is arguably a coarser discrimination than identifying which answers to real questions are correct vs. incorrect. The high AUROC reflects the success of the intervention for a specific safety use case rather than a general advance in medical uncertainty estimation.

### 5.5 Limitations

1. **Sample size:** 100 questions per dataset is sufficient for detecting the large effects observed but may miss smaller effects. The non-significant FCT results could reflect either a true null effect or insufficient power.

2. **Dataset representativeness:** Med-HALT fake questions may be more obviously absurd than real-world misinformation. Sophisticated medical misinformation (plausible but wrong) would be harder to detect.

3. **Single-turn evaluation:** We test single questions in isolation. In real clinical workflows, counterfactual information may be embedded in longer patient histories or appear gradually through conversation.

4. **Prompt sensitivity:** We tested one uncertainty prompt design. The optimal formulation likely varies by model and may require task-specific tuning. The Claude format compliance issue suggests prompt robustness is a practical concern.

5. **Parse error confound:** Claude Sonnet 4.5's apparent accuracy drop is partially attributable to format compliance failures rather than genuine knowledge degradation. A fairer evaluation would use structured output constraints.

6. **No fine-tuning comparison:** We compare only prompt-based interventions. Fine-tuned approaches (e.g., AFICE-style DPO alignment) may achieve better accuracy preservation.

7. **Temperature sensitivity:** We used temperature 0.3 for all conditions. Higher temperatures might produce more calibrated confidence distributions; lower temperatures might reduce Claude's format compliance issues.

---

## 6. Conclusions

### 6.1 Summary of Findings

1. **Uncertainty-augmented prompting dramatically reduces unsafe completions on counterfactual medical questions** (97% → 9% for GPT-4.1; 71% → 2% for Claude), with overwhelming statistical significance (p < 0.0001) and very large effect sizes (Cohen's d = 2.3–4.2).

2. **Self-reported confidence under uncertainty prompting is an effective counterfactual detector**, achieving AUROC 0.88–0.89, substantially above the prior best medical UE result of 0.58.

3. **The accuracy-safety tradeoff is model-dependent:** GPT-4.1 preserves accuracy perfectly while gaining safety; Claude Sonnet 4.5 shows accuracy degradation partly attributable to format compliance failures under complex prompts.

4. **The intervention is specific to implausible premises** and does not improve performance on subtle factual errors (Med-HALT FCT).

5. **Even baseline frontier models are dangerously overconfident on counterfactual inputs:** GPT-4.1 confidently answered 97% of absurd medical questions without hesitation.

### 6.2 Practical Recommendations

- **Deploy uncertainty-augmented prompts as a first-line safety layer** in medical LLM applications, especially for detecting grossly implausible inputs.
- **Use structured output constraints** (JSON mode, function calling) alongside uncertainty prompts to prevent format compliance degradation.
- **Do not rely on uncertainty prompting alone** for detecting subtle medical errors — it is a complement to, not a replacement for, medical knowledge verification.
- **Test prompt interventions per-model** — the same prompt can have very different effects on different models.
- **Consider hybrid approaches** combining prompt-based uncertainty with probability-based methods (semantic entropy, conformal prediction) for defense in depth.

### 6.3 Future Work

- Test with more sophisticated medical misinformation (plausible but subtly wrong premises)
- Evaluate prompt robustness across different uncertainty prompt formulations
- Compare prompt-based intervention with fine-tuning approaches (AFICE, DPO)
- Test in multi-turn conversational settings where misinformation accumulates gradually
- Investigate whether structured output modes (e.g., OpenAI's JSON mode) resolve the format compliance issues observed with Claude

---

## 7. Reproducibility

All code, data, and results are available in this repository:

- **Experiment code:** `src/experiment.py`
- **Analysis code:** `src/analyze.py`
- **Raw results:** `results/raw_results.json` (1,200 responses)
- **Computed metrics:** `results/metrics.csv`
- **Statistical tests:** `results/statistical_tests.json`
- **AUROC results:** `results/auroc_results.json`
- **Plots:** `results/plots/` (5 visualizations)
- **Sample IDs:** `results/samples.json`
- **Experiment config:** `results/config.json`
- **Literature review:** `literature_review.md`
- **Research plan:** `planning.md`

**Environment:** Python 3.12.8, openai, httpx, datasets, numpy, scipy, matplotlib, seaborn, pandas, scikit-learn. Full dependency list in `pyproject.toml`.

**API costs:** ~$5–15 total for 1,200 API calls.

---

## Appendix A: Full Results Tables

### A.1 Summary Metrics

| Model | Dataset | Condition | Cautiousness | Unsafe Rate | Confidence (mean ± std) | Accuracy | Parse Rate | Evidence Flagged |
|-------|---------|-----------|-------------|-------------|------------------------|----------|------------|-----------------|
| GPT-4.1 | MedQA | Baseline | 0.0% | 8.0% | 10.0 ± 0.2 | 92.0% | 100% | — |
| GPT-4.1 | MedQA | Uncertainty | 1.0% | 7.0% | 9.6 ± 0.9 | 92.0% | 100% | 0.0% |
| GPT-4.1 | Med-HALT Fake | Baseline | 3.0% | 97.0% | 9.5 ± 1.0 | — | 100% | — |
| GPT-4.1 | Med-HALT Fake | Uncertainty | 91.0% | 9.0% | 3.5 ± 3.5 | — | 100% | 93.0% |
| GPT-4.1 | Med-HALT FCT | Baseline | 0.0% | 11.0% | 9.7 ± 0.5 | 89.0% | 100% | — |
| GPT-4.1 | Med-HALT FCT | Uncertainty | 1.0% | 12.0% | 9.3 ± 1.1 | 88.0% | 100% | 1.0% |
| Claude 4.5 | MedQA | Baseline | 0.0% | 6.0% | 8.9 ± 0.7 | 93.0% | 96% | — |
| Claude 4.5 | MedQA | Uncertainty | 12.0% | 19.0% | 8.1 ± 2.7 | 66.0% | 66% | 0.0% |
| Claude 4.5 | Med-HALT Fake | Baseline | 22.0% | 71.0% | 8.0 ± 1.8 | — | 91% | — |
| Claude 4.5 | Med-HALT Fake | Uncertainty | 88.0% | 2.0% | 1.8 ± 1.1 | — | 90% | 90.0% |
| Claude 4.5 | Med-HALT FCT | Baseline | 3.0% | 15.0% | 8.8 ± 1.3 | 82.0% | 97% | — |
| Claude 4.5 | Med-HALT FCT | Uncertainty | 9.0% | 18.0% | 8.4 ± 2.3 | 75.0% | 88% | 4.0% |

### A.2 Statistical Tests

| Model | Dataset | Test | Baseline | Uncertainty | p-value | Effect Size |
|-------|---------|------|----------|-------------|---------|-------------|
| GPT-4.1 | Med-HALT Fake | Cautiousness (χ²) | 3.0% | 91.0% | < 0.0001 | OR = 326.9 |
| GPT-4.1 | Med-HALT Fake | Unsafe Rate (χ²) | 97.0% | 9.0% | < 0.0001 | — |
| GPT-4.1 | Med-HALT Fake | Confidence (t-test) | 9.5 | 3.5 | < 0.0001 | d = 2.31 |
| GPT-4.1 | MedQA | Confidence (t-test) | 10.0 | 9.6 | 0.0005 | d = 0.50 |
| Claude 4.5 | Med-HALT Fake | Cautiousness (χ²) | 22.0% | 88.0% | < 0.0001 | OR = 26.0 |
| Claude 4.5 | Med-HALT Fake | Unsafe Rate (χ²) | 71.0% | 2.0% | < 0.0001 | — |
| Claude 4.5 | Med-HALT Fake | Confidence (t-test) | 8.0 | 1.8 | < 0.0001 | d = 4.19 |
| Claude 4.5 | MedQA | Cautiousness (χ²) | 0.0% | 12.0% | 0.0011 | — |
| Claude 4.5 | MedQA | Unsafe Rate (χ²) | 6.0% | 19.0% | 0.0103 | — |
| Claude 4.5 | Med-HALT FCT | Cautiousness (χ²) | 3.0% | 9.0% | 0.137 | OR = 3.20 |

### A.3 AUROC for Counterfactual Detection

| Model | Condition | AUROC | N (Legitimate) | N (Counterfactual) |
|-------|-----------|-------|----------------|-------------------|
| GPT-4.1 | Baseline | 0.667 | 100 | 100 |
| GPT-4.1 | Uncertainty | 0.879 | 100 | 100 |
| Claude Sonnet 4.5 | Baseline | 0.660 | 99 | 94 |
| Claude Sonnet 4.5 | Uncertainty | 0.892 | 97 | 90 |

---

## References

- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature*, 630, 625–630.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*.
- Ness, R., et al. (2024). MedFuzz: Exploring the Robustness of Large Language Models in Medical Question Answering. *Microsoft Research*.
- Umapathi, L. K., et al. (2023). Med-HALT: Medical Domain Hallucination Test for Large Language Models. *CoNLL 2023*.
- Wang, Z., et al. (2024). Word-Sequence Entropy: Towards Uncertainty Estimation in Free-Form Medical Question Answering. *EAAI 2025*.
- Wu, Y., et al. (2024). Uncertainty Estimation of Large Language Models in Medical Question Answering. *arXiv preprint*.
- Zhao, L., et al. (2025). AFICE: Aligning for Faithful Integrity with Confidence Estimation. *AAAI 2025*.
