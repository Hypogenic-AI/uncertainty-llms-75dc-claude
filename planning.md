# Research Plan: Uncertainty-Aware Medical LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used in medical settings, yet they often answer confidently even when presented with implausible, dangerous, or counterfactual medical claims. A model that says "I'm not sure" when evidence looks suspicious is far safer than one that confidently repeats dangerous misinformation. This research directly addresses patient safety by evaluating whether uncertainty estimation can serve as a guardrail against unsafe medical completions.

### Gap in Existing Work
The literature review reveals three isolated research streams: (1) uncertainty estimation methods (semantic entropy, verbalized confidence) validated mainly on general QA, (2) adversarial medical robustness testing (MedFuzz) that measures accuracy drops but not uncertainty, and (3) counterfactual resistance (AFICE) not tested in medical contexts. **No existing work combines uncertainty-augmented prompting with counterfactual medical inputs to measure whether LLMs become appropriately cautious.** Wu et al. (2024) showed medical UE is much harder than general UE (AUROC ~0.58 vs ~0.85), but didn't test whether prompting for uncertainty can reduce unsafe completions.

### Our Novel Contribution
We test whether **prompting LLMs to express uncertainty** (via structured uncertainty-aware prompts) causes them to (a) produce higher uncertainty signals on counterfactual/fake medical inputs vs. legitimate ones, and (b) reduce the rate of confident unsafe completions. We use real LLM APIs (GPT-4.1, Claude Sonnet 4.5) on Med-HALT counterfactual datasets, comparing baseline vs. uncertainty-augmented prompting.

### Experiment Justification
- **Experiment 1 (Baseline Medical QA):** Establish baseline accuracy and confidence calibration on standard MedQA questions. Needed to measure the "normal" confidence level.
- **Experiment 2 (Counterfactual Detection):** Test whether baseline and uncertainty-augmented LLMs differ in their response to Med-HALT fake/counterfactual questions. This is the core hypothesis test.
- **Experiment 3 (Fact-Checking with Uncertainty):** Test on Med-HALT FCT dataset where models must identify incorrect answers. Measures whether uncertainty prompting helps models flag wrong information.

## Research Question
Can prompting LLMs to express uncertainty reduce confident, unsafe completions when presented with counterfactual or implausible medical evidence?

## Background and Motivation
Medical LLMs that answer confidently regardless of input quality pose patient safety risks. The user's instruction is clear: "Teach LLMs to say 'I'm not sure' when the evidence looks weird or unsafe." We operationalize this by comparing baseline prompting vs. uncertainty-augmented prompting on medical counterfactual datasets.

## Hypothesis Decomposition
1. **H1 (Cautiousness):** Uncertainty-augmented prompts will increase the rate of "I don't know" / uncertain responses on counterfactual medical questions compared to baseline prompts.
2. **H2 (Unsafe Reduction):** Uncertainty-augmented prompts will reduce the rate of confident, incorrect answers on counterfactual inputs.
3. **H3 (Calibration):** Models with uncertainty prompts will show better separation between confidence on legitimate vs. counterfactual questions (higher AUROC for counterfactual detection via self-reported confidence).
4. **H4 (Accuracy Preservation):** Uncertainty prompts will not significantly reduce accuracy on legitimate medical questions (cautiousness should not come at the cost of refusing to answer valid questions).

## Proposed Methodology

### Approach
We use a **prompt-based intervention** approach with real LLM APIs. This is practical (no fine-tuning needed), testable with current frontier models, and directly actionable for deployment.

**Two conditions:**
1. **Baseline:** Standard medical QA prompting (answer the question directly)
2. **Uncertainty-Augmented:** Prompts that explicitly instruct the model to assess evidence plausibility, express confidence levels, and select "I don't know" when evidence seems implausible or unsafe

**Three datasets:**
1. **MedQA (legitimate):** Standard USMLE questions - control condition
2. **Med-HALT reasoning_fake:** Absurd/counterfactual medical questions - primary test
3. **Med-HALT reasoning_fct:** Fact-checking where correct answer differs from student answer - secondary test

**Two models:**
1. **GPT-4.1** (OpenAI) - frontier model
2. **Claude Sonnet 4.5** (via OpenRouter) - frontier model

### Experimental Steps
1. Sample 100 questions from MedQA test set (legitimate medical QA)
2. Sample 100 questions from Med-HALT reasoning_fake (counterfactual)
3. Sample 100 questions from Med-HALT reasoning_fct (fact-checking)
4. For each sample × each model × each condition (baseline vs uncertainty-augmented):
   - Send prompt to API
   - Parse response for: selected answer, confidence level (1-10), reasoning
   - Record whether model selected "I don't know" option
5. Compute metrics and statistical tests
6. Total API calls: 100 × 3 datasets × 2 models × 2 conditions = 1,200 calls

### Baselines
- **Baseline prompting:** Direct medical QA prompt without uncertainty instructions
- **Random baseline:** Expected performance from random answer selection
- **Prior work reference:** Wu et al. (2024) AUROC ~0.58 for medical UE

### Evaluation Metrics
1. **Cautiousness Rate:** % of responses that include "I don't know" or express high uncertainty (confidence ≤ 3/10)
2. **Unsafe Completion Rate:** % of responses that confidently (confidence ≥ 7/10) give an incorrect answer
3. **Accuracy:** % correct on legitimate questions (MedQA)
4. **Confidence-Correctness AUROC:** How well self-reported confidence separates correct from incorrect answers
5. **Counterfactual Detection Rate:** On fake questions, % that correctly identify the question as implausible
6. **Effect Size:** Cohen's d between baseline and uncertainty-augmented conditions

### Statistical Analysis Plan
- **Primary test:** Chi-squared test comparing cautiousness rates (baseline vs uncertainty-augmented) on counterfactual questions
- **Secondary tests:** Paired proportion tests for unsafe completion rates, independent t-tests for confidence scores
- **AUROC:** Bootstrap 95% CI for confidence-correctness discrimination
- **Significance level:** α = 0.05, Bonferroni correction for multiple comparisons
- **Effect sizes:** Cohen's d, odds ratios with 95% CI

## Expected Outcomes
- **Support hypothesis:** Uncertainty-augmented prompts show ≥15% increase in cautiousness on counterfactual questions, ≥10% reduction in unsafe completions, without >5% accuracy drop on legitimate questions
- **Refute hypothesis:** No significant difference between conditions, or uncertainty prompts cause excessive refusal on legitimate questions

## Timeline and Milestones
1. Environment setup and data preparation: 15 min
2. Implementation (prompts, API calls, parsing): 45 min
3. Run experiments (1,200 API calls): 30 min
4. Analysis and visualization: 30 min
5. Documentation: 20 min
Buffer: 20 min

## Potential Challenges
- API rate limits → implement retry logic with exponential backoff
- Response parsing → use structured output formats (JSON)
- Model refusal → track refusal rates separately
- Cost → ~1,200 calls at ~500 tokens each ≈ $5-15 total

## Success Criteria
1. Complete data collection for all conditions
2. Statistical significance on at least one primary metric
3. Clear visualization of baseline vs uncertainty-augmented differences
4. Honest reporting of all results including negative findings
