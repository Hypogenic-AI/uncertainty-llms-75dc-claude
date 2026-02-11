# Literature Review: Uncertainty-Aware Medical LLMs

## 1. Introduction

Large language models (LLMs) have achieved remarkable performance on medical question-answering benchmarks, with GPT-4 surpassing the passing threshold on the United States Medical Licensing Examination (USMLE) and comparable assessments (Ness et al., 2024). Yet high average accuracy conceals a critical deficiency: these models lack reliable mechanisms for expressing *when they are uncertain or likely wrong*. In safety-critical domains such as clinical medicine, a confidently stated but incorrect answer can be far more harmful than an admission of uncertainty. This concern is amplified when models encounter adversarial or counterfactual inputs -- plausible-sounding but medically dangerous premises -- and respond with unwarranted confidence.

This literature review supports the research project **"Uncertainty-Aware Medical LLMs: Quantifying Doubt in the Face of Counterfactuals"**, which tests the hypothesis that *LLMs equipped with predictive and semantic uncertainty estimation will be less likely to provide confident, unsafe answers when presented with implausible or dangerous medical evidence*. We synthesize findings from seven deeply-read papers and contextualize them within eighteen additional works spanning uncertainty estimation methods, medical domain challenges, adversarial robustness, and available tooling. The review is organized as follows: Section 2 surveys uncertainty estimation methods for LLMs; Section 3 examines challenges specific to the medical domain; Section 4 addresses adversarial robustness and counterfactual resistance; Section 5 reviews frameworks and tooling; Section 6 identifies the research gap our project fills; and Section 7 provides a summary comparison table.

---

## 2. Uncertainty Estimation Methods for LLMs

### 2.1 Token-Level Approaches

The simplest uncertainty estimation (UE) methods operate at the token level. **Predictive entropy** computes the Shannon entropy of the next-token probability distribution, $H = -\sum_t p(t|x) \log p(t|x)$, and can be aggregated across the generated sequence via mean or max pooling. **Perplexity**, the exponentiated average negative log-likelihood, serves as a related measure. These approaches are computationally inexpensive and require only a single forward pass with access to output logits.

However, token-level methods suffer from a fundamental limitation: they conflate linguistic uncertainty (multiple valid phrasings of the same answer) with semantic uncertainty (genuine doubt about the answer's correctness). A model may assign high entropy across synonymous tokens ("myocardial infarction" vs. "heart attack") without being uncertain about the underlying medical fact. Wu et al. (2024) categorize these as "token probability" methods and find them largely ineffective for medical UE, with AUROC scores frequently near 0.5 -- no better than random -- across MedQA, MedMCQA, PubMedQA, and COVID-QA.

### 2.2 Semantic Entropy

Kuhn et al. (2023), published at ICLR 2023 (525+ citations), introduced **semantic entropy** to overcome the linguistic-vs-semantic confusion. The core idea is to compute entropy over *meaning clusters* rather than raw token sequences:

$$SE = -\sum_{C} p(C|x) \log p(C|x)$$

where $C$ represents clusters of semantically equivalent answers. The method samples multiple generations from the LLM (typically 5-10), clusters them using bidirectional natural language inference (NLI) via a DeBERTa-large classifier, and computes entropy over the resulting cluster distribution. Two answers are placed in the same cluster if each entails the other.

Farquhar et al. (2024) extended this work in a highly cited Nature publication (889+ citations), validating semantic entropy across a broader range of models (LLaMA 7B-65B, Falcon-40B, OPT family, GPT-3.5/4) and datasets (TriviaQA, SQuAD, BioASQ, Natural Questions, SVAMP). The method achieves **AUROC of 0.75-0.85+** for confabulation detection, substantially outperforming token-level baselines.

A critical limitation acknowledged by both papers is that semantic entropy only detects **confabulations** -- errors characterized by high variance across samples (the model "makes up" different answers each time). It cannot detect **systematic errors** where the model consistently produces the same wrong answer with high confidence. This distinction is particularly important for our project: counterfactual medical inputs may induce systematic errors where the model confidently adopts the false premise. The computational overhead of 5-10x over single generation is also a practical consideration for deployment.

### 2.3 Word-Sequence Entropy

Wang et al. (2024, EAAI 2025) proposed **word-sequence entropy**, which calibrates entropy at both word and sequence levels using semantic relevance weights. Their key insight is that standard entropy calculations suffer from **"generative inequality"**: irrelevant tokens (articles, filler words, formatting tokens) dilute the uncertainty signal by contributing entropy that is unrelated to the factual content of the answer.

The method assigns higher weight to tokens that are semantically relevant to the question and computes a weighted entropy measure. Tested specifically on medical datasets -- COVID-QA, MedQA, PubMedQA, MedMCQA, and MedQuAD -- using LLaMA-7B, LLaMA-2-7B-Chat, StableBeluga-7B, and Zephyr-7B-Alpha, word-sequence entropy achieves a **6.36% improvement on COVID-QA** in selecting low-uncertainty (higher-quality) responses. This method is particularly relevant to our project because it was designed explicitly for free-form medical QA, where answer length and format vary substantially.

### 2.4 Verbalized and Prompted Confidence

An alternative to probability-based methods is to ask the model directly about its confidence. Kadavath et al. (2022) demonstrated that language models "mostly know what they know" through self-evaluation experiments, including the $p(\text{True})$ method where the model is asked to evaluate the correctness of its own output. Lin et al. (2022) explored teaching models to express uncertainty in words, finding that verbalized confidence can be moderately calibrated but is sensitive to prompt design. Tian et al. (2023) showed that simply asking for calibrated confidence ("Just Ask for Calibration") via careful prompting can yield competitive results without fine-tuning.

Xiong et al. (2023) conducted a systematic evaluation of confidence elicitation strategies, finding that while LLMs can express uncertainty, their verbalized confidence is often poorly calibrated -- models tend toward overconfidence, particularly on tasks near the boundary of their knowledge. Wu et al. (2024) tested verbalized confidence specifically in the medical domain and found it underperforms even the modest baselines, with AUROC values that are often no better than chance. The survey by Geng et al. (2023) provides a comprehensive taxonomy of confidence estimation and calibration techniques, noting that calibration remains an open problem especially for instruction-tuned models.

Kapoor et al. (2024) explored whether models can be *taught* to express uncertainty more reliably through training, suggesting that fine-tuning on uncertainty-annotated data may improve calibration. However, this approach requires substantial labeled data indicating when uncertainty is warranted, which is expensive to obtain in medical domains.

### 2.5 Self-Verification and Consistency

Wu et al. (2024) proposed a **Two-Phase Verification** method specifically designed for medical UE. The approach decomposes reasoning into steps, generates verification questions for each step, answers these questions both independently and with the original context, and checks consistency via NLI. If the independent and contextual answers diverge, this signals potential unreliability.

Tested across MedQA, MedMCQA, PubMedQA, and COVID-QA using LLaMA-2-7B-Chat, LLaMA-3-8B-Instruct, and Mistral-7B-Instruct, Two-Phase Verification achieved the **best overall AUROC of 0.5858** across four UE categories (token probability, verbalized confidence, consistency-based, and self-verification). While this is the best result in the study, the absolute AUROC values remain alarmingly low -- barely above random classification. This constitutes one of the most important findings for our project: **medical uncertainty estimation is fundamentally harder than general-domain UE**, and methods that perform well on TriviaQA or SQuAD often fail in clinical contexts.

### 2.6 Conformal Prediction

Conformal prediction offers a distribution-free framework for constructing prediction sets with guaranteed coverage. Wang et al. (2024, EMNLP 2024) introduced **ConU** (Conformal Uncertainty), applying conformal prediction to black-box LLMs for open-ended natural language generation. Their pipeline samples multiple answers, computes a novel uncertainty criterion combining normalized frequency with semantic diversity, and applies conformal calibration to achieve **at least 90% correctness coverage guarantees**. Tested across TriviaQA, Natural Questions, CoQA, and SQuAD using a wide range of models (LLaMA-2-7B/13B/70B, LLaMA-3-8B/70B, Mistral-7B, Claude 3.5 Sonnet, GPT-3.5, GPT-4), ConU provides theoretically grounded abstention. The follow-up work SConU (ACL 2025) extends this to selective prediction.

Related conformal prediction approaches include Quach et al. (2023), who developed conformal language modeling methods, Kumar et al. (2023), who applied conformal prediction to multiple-choice QA, and Ren et al. (2023), whose **KnowNo** framework uses conformal prediction in robotics to determine when the system should ask for clarification rather than act. Su et al. (2024) addressed the practical setting of API-based conformal prediction where only text outputs (not logits) are available.

The conformal prediction paradigm is attractive for medical applications because it provides formal coverage guarantees rather than heuristic uncertainty scores. However, the calibration set must be representative of deployment conditions, which is challenging when adversarial or counterfactual inputs shift the distribution.

### 2.7 Internal State Probing

Chen et al. (2024) introduced **INSIDE** (INternal States for hallucination DEtection), which trains lightweight probes on the hidden states of transformer layers to detect hallucinations. Unlike generation-based methods, internal state probing can potentially detect systematic errors (where the model consistently generates the same wrong answer) by identifying patterns in how the model processes uncertain inputs. This complements semantic entropy, which only catches high-variance confabulations, and is relevant to our project because counterfactual inputs may produce consistent but wrong outputs.

### 2.8 Bilateral Confidence Estimation (AFICE)

Zhao et al. (2025, AAAI 2025) proposed **AFICE** (Aligning for Faithful Integrity with Confidence Estimation), which is the most directly relevant prior work to our project's counterfactual resistance objective. AFICE introduces **Bilateral Confidence Estimation (BCE)**, which combines two signals:

1. **Hidden state confidence** at the question level, extracted from internal representations.
2. **Cumulative probability ratios** at the answer level, measuring the model's token-level certainty during generation.

Critically, AFICE uses DPO-based alignment to train LLMs to *resist opposing arguments* -- a direct defense against sycophancy. When presented with a correct answer followed by a persuasive counterargument, AFICE-trained models maintain their correct answers rather than capitulating. This is precisely the behavior needed for counterfactual resistance in medical settings: a model should not abandon a correct diagnosis simply because the user presents a plausible-sounding but false premise.

---

## 3. Medical Domain Challenges

### 3.1 Medical QA Benchmarks and Datasets

The medical UE literature relies on several established benchmarks. **MedQA** contains USMLE-style multiple-choice questions requiring multi-step clinical reasoning. **MedMCQA** provides a large-scale medical MCQ dataset from Indian medical entrance exams. **PubMedQA** tests reasoning over biomedical literature abstracts. **COVID-QA** focuses on pandemic-related questions derived from scientific publications. **BioASQ** is a biomedical semantic indexing and QA challenge. **MedQuAD** aggregates medical question-answer pairs from NIH sources.

These benchmarks vary substantially in format (multiple-choice vs. open-ended), reasoning complexity, and domain specificity. Methods that perform well on one may fail on another, making comprehensive evaluation essential.

### 3.2 Why Uncertainty Estimation is Harder in Medicine

Wu et al. (2024) provide the most systematic evidence that **medical UE is fundamentally more difficult than general-domain UE**. Their comprehensive evaluation across four categories of UE methods (token probability, verbalized confidence, consistency-based, and self-verification) reveals that methods achieving AUROC of 0.75+ on general QA tasks often fall to near-random performance (AUROC approximately 0.50) on medical benchmarks.

Several factors contribute to this difficulty. First, medical reasoning often requires multi-step inference chains where uncertainty compounds across steps. Second, medical terminology is precise -- small differences in wording can change clinical meaning entirely ("systolic" vs. "diastolic"), making semantic similarity judgments harder. Third, medical knowledge has complex conditional dependencies (a treatment that is correct for one patient population may be dangerous for another), which challenges simple consistency-based methods. Fourth, the base rates of medical facts create calibration challenges: rare conditions may be systematically underrepresented in training data, leading to confident but incorrect answers.

### 3.3 Medical Hallucination Benchmarks

Umapathi et al. (2023) introduced **Med-HALT** (Medical Domain Hallucination Test), a benchmark specifically designed to evaluate hallucination in medical LLMs. Med-HALT includes reasoning hallucination tests (where models must detect planted errors in medical reasoning chains) and memory hallucination tests (where models must identify fabricated medical facts). Pandit et al. (2025) proposed **MedHallu**, a more recent benchmark focusing on hallucination detection in medical contexts with fine-grained annotation of hallucination types. These benchmarks complement our project by providing standardized evaluation protocols for medical hallucination detection, though they do not directly address uncertainty estimation as a detection mechanism.

---

## 4. Adversarial Robustness and Counterfactual Testing

### 4.1 MedFuzz: Adversarial Medical Question Fuzzing

Ness et al. (2024, Microsoft Research) introduced **MedFuzz**, an adversarial fuzzing framework that borrows the concept of "fuzzing" from software security testing and applies it to medical QA robustness evaluation. An attacker LLM iteratively modifies medical questions -- introducing plausible but misleading clinical details, altering demographic information, or adding irrelevant distractors -- while preserving the question's grammatical correctness and surface plausibility.

Under MedFuzz attacks, **GPT-4 accuracy dropped from 90.2% to 85.4%**, a clinically significant degradation. GPT-3.5, Claude-Sonnet, and BioMistral-7B showed comparable or larger drops. The key finding is that even state-of-the-art models are vulnerable to subtle, adversarially crafted modifications of medical questions.

Crucially, **MedFuzz does not measure uncertainty** -- it evaluates only whether the model's answer changes from correct to incorrect. This is precisely the gap our project addresses: we ask not just whether the model gets the answer wrong under adversarial perturbation, but whether its uncertainty estimates appropriately increase, signaling that the input should be treated with caution.

### 4.2 Sycophancy and Counterfactual Resistance

Sycophancy -- the tendency of LLMs to agree with the user's stated position even when it contradicts the model's own knowledge -- is a well-documented failure mode that is particularly dangerous in medical contexts. If a patient presents a false medical claim and the model agrees rather than correcting, the consequences could be severe.

Zhao et al. (2025) directly address this through AFICE's DPO-based alignment, training models to maintain correct answers even when presented with opposing evidence. Their bilateral confidence estimation provides a quantitative signal: when the model's internal confidence remains high despite user-presented counterarguments, it should maintain its position; when confidence genuinely drops (e.g., because the counterargument reveals a legitimate consideration), the model should acknowledge uncertainty.

This framework maps directly onto our counterfactual testing scenario. When we present models with implausible medical evidence, we want uncertainty estimates to increase (correctly signaling that something is wrong) rather than the model simply adopting the false premise with unchanged confidence.

### 4.3 Fragility of Uncertainty Estimates

Zeng et al. (2024) demonstrated that **uncertainty estimates in LLMs are fragile** -- small, semantically irrelevant changes to inputs (such as rephrasing, reordering options, or adding benign context) can cause large changes in uncertainty scores without affecting the underlying answer quality. This fragility raises concerns about deploying uncertainty-based safety mechanisms: if an adversary can manipulate not just the model's answer but also its uncertainty signal, then UE-based safeguards may be circumvented. Hou et al. (2023) explored a complementary angle, examining how input clarification questions can improve understanding and implicitly reduce uncertainty in ambiguous situations.

For our project, fragility is both a challenge and a research question. We must evaluate whether the UE methods we deploy are robust to the specific types of perturbations introduced by counterfactual medical scenarios, or whether adversarial inputs can simultaneously degrade accuracy and suppress uncertainty signals.

---

## 5. Frameworks and Tooling

### 5.1 LM-Polygraph

Fadeeva et al. (2023) introduced **LM-Polygraph**, an open-source framework implementing over 40 uncertainty estimation methods for LLMs in a unified API. The framework supports token-level methods (entropy, perplexity), sequence-level methods (predictive entropy, mutual information), semantic methods (semantic entropy variants), and verbalized confidence approaches. LM-Polygraph is model-agnostic and supports both white-box (logit access) and black-box (text-only) settings.

For our project, LM-Polygraph provides a practical foundation for implementing and comparing multiple UE methods without building each from scratch. The framework's unified evaluation protocol ensures fair comparison across methods.

### 5.2 Available Codebases and Benchmarks

Several codebases relevant to our project are publicly available. The **semantic_uncertainty** repository (Kuhn et al., 2023; Farquhar et al., 2024) provides the reference implementation of semantic entropy, including NLI-based clustering and entropy computation. The **Conformal-Uncertainty-Criterion** codebase (Wang et al., 2024) implements the ConU pipeline for conformal prediction with LLMs. The **AFICE** repository (Zhao et al., 2025) provides code for bilateral confidence estimation and DPO-based alignment for counterfactual resistance. The **LLM-Uncertainty-Bench** repository (Ye et al., 2024) offers standardized benchmarks for evaluating UE methods. The **MedHallu** and **Med-HALT** codebases (Pandit et al., 2025; Umapathi et al., 2023) provide medical hallucination evaluation infrastructure. The **semantic-entropy-probes** repository provides lightweight probe-based alternatives to full semantic entropy computation.

Ye et al. (2024) conducted a comprehensive benchmarking of LLM uncertainty quantification methods across multiple tasks, providing a valuable reference for expected performance ranges. Shorinwa et al. (2024) survey the broader landscape of uncertainty quantification for LLMs, offering a taxonomy that situates our work within the field.

---

## 6. Research Gap and Our Contribution

The literature reveals a clear gap at the intersection of three active research areas:

1. **Uncertainty estimation methods** (semantic entropy, conformal prediction, word-sequence entropy, bilateral confidence estimation) have been developed and validated primarily on general-domain QA tasks, with limited medical-specific evaluation.

2. **Medical adversarial robustness** (MedFuzz) has demonstrated that models are vulnerable to subtle input perturbations but has not measured whether uncertainty signals detect these attacks.

3. **Counterfactual resistance** (AFICE) has shown that alignment can help models resist opposing arguments but has not been evaluated in the context of medical counterfactuals specifically.

No existing work combines uncertainty estimation with counterfactual perturbation testing in the medical domain. Our project fills this gap by:

- Systematically evaluating whether established UE methods (semantic entropy, word-sequence entropy, conformal prediction, bilateral confidence estimation) produce appropriately elevated uncertainty scores when models are presented with MedFuzz-style counterfactual medical inputs.
- Testing the hypothesis that uncertainty-aware models are less likely to provide confident, unsafe answers to adversarial medical queries.
- Quantifying the degree to which different UE methods are robust to (or fragile under) adversarial medical perturbations, directly addressing the concerns raised by Zeng et al. (2024).
- Providing practical recommendations for deploying uncertainty-based safety mechanisms in medical LLM applications.

This work bridges the evaluation gap between MedFuzz (which measures accuracy degradation but not uncertainty) and semantic entropy / conformal prediction (which measure uncertainty but have not been tested under adversarial medical conditions).

---

## 7. Summary Table of Methods

| Method | Paper | Year | Venue | Type | Key Metric | Models Tested | Medical Datasets | Key Limitation |
|--------|-------|------|-------|------|------------|---------------|-----------------|----------------|
| Token-level entropy | Various | -- | -- | White-box | AUROC ~0.50 (medical) | Various | MedQA, MedMCQA, PubMedQA, COVID-QA | Confuses linguistic and semantic uncertainty |
| Semantic Entropy | Kuhn et al. | 2023 | ICLR | White-box | AUROC 0.75-0.85 (general) | OPT (6.7B-66B), LLaMA | TriviaQA, SQuAD, BioASQ | Only detects confabulations; 5-10x overhead |
| Semantic Entropy (extended) | Farquhar et al. | 2024 | Nature | White-box | AUROC 0.75-0.85+ | LLaMA (7B-65B), Falcon-40B, GPT-3.5/4 | BioASQ | Same confabulation-only limitation |
| Word-Sequence Entropy | Wang et al. | 2024 | EAAI 2025 | White-box | +6.36% on COVID-QA | LLaMA-7B, LLaMA-2-7B-Chat, StableBeluga-7B, Zephyr-7B | COVID-QA, MedQA, PubMedQA, MedMCQA, MedQuAD | Limited model scale tested |
| Two-Phase Verification | Wu et al. | 2024 | -- | Black-box | AUROC 0.5858 (avg) | LLaMA-2-7B-Chat, LLaMA-3-8B-Instruct, Mistral-7B | MedQA, MedMCQA, PubMedQA, COVID-QA | Best medical result still near random |
| Verbalized Confidence | Kadavath; Lin; Tian; Xiong | 2022-2023 | Various | Black-box | Variable, often poorly calibrated | Various | Various | Overconfidence; prompt-sensitive |
| ConU (Conformal) | Wang et al. | 2024 | EMNLP | Black-box | >=90% coverage | LLaMA-2/3 (7B-70B), Mistral-7B, Claude 3.5, GPT-3.5/4 | TriviaQA, NQ, CoQA, SQuAD | Requires representative calibration set |
| INSIDE | Chen et al. | 2024 | -- | White-box | Probe-based | Various | -- | Requires hidden state access |
| AFICE (BCE) | Zhao et al. | 2025 | AAAI | White-box | DPO-aligned | Various | -- | Requires fine-tuning; not tested on medical data |
| MedFuzz | Ness et al. | 2024 | -- | Adversarial eval | Accuracy 90.2% to 85.4% (GPT-4) | GPT-4, GPT-3.5, Claude-Sonnet, BioMistral-7B | MedQA variants | Does not measure uncertainty |
| LM-Polygraph | Fadeeva et al. | 2023 | -- | Framework | 40+ methods | Model-agnostic | -- | Framework, not a method |
| Med-HALT | Umapathi et al. | 2023 | -- | Benchmark | Hallucination rates | Various | Medical hallucination test | Evaluation only; no UE method |
| MedHallu | Pandit et al. | 2025 | -- | Benchmark | Hallucination detection | Various | Medical hallucination benchmark | Evaluation only; no UE method |

---

## References

- Chen, Y., et al. (2024). INSIDE: LLM's Internal States Retain the Power of Hallucination Detection. *arXiv preprint*.
- Fadeeva, E., et al. (2023). LM-Polygraph: Uncertainty Estimation for Language Models. *arXiv preprint*.
- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature*, 630, 625-630.
- Geng, J., et al. (2023). A Survey of Language Model Confidence Estimation and Calibration. *arXiv preprint*.
- Hou, Y., et al. (2023). From Ambiguity to Clarity: Input Clarification for Better Understanding. *arXiv preprint*.
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv preprint*.
- Kapoor, S., et al. (2024). Models That Can Be Taught to Express Uncertainty. *arXiv preprint*.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*.
- Kumar, B., et al. (2023). Conformal Prediction with Large Language Models for Multi-Choice Question Answering. *arXiv preprint*.
- Lin, S., Hilton, J., & Evans, O. (2022). Teaching Models to Express Their Uncertainty in Words. *TMLR*.
- Ness, R., et al. (2024). MedFuzz: Exploring the Robustness of Large Language Models in Medical Question Answering. *Microsoft Research*.
- Pandit, S., et al. (2025). MedHallu: A Benchmark for Detecting Medical Hallucinations. *arXiv preprint*.
- Quach, V., et al. (2023). Conformal Language Modeling. *arXiv preprint*.
- Ren, A., et al. (2023). KnowNo: Knowing When You Don't Know for Calibrated Prediction with Language Models. *arXiv preprint*.
- Shorinwa, O., et al. (2024). A Survey on Uncertainty Quantification for Large Language Models. *arXiv preprint*.
- Su, W., et al. (2024). API-Based Conformal Prediction for Large Language Models. *arXiv preprint*.
- Tian, K., et al. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models. *EMNLP 2023*.
- Umapathi, L. K., et al. (2023). Med-HALT: Medical Domain Hallucination Test for Large Language Models. *CoNLL 2023*.
- Wang, J., et al. (2024). Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration. *EMNLP 2024*.
- Wang, Z., et al. (2024). Word-Sequence Entropy: Towards Uncertainty Estimation in Free-Form Medical Question Answering Applications and Beyond. *Engineering Applications of Artificial Intelligence (EAAI) 2025*.
- Wu, Y., et al. (2024). Uncertainty Estimation of Large Language Models in Medical Question Answering. *arXiv preprint*.
- Xiong, M., et al. (2023). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *arXiv preprint*.
- Ye, J., et al. (2024). Benchmarking LLM Uncertainty Quantification Methods. *arXiv preprint*.
- Zeng, Z., et al. (2024). The Uncertainty in LLMs is Fragile. *arXiv preprint*.
- Zhao, L., et al. (2025). AFICE: Aligning for Faithful Integrity with Confidence Estimation. *AAAI 2025*.
