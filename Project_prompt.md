MASTER PROMPT — FLAGSHIP DEEP LEARNING RESEARCH PROJECT

I want to build a flagship Deep Learning + Computer Vision research project for my resume and placements.

I have been assigned a research paper for my Deep Learning in Computer Vision course. I will attach the paper in this conversation. The project must be based on the core idea and methodology of that paper, specifically the LOTA approach for AI-generated image detection.

Your job is to act as my research mentor, Deep Learning instructor, experiment designer, ML engineer, and project reviewer throughout the project.

Do not simply agree with my ideas. Critically evaluate them. If something has poor effort-to-reward ratio, is unnecessary, technically weak, scientifically unjustified, or likely to waste time, tell me clearly and suggest a better alternative.

The goal is not to build the largest project possible. The goal is to build the strongest, most defensible, research-oriented Deep Learning project possible for the effort and compute available.

1. PROJECT CONTEXT

The assigned research paper is based on detecting AI-generated images using low-bit-plane artifacts.

The paper's core methodology includes:

Bit-Plane Guided Noise Image Generation / Extraction
Bit-plane slicing
Combining low-order bit planes to construct a noise representation
Different normalization strategies such as scaling and thresholding
Maximum Gradient Patch Selection (MGPS)
A Noise-Based Classifier (NBC)
A Noise-Guided Classifier (NGC)
Attention-based guidance/fusion in the more advanced classifier

The exact terminology, equations, implementation details, and architecture must be taken directly from the attached research paper.

Do not assume that my earlier descriptions of the paper are perfectly accurate. Once I provide the paper, first carefully analyze it and establish the ground truth implementation plan from the paper itself.

The project must begin as a faithful reproduction of the research paper.

Only after the baseline reproduction is working should we add our own experiments and extensions.

2. FINAL PROJECT GOAL

The final project should be positioned as:

A research-oriented Deep Learning system for detecting AI-generated images using low-bit-plane forensic artifacts, with a systematic investigation of cross-generator generalization, robustness to real-world image transformations, focused ablation studies, and an experiment intelligence layer for querying experimental results.

The project is not supposed to be:

A SaaS platform
A generic AI chatbot
A generic RAG application
An Agentic AI project
A project that combines every trending technology

I am separately planning to build another project focused on Agentic AI, LangGraph, planning, tool use, orchestration, and autonomous workflows.

Therefore, do not try to force agents, LangGraph, multi-agent systems, or agentic workflows into this project.

This project should remain primarily:

Deep Learning
+
Computer Vision
+
AI Image Forensics
+
Research Reproduction
+
Experimental Design
+
Generalization Analysis
+
Robustness Testing
+
Lightweight Experiment Intelligence

The project should demonstrate depth rather than technology count.

3. CORE PROJECT PHILOSOPHY

Every component must justify its existence.

Before recommending any new architecture, model, framework, or technology, evaluate:

Expected Interview/Resume Value
×
Scientific Value
×
Learning Value
--------------------------------
Implementation Time
+
Compute Cost
+
Complexity
+
Risk of Failure

We are optimizing for effort-to-reward ratio.

Do not recommend something simply because it sounds impressive.

For example:

Bad reasoning:

"Let's add a Vision Transformer because Transformers look good on a resume."

Good reasoning:

"Let's test whether Transformer-based global attention improves generalization to unseen generators compared with convolutional inductive biases."

However, even a scientifically valid idea should be added only if its expected value justifies the compute and implementation cost.

The project should follow this principle:

A smaller number of deeply investigated ideas is better than many shallow components.

4. FINAL LOCKED PROJECT SCOPE

The project has four major components.

COMPONENT A — FAITHFUL REPRODUCTION OF THE PAPER

This is non-negotiable.

First, fully understand and implement the methodology described in the attached paper.

The initial pipeline should conceptually look like:

INPUT IMAGE
     ↓
Bit-Plane Extraction
     ↓
Low-Bit-Plane Noise Representation
     ↓
Normalization / Composition
     ↓
Maximum Gradient Patch Selection
     ↓
Selected Forensic Patch
     ↓
Classifier
     ↓
REAL / AI-GENERATED

The paper may contain multiple classifier variants.

The project should reproduce them in increasing complexity.

Conceptually:

Noise Representation
        ↓
NBC
(Noise-Based Classifier)

Then, after NBC works:

Original Image Features
        +
Selected Noise Patch
        ↓
NGC
(Noise-Guided Classifier)
        ↓
Attention / Guidance mechanism
        ↓
REAL / FAKE

However, the exact architecture must come from the paper.

Required workflow for reproduction

Before writing the full training pipeline:

Read the entire paper carefully.
Identify every major module.
Extract every equation.
Understand the purpose of every operation.
Create a module-by-module explanation.
Implement small test cases for every module.
Visually inspect intermediate outputs.

For example:

Bit-plane sanity checks

Given an image:

Original Image
      ↓
Bit Plane 0
Bit Plane 1
Bit Plane 2
...

Verify that:

Bit extraction is correct.
Bit-plane ordering is correct.
Composition follows the paper's equation.
Scaling/thresholding behaves correctly.

Generate visualizations similar in spirit to the paper's examples.

Do not proceed if the implementation is obviously inconsistent with the paper.

MGPS sanity checks

For Maximum Gradient Patch Selection:

Visualize candidate patches.
Compute the gradient score exactly as defined in the paper.
Verify directional gradients.
Verify patch selection.
Overlay the selected patch on the original/noise image.

Do not treat MGPS as a black box.

We should be able to visually and mathematically explain:

Why was this patch selected?

5. DATASET

The primary dataset should be the dataset used by the paper wherever feasible.

The paper discusses the GenImage dataset, which contains real images and images generated by multiple AI image generators.

Before implementation, verify from the attached paper:

Which dataset version the authors used
Which generators were included
Train/test splits
Preprocessing
Image resolution
Dataset scale

Potential dataset locations to investigate include the official GenImage project/repository and official dataset hosting pages. Do not blindly download from an unofficial mirror without first checking the official source and paper documentation.

When helping me start the project:

Identify the official dataset source used by the paper.
Identify whether the full dataset is realistically usable with our compute.
Provide exact download/setup instructions.
Recommend a reproducible subset if full-scale training is infeasible.
Clearly document any deviation from the paper.

Possible generator families in the dataset may include models such as:

BigGAN
Midjourney
Stable Diffusion variants
ADM
GLIDE
Wukong

Do not assume this exact list without checking the paper/dataset.

6. COMPUTE-CONSTRAINED REPRODUCTION

We may not have access to research-scale GPU infrastructure.

Possible compute environments include:

Google Colab
Kaggle Notebooks
College GPU resources
Paid cloud GPU access if affordable

Therefore, do not assume we can reproduce the paper using the full training scale.

Before large experiments, define a documented compute budget.

For example:

Dataset:
N images per class/generator

Image resolution:
X × X

Backbone:
Specified architecture

Batch size:
X

Epochs:
X

GPU:
X

Estimated training time:
X

The project must be honest about reproduction scale.

If the paper achieves a higher score because it trained on millions of images while we train on a smaller subset, the README/report should explicitly state:

We reproduce the methodology and investigate relative trends and behavior under constrained computational resources rather than claiming exact reproduction of full-scale paper results.

However:

Compute constraints must not become an excuse for implementation errors.

First validate the implementation carefully. Only then attribute remaining performance differences to data/compute differences.

7. REPRODUCTION STRATEGY

Start small.

Initial debugging scale:

A few thousand images
per generator

The exact number should be selected based on available compute.

The sequence should be:

Small dataset
      ↓
Validate preprocessing
      ↓
Validate bit-plane extraction
      ↓
Validate noise generation
      ↓
Validate MGPS
      ↓
Train NBC
      ↓
Verify end-to-end pipeline
      ↓
Scale data if necessary
      ↓
Implement NGC

Do not attempt to debug:

Data pipeline
+
BGNIG
+
MGPS
+
Attention
+
Training

all at the same time.

Build incrementally.

The first meaningful checkpoint is:

A working end-to-end single-generator train/evaluation pipeline with honest comparison against the relevant results reported in the paper.

Do not expect to reproduce the exact paper number.

Compare:

Relative behavior
Trends
Architecture comparisons
Generalization patterns

not only absolute accuracy.

8. MAIN RESEARCH EXTENSION — CROSS-GENERATOR GENERALIZATION

This is the main contribution beyond reproduction.

The central research question is:

How well does a low-bit-plane forensic detector generalize when it encounters an AI image generator that was completely excluded from training?

We will use a controlled Leave-One-Generator-Out (LOGO) experimental design.

However, full LOGO across every available generator may be too computationally expensive.

Therefore, select approximately four deliberately diverse generators.

The selection must not be random.

The goal is to maximize diversity across generator families.

Conceptually, the selected generators should represent categories such as:

GAN-based generator
        +
Latent diffusion / Stable Diffusion family
        +
A difficult stress-test generator
        +
Architecturally distinct diffusion generator

A potential example, only if supported by the available dataset:

BigGAN
One Stable Diffusion variant
Midjourney
ADM or GLIDE

The final choices must be documented with an explicit rationale.

For example:

These generators were selected to maximize architectural and distributional diversity rather than choosing generators arbitrarily.

LOGO experimental structure

For four selected generators:

Experiment 1:
Train on B + C + D
Test on A

Experiment 2:
Train on A + C + D
Test on B

Experiment 3:
Train on A + B + D
Test on C

Experiment 4:
Train on A + B + C
Test on D

Each experiment must record:

Experiment ID
Training Generators
Excluded Generator
Model
Bit-plane Configuration
Patch Configuration
Dataset Size
Epochs
Seed
Accuracy
AUROC
F1
Precision
Recall
Training Time
Inference Time

Use the same training protocol wherever scientifically appropriate.

Do not manually rewrite training code for every rotation.

The training pipeline should accept configuration such as:

excluded_generator: Midjourney

or equivalent command-line/config parameters.

9. FULL GENERATOR EVALUATION

After training each LOGO model, evaluate it against all available generator test sets, not only the held-out generator.

This is important because:

Training = expensive
Evaluation = relatively cheap

This provides a richer performance matrix without requiring additional training runs.

The final result should allow us to analyze:

Training distribution
        ↓
Model
        ↓
Performance on:
Generator A
Generator B
Generator C
Generator D
...

Create:

Performance tables
Heatmaps
Seen vs unseen comparisons
Per-generator performance plots

Possible research questions:

Which generator is hardest to detect?
Which generator family generalizes poorly?
Does training diversity improve generalization?
Are GAN-generated images easier or harder to detect than diffusion-generated images?
Which unseen generator causes the largest performance drop?

Do not invent explanations.

If the cause is uncertain, say:

The observed result suggests hypothesis X, but additional experiments would be required to establish causation.

10. FOCUSED ABLATION STUDIES

Do not run every possible ablation.

We are intentionally limiting the scope.

The core ablations are:

Ablation 1 — Number of low bit planes

Investigate different numbers of low-order bit planes.

The exact range should follow the paper where possible.

For example:

1 bit plane
2 bit planes
3 bit planes
4 bit planes
...

The goal is to answer:

How does the amount of low-bit information affect detection and generalization?

Evaluate using appropriate metrics.

Ablation 2 — Maximum Gradient Patch Selection

Compare MGPS against simple alternatives.

For example:

MGPS-selected patch
vs
Random patch
vs
Center patch

The purpose is to determine whether MGPS actually contributes meaningful information.

Research question:

Does selecting high-gradient forensic regions improve detection compared with naive patch selection?

Ablation 3 — NBC vs NGC

Compare the classifier variants described in the paper.

Conceptually:

Noise-only classification
vs
Noise-guided classification using original image information

The exact architecture must follow the paper.

Research question:

Does incorporating information from the original image improve detection or generalization compared with using only low-bit-plane noise?

11. ROBUSTNESS TESTING

This is a high-effort-to-reward component and should definitely be included.

Use already-trained models.

Do not retrain unless there is a scientifically justified reason.

Evaluate robustness under realistic image transformations.

Core transformations:

JPEG compression

Use multiple quality levels.

Prefer values that can be compared with the paper where applicable.

For example:

100
95
90
85
...
Gaussian blur

Test multiple sigma levels.

Again, align with the paper where possible.

For example:

σ = 0
σ = 1
σ = 2
σ = 3

Additional low-cost tests if time permits:

Resize/downsample
Cropping
Gaussian noise
Other realistic image degradation

For each transformation:

Transformation Strength
        ↓
Accuracy
AUROC
F1
Performance Drop

Create degradation curves.

The research question is:

How dependent is the low-bit-plane forensic signal on image fidelity?

This is especially important because compression and resizing may destroy low-order pixel information.

12. ADVERSARIAL ATTACKS

FGSM/PGD and other gradient-based adversarial attacks are not part of the required scope.

They are stretch/future work.

Do not spend core project time implementing them unless:

The reproduction is complete
LOGO experiments are complete
Ablations are complete
Robustness testing is complete
Documentation is complete

If implemented, clearly separate:

Real-world robustness

from:

Adversarial robustness

Do not treat them as equivalent.

13. CNN VS VISION TRANSFORMER

This is optional.

Do not implement it just because ViTs are popular.

Only consider it after all core work is complete.

The experiment must answer a real question:

Does Transformer-based attention improve generalization to unseen AI image generators compared with convolutional architectures?

If compute, time, or experimental evidence does not justify this comparison, skip it without guilt.

The project is already strong without ViT.

If implemented, use the existing experimental pipeline and avoid building an entirely new infrastructure.

14. EXPERIMENT TRACKING

Set up experiment tracking from the beginning.

Use a tool such as:

Weights & Biases

or another suitable experiment tracking system.

Track:

Experiment ID
Git commit/version
Model architecture
Dataset subset
Generators
Excluded generator
Bit planes
Patch size
Normalization method
Learning rate
Batch size
Epochs
Random seed
Loss curves
Validation metrics
AUROC
F1
Accuracy
Training time

The goal is reproducibility.

Do not wait until the end to start tracking experiments.

Every significant experiment should be traceable.

15. EXPERIMENT DATABASE

Maintain a lightweight structured database.

SQLite is sufficient initially.

PostgreSQL is optional if there is a practical reason.

Possible conceptual schema:

Experiments
----------
experiment_id
name
timestamp
git_version
configuration_id
model_id

Models
------
model_id
architecture
parameters
description

Generators
----------
generator_id
name
family

Configurations
--------------
configuration_id
bit_planes
patch_size
normalization
learning_rate
batch_size
epochs
seed

Metrics
-------
experiment_id
generator_id
split
accuracy
auroc
f1
precision
recall

RobustnessResults
-----------------
experiment_id
transformation
strength
accuracy
auroc
f1

Do not finalize the schema blindly at the beginning.

Allow the schema to evolve based on the actual experiments.

The database should contain real experimental evidence, not fake/sample records created only for the demo.

16. EXPERIMENT INTELLIGENCE SYSTEM

This is the placement-relevant LLM/RAG-style component.

However, it must remain lightweight and solve a real problem.

The purpose is:

Allow a user/researcher to query the accumulated experimental results using natural language.

Example questions:

Which model performed best on unseen generator X?
Which generator was hardest to detect?
Which model was most robust to JPEG compression?
How much performance dropped after blur?
Did MGPS improve performance?
Which bit-plane configuration had the best average AUROC?
Which experiment had the largest generalization gap?
IMPORTANT: DO NOT BUILD GENERAL NL-TO-SQL

Do not create a generic system where an LLM can generate arbitrary SQL against the database.

That is unnecessary and creates reliability and scope problems.

Instead, create a constrained system.

Architecture:

USER QUESTION
      ↓
Intent / Query Template Detection
      ↓
Parameter Extraction
      ↓
Validated Query Template
      ↓
Parameterized SQL Query
      ↓
Experiment Database
      ↓
Structured Evidence
      ↓
LLM or deterministic explanation
      ↓
FINAL ANSWER

Support approximately 8–12 useful query templates.

Examples:

BEST_MODEL_FOR_GENERATOR
(metric, generator)

HARDEST_GENERATOR
(metric)

COMPARE_MODELS
(model_a, model_b, metric)

MOST_ROBUST_MODEL
(transformation, strength, metric)

LARGEST_PERFORMANCE_DROP
(transformation)

BEST_BIT_PLANE_CONFIGURATION
(metric)

MGPS_EFFECT
(metric)

UNSEEN_GENERATOR_GENERALIZATION
(model/configuration)

The LLM's role should primarily be:

Understanding the user's question
Mapping it to a supported intent/template
Extracting parameters
Explaining retrieved results

The LLM should not invent results.

All answers must be grounded in actual database query results.

For unsupported questions, the system should gracefully say that the requested analysis is not currently supported.

17. VECTOR RETRIEVAL

Do not add a vector database simply to claim that the project uses RAG.

Only add vector retrieval if we actually accumulate useful unstructured information such as:

Experiment observations
Failure analyses
Research notes
Training anomalies
Written conclusions
Error-analysis reports

Then use:

Structured quantitative data
        → SQL

Qualitative observations
        → Vector retrieval

Potential hybrid architecture:

USER QUESTION
       ↓
Query Router
      / \
     /   \
SQL       Semantic Retrieval
 |             |
Metrics     Research Notes
     \       /
      Evidence
         ↓
        LLM
         ↓
      Answer

However, this is optional.

Do not implement vector retrieval unless the data justifies it.

18. CLEAN DEMO

The final system should have a clean demonstration interface.

It is not a SaaS platform.

Do not add:

Authentication
User accounts
Billing
Complex backend infrastructure
Unnecessary dashboards

The main detection demo should allow:

UPLOAD IMAGE
      ↓
Prediction:
Real / AI-generated
      ↓
Confidence
      ↓
Bit-plane / noise visualization
      ↓
Selected MGPS patch
      ↓
Optional model explanation

A separate interface/tab/page can provide:

EXPERIMENT EXPLORER

Ask:
"Which model generalizes best to unseen generators?"

      ↓

Retrieve evidence from experiment database

      ↓

Explain result

The demo should help visualize the research, not pretend to be a commercial product.

19. CODEBASE STRUCTURE

Create the repository properly from the beginning.

Suggested structure:

project-root/

├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── reproduction.yaml
│   ├── logo.yaml
│   └── ablation.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── splits.py
│   │
│   ├── forensic/
│   │   ├── bitplanes.py
│   │   ├── noise_generation.py
│   │   └── mgps.py
│   │
│   ├── models/
│   │   ├── nbc.py
│   │   ├── ngc.py
│   │   └── backbones.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── robustness.py
│   │   └── visualization.py
│   │
│   ├── experiments/
│   │   ├── database.py
│   │   ├── logger.py
│   │   └── queries.py
│   │
│   └── intelligence/
│       ├── intent_router.py
│       ├── templates.py
│       └── explanation.py
│
├── notebooks/
│   ├── paper_understanding.ipynb
│   ├── bitplane_debugging.ipynb
│   └── exploratory_analysis.ipynb
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_logo.py
│   └── run_ablations.py
│
├── experiments/
│   ├── results/
│   └── figures/
│
├── docs/
│   ├── paper_notes.md
│   ├── decisions.md
│   └── experiment_protocol.md
│
└── app/
    └── demo.py

This structure may be improved if needed, but do not overengineer it.

20. DEVELOPMENT ORDER

Follow this order unless there is a strong technical reason to change it.

Step 1 — Understand the paper

Before implementation:

Explain every major module.
Explain every equation.
Create a pipeline diagram.
Identify unclear implementation details.
Identify hyperparameters reported in the paper.

Do not move forward until I understand the paper.

Step 2 — Set up the repository and environment

Install and configure necessary tools.

Potential core dependencies may include:

Python
PyTorch
torchvision
NumPy
OpenCV
Pillow
scikit-learn
pandas
matplotlib
Weights & Biases
SQLite/PostgreSQL tooling
Gradio or Streamlit for the demo

Verify exact package compatibility before installation.

Step 3 — Acquire and inspect the dataset

Download a small subset first.

Verify:

Class balance
Image dimensions
File integrity
Generator labels
Train/test separation
Step 4 — Implement and test bit-plane extraction

Create unit-style tests.

Visualize outputs.

Step 5 — Implement BGNIG/noise generation

Follow the paper exactly.

Compare scaling and thresholding.

Visualize results.

Step 6 — Implement MGPS

Verify gradient computation.

Visualize selected patches.

Step 7 — Implement NBC

Train on a small controlled dataset.

Get a baseline.

Step 8 — Implement NGC

Only after NBC is stable.

Step 9 — Establish reproduction results

Compare honestly with the paper.

Document differences.

Step 10 — Run representative LOGO experiments

Use the four diverse generators.

Automate the rotations.

Step 11 — Evaluate trained models on the broader generator set

Generate performance matrices and heatmaps.

Step 12 — Run focused ablations

Only the three locked ablations unless there is a compelling reason for another.

Step 13 — Run robustness evaluation

Use trained models.

Generate degradation curves.

Step 14 — Build experiment database

Ensure previous experiments are captured correctly.

Step 15 — Build constrained experiment query system

Use templates and parameterized SQL.

Step 16 — Build the demo

Visualize prediction and forensic pipeline.

Step 17 — Documentation and final analysis

Write the project like a short research paper.

21. DOCUMENTATION REQUIREMENTS

Maintain a docs/decisions.md file throughout the project.

Every important decision should have:

Decision:
What we chose.

Alternatives:
What we considered.

Reason:
Why we selected this option.

Trade-off:
What we gave up.

Examples:

Decision:
Use 4 representative LOGO rotations instead of 8.

Reason:
Preserve the scientific question while reducing training cost.

Trade-off:
Less exhaustive generator coverage.

Mitigation:
Evaluate final trained models on all available generator test sets.

Also document:

Dataset scale decisions
Normalization decisions
Architecture choices
Generator selection rationale
Why optional components were excluded

This documentation is important for interviews.

22. REQUIRED VISUALIZATIONS

The final project should include high-quality visualizations.

At minimum:

Forensic pipeline visualization
Original Image
→ Bit Planes
→ Noise Representation
→ MGPS Score/Selection
→ Selected Patch
→ Prediction
LOGO results
Generator × model performance matrix
Heatmap
Seen vs unseen performance comparison
Ablations
Bit-plane count vs performance
MGPS comparison
NBC vs NGC comparison
Robustness
JPEG quality vs AUROC/F1
Blur strength vs AUROC/F1
Optional additional transformation curves
Failure analysis

Show representative examples of:

Correct real detection
Correct fake detection
False positives
False negatives

Whenever possible, connect failure analysis to the forensic representation rather than only displaying raw images.

23. EVALUATION METRICS

Use metrics appropriate for binary AI-image detection.

At minimum consider:

Accuracy
Precision
Recall
F1 Score
AUROC

Potentially also:

ROC curves
Precision-Recall curves
Confusion matrices

Do not rely only on accuracy.

Because the research focus includes generalization, emphasize:

Average performance
+
Per-generator performance
+
Seen vs unseen performance
+
Performance degradation
24. HONEST RESEARCH PRACTICES

Do not:

Cherry-pick only good results.
Hide failed experiments.
Claim exact paper reproduction when using a much smaller dataset.
Claim novelty that we did not actually establish.
Claim that one observed result proves causation.
Artificially inflate performance.

If a result is negative, analyze it.

For example:

The NGC architecture did not improve unseen-generator performance under our constrained training setup.

This is still valuable if supported by data.

The project should prioritize:

Scientific honesty
+
Experimental rigor
+
Interpretability

over artificially impressive results.

25. WHAT NOT TO ADD

Unless there is an extremely strong reason, do not add:

Generic chatbot over research papers
General-purpose RAG
LangGraph
Agents
Multi-agent systems
Authentication
User management
SaaS infrastructure
Kubernetes
Complex microservices
Frequency-domain fusion architecture
Multi-branch architecture
Multiple unnecessary backbones
Large-scale hyperparameter optimization
Full arbitrary Text-to-SQL

Potential future work may include:

Frequency-domain fusion
Adaptive bit-plane selection
FGSM/PGD adversarial robustness
Full 8-generator LOGO
CNN vs ViT
More advanced cross-generator/domain adaptation methods

But these should not delay the core project.

26. HOW YOU SHOULD HELP ME

As my project mentor:

Do:
Teach concepts before implementation when necessary.
Explain code line by line when I am learning.
Help me debug systematically.
Challenge weak experimental assumptions.
Suggest the smallest experiment that answers a question.
Help optimize GPU/compute usage.
Maintain reproducibility.
Help analyze failed experiments.
Help write the README and report.
Help prepare interview explanations.
Do not:
Give me huge amounts of code without explanation.
Assume results without seeing them.
Add technologies for buzzword value.
Tell me something is novel without justification.
Agree with my idea simply because I proposed it.
Overcomplicate a simple problem.

Whenever we consider a new component, evaluate:

Scientific Value:
?

Learning Value:
?

Resume Value:
?

Implementation Effort:
?

Compute Cost:
?

Risk:
?

Final Recommendation:
Add / Skip / Future Work
27. FINAL SUCCESS CRITERIA

By the end, I should have:

Research
A working implementation of the paper's core methodology.
A documented reproduction under realistic compute constraints.
A controlled cross-generator generalization study.
Representative LOGO experiments.
Broader generator evaluation.
Three focused ablation studies.
Real-world robustness analysis.
Failure analysis and honest limitations.
Deep Learning knowledge

I should understand and be able to explain:

CNN fundamentals
Feature extraction
Transfer learning/backbones if used
Bit-plane representation
Image preprocessing
Gradient-based patch selection
Attention mechanism used in NGC
Training and optimization
Generalization
Distribution shift
Overfitting
Evaluation metrics
Ablation studies
Robustness
Engineering
Clean repository
Reproducible experiments
Configuration-driven training
W&B experiment tracking
Experiment database
Automated evaluation
Clean visualizations
Functional demo
Placement relevance

The project should demonstrate:

Deep Learning
+
Computer Vision
+
Research Thinking
+
Experimental Design
+
SQL / Data Modeling
+
LLM-assisted Experiment Analysis
+
Software Engineering

without becoming a kitchen-sink project.

28. HOW TO BEGIN

I will now provide:

The LOTA research paper.
Any information about my available compute.
The number of people in my team, if relevant.

Your first task should be:

Part 1 — Analyze the paper deeply

Create:

A concise paper summary.
The exact problem being solved.
The complete methodology.
Every major component.
Every equation explained intuitively.
Input/output of every module.
Architecture/data-flow diagram.
Hyperparameters and training details.
Dataset details.
Results reported by the paper.
Implementation ambiguities or missing details.
Part 2 — Convert the paper into an implementation plan

Before writing the full model, provide:

Module
Purpose
Inputs
Outputs
Dependencies
Implementation difficulty
Testing strategy

for every major module.

Part 3 — Identify exact resources

Find and verify:

Official paper repository, if available
Official GenImage dataset location
Dataset documentation
Any supplementary material
Required pretrained weights/backbones
Relevant official implementation resources

Clearly distinguish:

Information directly from the paper
Information from official project resources
Your own engineering recommendations
Part 4 — Ask only the necessary setup questions

Do not ask unnecessary questions.

The important questions are likely to include:

Available GPU/compute
Team size
Available storage
Whether we have a strict course submission requirement
Whether the paper provides official code
Any deadline constraints only if they affect scope

Then help me begin with the smallest correct implementation step, not the entire project at once.

IMPORTANT FINAL INSTRUCTION

Treat this as a serious research-oriented Deep Learning project.

The final objective is not to maximize the number of technologies used.

The objective is to produce a project where I can confidently say:

"I reproduced a recent AI image forensics method, understood its low-bit-plane detection mechanism, systematically studied its generalization to unseen generators under controlled experiments, investigated robustness and component importance through ablations, and built a lightweight experiment intelligence system grounded in the actual results of my research."

At every stage, prioritize:

Correctness → Understanding → Experimental Rigor → Reproducibility → Clear Communication → Additional Features

Do not move to advanced features until the previous layer is working and understood.