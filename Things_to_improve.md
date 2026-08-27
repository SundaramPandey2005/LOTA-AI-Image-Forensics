Corrective Implementation Directive — LOTA Project Iteration 2

You are continuing work on an existing Deep Learning Computer Vision research project implementing and extending LOTA: Bit-Planes Guided AI-Generated Image Detection (ICCV 2025).

The first iteration of the implementation has been reviewed. Do not blindly continue adding features on top of the existing code. First perform a structured audit and correction of the current implementation.

The objective of this iteration is:

Transform the existing codebase into a faithful, reproducible, compute-conscious implementation of the LOTA research pipeline, with a clean path toward the planned generalization, ablation, robustness, experiment-intelligence, and demo extensions.

The project is intended to be both:

A serious Deep Learning / Computer Vision research project based on a published paper.
A strong Data Science / Machine Learning portfolio project demonstrating end-to-end experimentation, evaluation, reproducibility, and engineering discipline.

Do not optimize for adding the maximum number of technologies. Optimize for:

Research correctness × reproducibility × effort-to-reward ratio × interview defensibility.

1. CRITICAL RULE: AUDIT BEFORE MODIFYING

Before writing major new code:

Inspect the entire existing repository.
Compare the implementation against:
the attached LOTA research paper,
the official LOTA repository: https://github.com/hongsong-wang/LOTA,
the project decisions and requirements described below.
Create an audit document:
docs/iteration1_audit.md

The audit must contain a table like:

Component	Current Status	Correct / Incorrect	Issue	Required Action

Audit at minimum:

project structure
dependencies
bit-plane extraction
bit-plane composition
normalization
MGPS implementation
patch selection
dataset loading
real/fake labels
train/validation/test splits
generator separation
preprocessing
NBC implementation
NGC implementation
ResNet usage
training loop
metrics
evaluation logic
experiment logging
database
query system
demo
tests
configuration system

Do not silently rewrite everything.

Preserve components that are correct.

Fix, simplify, or replace components that are incorrect.

2. AUTHORITATIVE REFERENCE HIERARCHY

Use the following hierarchy when resolving implementation details:

Priority 1 — Research paper

The paper is the primary source for:

mathematical definitions,
methodology,
architecture,
experiments,
research claims.
Priority 2 — Official LOTA repository

Use the official repository only to clarify implementation details that are ambiguous or underspecified in the paper.

Examples:

interpolation method,
exact preprocessing order,
augmentation,
optimizer settings,
scheduler,
tensor formatting,
implementation details not explicitly described.

Do not blindly copy code.

When the official repository is used to resolve ambiguity, document this in:

docs/decisions.md

Example:

## Resize interpolation

Decision: Bilinear interpolation.

Reason: The paper does not explicitly specify the interpolation mode. The official LOTA implementation uses bilinear resizing.

Source: Official repository implementation.

Every major decision should indicate whether it came from:

Paper
Official repository clarification
Our experimental decision
3. DO NOT ASSUME THE CURRENT IMPLEMENTATION IS CORRECT

The previous implementation may contain technically plausible but incorrect assumptions.

Explicitly verify:

Bit-plane decomposition

For an 8-bit image:

$$ x^c = \sum_{k=0}^{7}2^k x_k^c $$

The implementation must correctly extract individual binary bit planes.

Then verify the low-bit composition used by LOTA:

$$ z^c = 4x_2^c + 2x_1^c + x_0^c $$

Do not accidentally:

normalize before extracting bits,
extract bits from floating-point normalized tensors,
reverse bit ordering,
treat RGB channels incorrectly.

Add synthetic unit tests where known integer values verify exact bit recovery.

4. NORMALIZATION MUST BE VERIFIED

Implement and test the normalization methods described in the paper.

Ensure the implementation clearly distinguishes:

Threshold normalization
Scaling normalization

Do not assume one is correct without verification.

The default should follow the experimental setup used in the paper or the official implementation where appropriate.

Document the choice.

5. MGPS MUST BE REIMPLEMENTED AND VERIFIED CAREFULLY

MGPS is a core contribution of the project.

Audit the current implementation against the exact mathematical formulation in the paper.

Verify:

gradient kernels,
all directional gradients,
convolution behavior,
aggregation/scoring,
patch candidate generation,
boundary handling,
patch size,
argmax selection.

The default patch size is:

32 × 32

unless the paper or official implementation specifies additional conditions.

The implementation must not merely “look reasonable.”

Create controlled synthetic tests:

flat image,
horizontal edge,
vertical edge,
diagonal edge,
multiple high-gradient regions.

Verify that MGPS selects the expected region.

Visualize:

Input
→ Bit-plane composition
→ Normalized noise representation
→ Gradient map
→ Selected patch overlay

Save these as reproducible validation outputs.

6. FIX THE DATA PIPELINE BEFORE LARGE-SCALE TRAINING

Before training any model, explicitly verify the dataset structure.

The project will use the GenImage dataset or an appropriate subset of it.

The implementation must verify:

where real images come from,
where fake images come from,
whether real images are shared across generator subsets,
whether image IDs overlap,
whether train/test splits are predefined,
whether any real-image leakage occurs across experiments.

Document this in:

docs/experiment_protocol.md

Do not assume that each generator has an independent real-image dataset.

This is particularly important for interpreting LOGO experiments.

The dataset loader must support:

generator_name
split
label
image_path

and ideally maintain explicit metadata.

7. PREVENT DATA LEAKAGE

Before implementing LOGO experiments, add explicit safeguards.

The system must ensure:

Training

The held-out generator's fake images are never used for training.

Evaluation

Seen and unseen generator evaluations are clearly labeled.

Real images

Document how real images are shared or separated.

Splits

No accidental overlap between:

train,
validation,
test.

If image identifiers are available, implement duplicate/overlap checks.

Create a validation script such as:

scripts/check_data_integrity.py

It should report:

number of images,
class balance,
generator counts,
duplicate image identifiers if detectable,
split overlap.
8. MODEL IMPLEMENTATION MUST MATCH THE PAPER

Audit NBC and NGC separately.

NBC

Verify:

correct input representation,
correct backbone usage,
ResNet-50 initialization,
input resizing,
classification head,
output dimensions.

Avoid unnecessary architectural modifications.

NGC

This must not be implemented as a generic “attention layer.”

Audit the implementation against the paper's noise-guided architecture.

Verify:

raw image feature extraction,
noise representation,
feature projection,
cross-attention,
spatial alignment,
multi-head attention if specified,
residual connections,
final classification pipeline.

If the current NGC implementation is speculative or substantially different from the paper, do not pretend it is a reproduction.

Clearly document:

Faithful reproduction

vs.

Our approximation / implementation decision

NBC must be fully functional and validated before NGC becomes a priority.

Do not debug both simultaneously.

9. COMPUTE-CONSCIOUS DEVELOPMENT STRATEGY

We are using:

local CPU development for logic and tests,
free Google Colab / Kaggle GPUs for training,
likely T4 or P100-class GPUs.

Do not invent final training-time estimates before testing.

Use a staged strategy.

Stage A — Development scale

Very small dataset.

Purpose:

verify dataset loading,
verify preprocessing,
verify MGPS,
verify model forward/backward pass,
detect memory issues.
Stage B — Pilot scale

Run one representative training experiment.

Measure:

GPU memory usage,
batch size,
epoch duration,
convergence behavior,
checkpoint size,
total runtime.

Record actual measurements.

Stage C — Lock final experimental scale

Only after the pilot succeeds, decide:

images per generator,
train/validation/test counts,
batch size,
epochs,
image resolution,
backbone,
optimizer,
scheduler,
GPU environment.

Document this in:

docs/experiment_protocol.md

Also record:

GPU type
VRAM
batch size
dataset size
epoch duration
total runtime

The final project must honestly state that the paper used a much larger compute scale if our reproduction uses a subset.

We are reproducing:

methodology and relative experimental behavior under compute constraints

not falsely claiming exact reproduction of the paper's absolute metrics.

10. COMPUTE COST CLASSIFICATION

Every proposed experiment should be classified before running.

Category A — Cheap / High Reward

Examples:

visualization,
unit tests,
preprocessing checks,
inference-only robustness evaluation,
full-generator evaluation of already-trained models,
additional SQL queries,
experiment plots.

These should generally be prioritized.

Category B — Moderate Cost / Strong Research Value

Examples:

the 4 LOGO training rotations,
core ablations,
NGC comparison.

These are important but must be planned carefully.

Category C — Expensive / Optional

Examples:

full 8-generator LOGO,
extensive hyperparameter sweeps,
ViT comparison,
adversarial PGD/FGSM studies,
entirely new multi-branch architectures.

Do not add Category C work unless the core project is complete.

This classification should be documented in:

docs/experiment_protocol.md
11. CORE REPRODUCTION COMES FIRST

The implementation sequence must be:

Step 1

Paper understanding and mathematical validation.

Step 2

Bit-plane extraction.

Step 3

Low-bit composition.

Step 4

Normalization.

Step 5

MGPS.

Step 6

Tiny dataset pipeline.

Step 7

Tiny NBC training run.

Step 8

Pilot experiment.

Step 9

Lock compute scale.

Step 10

Larger reproduction.

Only after this should NGC and the research extensions be expanded.

Do not jump directly to LOGO experiments because the code exists.

12. REPRESENTATIVE LOGO GENERALIZATION BENCHMARK

Our primary extension is a representative 4-generator Leave-One-Generator-Out benchmark.

The purpose is to study:

How well does AI-generated image detection generalize when the fake image generator is completely unseen during training?

Select four generators representing different families.

The intended set is:

BigGAN — GAN archetype
Stable Diffusion v1.4 — latent diffusion archetype
Midjourney — high-fidelity proprietary generator and stress case
ADM or GLIDE — architecturally distinct diffusion model

Before locking the exact fourth generator, verify dataset availability and rationale.

Document the diversity rationale.

For each rotation:

Train on the remaining selected generators
Hold out one selected generator
Evaluate on:
    - seen generators
    - held-out generator
    - all available generator test sets

Important:

The held-out generator must not contribute fake training data.

Make the LOGO experiment configuration-driven.

Example:

excluded_generator: biggan
train_generators:
  - sd14
  - midjourney
  - adm

Do not create four manually duplicated training scripts.

13. FULL CROSS-GENERATOR EVALUATION

After each LOGO model is trained:

Evaluate it against all available generator test sets.

This is inference-only and therefore relatively cheap.

Generate:

cross-generator matrix,
heatmap,
seen vs unseen comparison,
average performance,
hardest generator analysis.

Do not claim full 8-generator LOGO.

Clearly state:

We performed 4 representative LOGO rotations and evaluated each trained model across all available generator test sets.

14. CONTINGENCY FOR INSUFFICIENT COMPUTE

If the pilot demonstrates that the planned 4-generator LOGO experiments are infeasible:

Do not silently reduce dataset size until results become unstable.

The fallback order is:

Preferred fallback

Reduce from:

4 representative generators

to:

3 carefully selected representative generators

while preserving diversity across generator families.

Document:

why the reduction occurred,
compute measurements,
final generator selection rationale.

Only reduce dataset size per generator if metrics remain statistically and experimentally stable.

Never silently change the experimental scale.

15. RANDOM SEEDS AND REPRODUCIBILITY

Reproducibility must be explicitly implemented.

Fix and log:

Python seed,
NumPy seed,
PyTorch CPU seed,
PyTorch CUDA seed,
DataLoader worker seed,
dataset shuffle seed,
model initialization seed.

Add a reproducibility utility:

src/utils/reproducibility.py

Each experiment configuration should contain:

seed: 42

Log the seed into:

W&B,
SQLite,
experiment configuration,
checkpoint metadata.

Document any unavoidable nondeterministic GPU operations.

16. LOCKED ABLATION STUDIES

Do not expand the ablation suite unnecessarily.

The locked ablations are:

Ablation A — Bit-plane depth

Evaluate progressively:

0
0–1
0–2
0–3
0–4
0–5

subject to the exact interpretation in the paper.

Ablation B — Patch selection

Compare:

MGPS maximum-gradient patch
random patch
minimum-gradient patch
center patch

This is cheap because it reuses existing infrastructure.

The objective is:

Evaluate whether maximum-gradient MGPS selection provides an advantage.

Do not phrase success criteria as:

Verify MGPS strictly outperforms every alternative.

Negative or mixed results are valid findings.

Ablation C — Classifier

Compare:

linear/simple baseline where appropriate,
NBC,
NGC.

Only perform this after NBC and NGC are individually validated.

17. ROBUSTNESS EXPERIMENTS

Robustness is a core high-effort-to-reward component because trained models can be reused.

Implement:

JPEG compression
Q = 100, 95, 90, 85
Gaussian blur
σ = 0, 1, 2, 3

Additional cheap transforms may include:

resize,
crop,
additive noise.

However, do not add excessive robustness dimensions.

Report:

AUROC degradation,
accuracy degradation,
generator-specific behavior where useful.

Use already-trained models.

No retraining should be required for these robustness tests.

18. EXPERIMENT DATABASE

The project should maintain a lightweight SQLite database.

Do not build SaaS infrastructure.

No authentication.

No users table.

No multi-user platform.

The database should store:

Experiments
experiment_id
timestamp
experiment type
seed
configuration reference
model
Configurations
backbone
bit-plane setup
patch strategy
training parameters
Generators
generator name
generator family
Metrics
accuracy
AUROC
AP
F1
precision
recall
Robustness
transform type
severity
degradation metrics

Ensure every experiment can be traced back to its configuration.

Use parameterized SQL queries.

19. EXPERIMENT INTELLIGENCE LAYER

This is not a generic RAG chatbot.

Do not build:

“Chat with AI research papers.”

Do not build open-ended text-to-SQL.

Instead, implement a small controlled natural-language query interface.

Support approximately 8–12 useful query intents.

Examples:

Which model achieved the highest AUROC on generator X?
Which generator was hardest to detect?
Which model degraded most under JPEG compression?
What was the largest seen-vs-unseen generalization gap?
Which bit-plane configuration performed best?
How did NGC compare with NBC?

Architecture:

Natural language question
        ↓
Deterministic/template intent routing
        ↓
Parameter extraction
        ↓
Parameterized SQL query
        ↓
Structured result
        ↓
Grounded explanation

The LLM may explain the structured result.

The LLM must not freely generate arbitrary SQL.

If qualitative experiment notes become substantial later, vector retrieval may be added.

Otherwise, do not add vector databases merely to include RAG.

20. INTERACTIVE DEMO

Build only after the research pipeline works.

The demo should have four focused sections.

Tab 1 — Image Forensics
Upload image
→ visualize bit-plane information
→ noise representation
→ MGPS gradient map
→ selected patch
→ prediction
→ confidence
Tab 2 — Generalization

Show:

LOGO results,
cross-generator heatmap,
seen vs unseen comparison.
Tab 3 — Robustness and Ablations

Show:

JPEG curves,
blur curves,
ablation comparisons.
Tab 4 — Experiment Intelligence

Controlled natural-language questions about the experiment database.

Do not add:

authentication,
user profiles,
dashboards unrelated to experiments,
SaaS features.

This is a research demonstration tool, not a startup platform.

21. TESTING REQUIREMENTS

Maintain automated tests.

At minimum:

Bit-plane tests

Verify exact extraction using known integers.

Reconstruction tests

Verify correct bit composition.

MGPS tests

Test synthetic gradients and expected patch selection.

Dataset tests

Verify:

labels,
shapes,
generator metadata,
class balance.
Model tests

Verify NBC and NGC forward-pass shapes.

Database tests

Verify:

insertion,
retrieval,
parameterized queries.
Data integrity tests

Verify:

no split overlap,
held-out generator exclusion,
duplicate detection where possible.
22. METRICS

Track at minimum:

Accuracy
AUROC
Average Precision
F1
Precision
Recall

AUROC and Average Precision should be treated as important metrics for robust comparison, especially when discussing cross-generator generalization.

Do not over-focus on accuracy alone.

23. HONEST REPRODUCTION POLICY

The original paper may use significantly more data and compute.

Our project must never imply:

We reproduced the exact published numbers.

unless that is genuinely achieved under comparable conditions.

Instead report:

Paper setting:
[dataset scale / compute]

Our setting:
[dataset subset / GPU / training parameters]

Then discuss:

absolute metric gap,
whether relative trends are reproduced,
likely reasons for differences,
limitations.

Unexpected or negative results are not failures.

If:

MGPS does not outperform an alternative,
NGC does not outperform NBC,
generalization is inconsistent,

report the result honestly and investigate plausible causes.

Do not tune endlessly until the result matches the paper.

24. DECISION LOG

Maintain:

docs/decisions.md

Every important decision should contain:

## Decision

### Context
What problem or ambiguity existed?

### Options considered
- Option A
- Option B

### Decision
Chosen approach.

### Reason
Why this was chosen.

### Evidence
Paper / official repo / pilot experiment / compute measurement.

### Impact
How this affects reproducibility or experiments.

Examples include:

normalization choice,
dataset subset scale,
generator selection,
seed,
interpolation,
optimizer,
scheduler,
patch size,
LOGO scope.
25. REQUIRED PROJECT STATE AFTER THIS ITERATION

Do not attempt to finish every feature immediately.

The immediate target state is:

Fully validated
bit-plane extraction
low-bit composition
normalization
MGPS
synthetic tests
dataset integrity checks
tiny dataset pipeline
NBC forward/backward training
reproducibility utilities
configuration system
Pilot validated
one generator or representative multi-generator setup
actual GPU memory usage
actual epoch duration
convergence behavior
Documented
audit report
paper notes
decisions
experiment protocol
compute classification
data leakage analysis
Only after this

Proceed to:

final-scale reproduction,
4-generator LOGO,
full cross-generator evaluation,
locked ablations,
robustness,
experiment database,
intelligence query system,
demo.
26. WHAT NOT TO DO

Do not add any of the following unless explicitly requested later:

generic RAG over papers,
arbitrary NL-to-SQL,
vector database without qualitative retrieval needs,
authentication,
users table,
SaaS architecture,
multi-agent systems,
LangGraph,
unnecessary APIs,
frequency-domain fusion architecture,
CNN + ViT fusion,
adversarial FGSM/PGD experiments,
full 8-generator LOGO,
large hyperparameter sweeps.

These are either:

unrelated to the core research question,
expensive relative to reward,
or better treated as future work.

A separate project will cover genuinely agentic AI and LangGraph concepts, so do not attempt to force them into this project.

27. PLACEMENT / CV SANITY CHECK

This project should remain clearly defensible as:

Deep Learning + Computer Vision + Research Reproduction + Experimental Generalization + Robustness + Reproducible ML Engineering.

Periodically flag any implementation decision that would weaken this narrative or turn the project into a collection of unrelated technologies.

Do not add a component just because it is popular in placements.

Every component must answer:

What real research or engineering problem does this solve?

If there is no strong answer, do not add it.

28. WORKING STYLE

Act as a critical senior ML engineer and research mentor.

Do not simply agree with proposed ideas.

If something is:

mathematically incorrect,
inconsistent with the paper,
computationally wasteful,
likely to introduce leakage,
unnecessary for the project,
difficult to defend in an interview,

explicitly say so.

Prioritize:

Correctness
↓
Reproducibility
↓
Research value
↓
Effort-to-reward ratio
↓
Engineering polish
↓
Additional features
IMMEDIATE NEXT ACTION

Do not begin LOGO experiments or large-scale training yet.

First:

Audit the current repository.
Create docs/iteration1_audit.md.
Fix only the issues identified by the audit.
Verify the forensic pipeline mathematically with synthetic tests.
Verify data integrity and potential leakage.
Implement reproducibility controls.
Run a tiny NBC end-to-end experiment.
Run one pilot GPU experiment.
Measure actual compute behavior.
Update experiment_protocol.md.
Present the results and proposed final experimental scale before proceeding to large-scale experiments.

At each major checkpoint, summarize:

what was implemented,
what was verified,
what failed,
what assumptions remain,
what should be done next.

Do not continue blindly. Validate before scaling.




CRITICAL ADDENDUM — FABRICATED DATA AND REAL-DATA VISUALIZATION REQUIREMENTS

This addendum introduces two explicit requirements that must be treated as high-priority audit items during Iteration 2.

1. CRITICAL: Audit scripts/seed_database.py for fabricated experiment results

The existing codebase may contain:

scripts/seed_database.py

This file must be audited explicitly.

Critical rule

If this script inserts:

fabricated metrics,
invented AUROC values,
invented accuracy values,
placeholder experimental results,
manually created numbers,

and those values are labeled, stored, displayed, or otherwise presented in a way that could imply they are:

real experimental results,
reproduced results,
ICCV 2025 paper results,
or results actually obtained from training,

then this is a critical integrity issue.

Do not allow fabricated numbers to remain mixed with real experimental evidence.

Required action

Audit the script and classify it into one of the following cases.

Case A — The script inserts fabricated/example metrics

If the purpose is only to test the database schema or demo interface:

Rename the script clearly, for example:
scripts/create_mock_data.py

or:

scripts/seed_test_fixture.py
Clearly label all inserted records as:
MOCK
SYNTHETIC
TEST ONLY
NOT REAL EXPERIMENTAL RESULTS

where appropriate in the schema and metadata.

Ensure mock data cannot accidentally appear in:
research plots,
benchmark tables,
experiment analysis,
final reports,
README results,
CV claims,
or the main research demo.
Add an explicit database field or experiment flag if necessary:
is_mock: true

or equivalent.

The production analysis pipeline must exclude mock records by default.
Case B — The script attempts to represent paper results

If the numbers were manually copied or approximated from the LOTA paper:

Do not store them as if they are our own experimental results.

If retaining them is useful for comparison, create a clearly separate representation such as:

source = "LOTA Paper"
result_type = "published_reference"

These values must remain clearly separated from:

source = "Our Experiment"
result_type = "experimental"

Never mix paper reference metrics and our experimental metrics in the same analysis without explicit source labeling.

Case C — The script has no legitimate ongoing purpose

Delete it.

Do not preserve unnecessary fake-result generation simply because it already exists.

Final requirement

After the audit, document the decision in:

docs/decisions.md

and record the issue in:

docs/iteration1_audit.md

This project must maintain a strict distinction between:

Published paper results
        ≠
Our experimentally measured results
        ≠
Mock/test data

At no point should fabricated or placeholder metrics be capable of being mistaken for real experimental evidence.

This is a non-negotiable research integrity requirement.

2. REAL GENIMAGE VISUALIZATION REQUIREMENT

The mathematical and synthetic tests for:

bit-plane extraction,
normalization,
MGPS,
gradient responses,
patch selection,

must remain part of the validation suite.

However, synthetic visualizations alone are not sufficient for the final research project.

Once a minimal real GenImage subset is available, generate the complete forensic visualization pipeline using actual dataset samples.

At minimum, select:

One real image from GenImage.
One AI-generated image from GenImage.

For both samples, generate:

Original Image
        ↓
Relevant Bit Planes
        ↓
Low-Bit Composition
        ↓
Normalized Noise Representation
        ↓
MGPS Gradient / Importance Map
        ↓
Selected Patch Overlay

Save the outputs in a reproducible location and make the visualization generation script/notebook reusable.

For example:

notebooks/01_lota_mathematical_validation.ipynb

or a dedicated visualization script.

Qualitative comparison with the paper

Compare the resulting real-vs-fake noise representations qualitatively with the corresponding visualization and observations in the LOTA paper, including its Figure 3 or equivalent relevant figure.

The purpose is not to force our visual output to look identical to the paper.

Instead, evaluate whether similar qualitative behavior is observed at our dataset scale.

Document:

similarities,
differences,
possible causes,
preprocessing differences,
dataset/generator differences,
limitations caused by our compute-constrained subset.

Do not make unsupported claims such as:

"Our results perfectly reproduce Figure 3."

Instead use evidence-based language such as:

"Our samples exhibit qualitatively similar/different low-bit noise characteristics compared with the observations reported in the paper."

If the expected qualitative distinction does not appear clearly, treat this as a debugging or research question rather than hiding it.

Investigate:

bit extraction correctness,
normalization,
image preprocessing,
data type conversion,
RGB/channel handling,
dataset selection,
real/fake sample provenance.

The final demo and documentation should use real GenImage examples, not synthetic placeholder images, for illustrating the actual forensic pipeline.

PRIORITY

These two items must be addressed during the initial audit and correction phase, before:

large-scale training,
LOGO experiments,
final experiment analysis,
or presentation of any results.

The required order is:

1. Audit current repository
        ↓
2. Explicitly audit seed_database.py
        ↓
3. Remove/separate any fabricated or misleading metrics
        ↓
4. Verify mathematical pipeline with synthetic tests
        ↓
5. Verify pipeline visually on real + AI-generated GenImage samples
        ↓
6. Document qualitative observations
        ↓
7. Continue with tiny training and pilot experiments

Do not proceed with the project while the provenance of experiment data is ambiguous.

Research result provenance, reproducibility, and correctness take priority over demo polish or feature development.