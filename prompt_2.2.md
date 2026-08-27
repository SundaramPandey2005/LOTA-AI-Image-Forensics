LOTA Project — Real-Data Readiness Gate and Transition to Experimental Phase

The previous implementation iterations are complete enough to move toward real experimentation, but do not begin full reproduction, LOGO rotations, ablations, robustness experiments, or any expensive training yet.

Before entering the main experimental phase, implement the following final Real-Data Readiness Gate.

This is a targeted transition step, not another architecture redesign or broad audit.

The objective is:

Prove that the complete LOTA pipeline works correctly on real GenImage data, measure actual compute behavior, and lock the experimental scale based on evidence rather than estimates.

After completing this work, stop and provide a report for review before proceeding to large-scale experiments.

1. Remove All Silent Mock-Data Fallbacks
Critical research-integrity requirement

Review:

src/data/dataset.py

and any related dataset-loading code.

The current implementation may automatically switch to mock/synthetic data when:

the GenImage dataset directory does not exist, or
no samples are discovered.

This behavior must be removed.

Required behavior

Mock data must only be used when explicitly requested:

use_mock_data=True

If:

use_mock_data=False

then the system must never silently substitute synthetic data.

Instead, fail loudly with clear errors.

For example:

RuntimeError: GenImage dataset was not found.

Expected path:
<data_path>

Real-data training cannot proceed.

If you intentionally want to run a development test with synthetic data,
set use_mock_data=True explicitly.

Similarly, raise explicit errors if:

the dataset directory exists but contains no valid samples,
real samples are missing,
fake samples are missing,
generator-specific directories cannot be resolved,
the dataset is severely imbalanced beyond configured expectations.

Do not silently continue.

2. Clearly Separate Mock Infrastructure Testing from Real Experimental Testing

The project should now have two explicitly different readiness gates.

Gate A — Infrastructure Gate

This gate may use synthetic/mock data.

Purpose:

verify Python environment,
dataset class construction,
model initialization,
forward pass,
backward pass,
optimizer,
AMP,
checkpointing,
experiment logging,
database integration.

The result should be reported as:

INFRASTRUCTURE GATE: PASSED

Passing this gate means only:

The software infrastructure is operational.

It must not imply readiness for real experiments.

Gate B — Real-Data Pilot Gate

This gate requires actual GenImage images.

It must validate:

real dataset discovery,
real/fake label correctness,
image loading,
preprocessing,
bit-plane extraction,
BGNIG generation,
normalization,
MGPS patch selection,
dataloader throughput,
GPU/CPU memory behavior,
model training,
validation metrics,
experiment logging.

The result should only be reported as:

REAL-DATA PILOT GATE: PASSED

This is the gate that unlocks the full reproduction experiment.

3. Update run_pilot_check.py

Review:

scripts/run_pilot_check.py

The current script should be refactored so that it does not present a mock-data training run as proof that the system is ready for full experiments.

Implement one of the following clean approaches:

Preferred approach

Split the pilot into:

scripts/run_infrastructure_check.py

and:

scripts/run_real_data_pilot.py
Infrastructure check

Can explicitly use:

use_mock_data=True

Output:

INFRASTRUCTURE GATE: PASSED
Real-data pilot

Must explicitly use:

use_mock_data=False

and fail if GenImage is unavailable.

Output should include:

REAL-DATA PILOT GATE
Dataset: <dataset>
Generator: <generator>
Real samples discovered: X
Fake samples discovered: X
Image size: 256x256
Batch size: X
Device: <device>
Mixed precision: enabled/disabled

After the pilot, report:

Training time
Validation time
Images/sec
Approximate GPU memory usage if available
Final training loss
Final validation loss
Accuracy
AUROC
Average Precision

Do not claim that the pilot reproduces the paper's performance.

The purpose is to validate the real pipeline and estimate compute.

4. Add a Proper Real GenImage Dataset Readiness Check

Create or improve a script such as:

scripts/check_data_integrity.py

or:

scripts/check_genimage_ready.py

The script should inspect the configured dataset path and report:

GENIMAGE REAL-DATA READINESS CHECK

It should verify:

Dataset structure
dataset root exists,
expected generator directories exist,
real-image directories exist,
fake-image directories exist.
Sample availability

Report:

Generator: SD v1.5
Real samples: X
Fake samples: X

Do this for every generator discovered.

Image validity

Attempt to load a small sample from each class and verify:

valid image file,
correct dimensions after preprocessing,
no corrupted files.
Class balance

Report class counts and imbalance.

Leakage awareness

For the initial pilot, clearly document whether the real images are shared across generators or generator-specific.

Do not claim a leakage problem unless one is actually detected. Simply report the dataset structure and pairing behavior accurately.

Final status

Example:

REAL-DATA VALIDATION STATUS: READY

or:

REAL-DATA VALIDATION STATUS: NOT READY
Reason: SD v1.5 fake images not found.

This check must fail loudly rather than falling back to mock data.

5. Prepare the Project for a Minimal Real GenImage Subset

Do not download or require the full GenImage dataset yet.

The immediate goal is to obtain a small legitimate subset suitable for:

real vs AI-generated forensic visualization,
dataset integrity testing,
BGNIG/MGPS validation,
tiny NBC pilot training,
compute measurement.

The initial recommended target is approximately:

100–500 real images
100–500 AI-generated images

from one generator initially.

The preferred first generator is:

Stable Diffusion v1.5

because it aligns naturally with the planned reproduction baseline.

Document clearly in:

README.md
docs/experiment_protocol.md

the exact expected local directory structure.

For example, adapt to the actual GenImage structure rather than inventing one:

data/
└── GenImage/
    └── <generator>/
        ├── real/
        └── fake/

Important: determine the actual structure from the dataset source or downloaded subset. Do not hardcode an assumed structure if the real dataset differs.

6. Add a Configuration for the Real-Data Pilot

Create a dedicated configuration such as:

configs/pilot_real_data.yaml

The configuration should contain fields such as:

experiment_name: real_data_pilot
generator: sd_v1_5

data:
  root_dir: ./data/GenImage
  use_mock_data: false
  max_real_samples: <configurable>
  max_fake_samples: <configurable>

model:
  type: nbc
  backbone: resnet50
  input_size: 256

training:
  batch_size: <configurable>
  epochs: <small pilot value>
  learning_rate: <existing/default value>
  mixed_precision: true

reproducibility:
  seed: <fixed seed>

Do not hardcode experimental values unnecessarily.

The pilot should be easy to adjust after observing memory and throughput.

7. Run Real Forensic Validation Before Training

Once a minimal real GenImage subset is available, rerun:

notebooks/01_lota_mathematical_validation.ipynb

or the equivalent visualization script.

Use:

one real GenImage sample,
one AI-generated sample from the selected generator.

The visualization must include:

REAL SAMPLE
Dataset: GenImage
Generator context: <appropriate label>

AI-GENERATED SAMPLE
Dataset: GenImage
Generator: <generator>

For both images, visualize:

Original Image
↓
Bit-Plane Decomposition
↓
Low-Bit Composition / BGNIG Representation
↓
Normalization
↓
MGPS Gradient Map
↓
Selected Patch

Save the output under a clearly real-data-specific filename, for example:

forensic_decomposition_genimage_real_vs_fake.png

Do not overwrite the synthetic validation artifact.

8. Keep Synthetic and Real Validation Completely Separate

The project should now support two valid modes.

Synthetic validation

Purpose:

Mathematical correctness and controlled behavior.

Examples:

bit recovery,
directional gradients,
MGPS patch selection,
localized artifacts.

Allowed conclusions:

The implementation behaves as expected under controlled conditions.

Real-data validation

Purpose:

Validate that the complete pipeline executes correctly on actual GenImage samples.

Allowed conclusions should remain careful.

For example:

"This sample exhibits a qualitative pattern that can be compared with the type of behavior illustrated in Figure 3."

Do not make statistical or general claims based on one or two images.

9. Implement a Small Real NBC Pilot

After real-data readiness and visualization succeed, run a deliberately small pilot experiment.

Use:

Model: NBC
Generator: SD v1.5
Dataset: small balanced real/fake subset
Image size: 256×256
Epochs: small pilot run

The purpose is not to achieve the paper's reported performance.

The purpose is to measure:

Compute
training time per epoch,
validation time,
images/sec,
approximate memory usage,
batch size limits.
Learning behavior
training loss trend,
validation loss trend,
accuracy,
AUROC,
AP.
Pipeline correctness

Confirm that:

data is real,
BGNIG runs correctly,
MGPS runs correctly,
selected patches are valid,
training loss behaves sensibly,
metrics are not obviously broken.

Store the result as:

source_type = experimental
is_mock = 0

This should become the project's first genuine experimental result.

10. Compute Scale Decision Gate

After the real NBC pilot completes, do not immediately start LOGO.

First generate a short pilot report containing:

GPU/device:
Dataset size:
Images per class:
Batch size:
Image resolution:
Epochs:
Time per epoch:
Total training time:
Peak/approximate memory usage:
Training throughput:
Validation throughput:

Based on this real measurement, propose a final experimental scale.

The proposed scale should specify:

Samples per generator
Real samples per generator
Fake samples per generator
Train/validation/test strategy
Batch size
Epoch count
Expected runtime per experiment
Expected runtime for:
    - reproduction
    - 4-generator LOGO
    - 3 locked ablations

Classify experiments according to the existing effort-to-reward framework:

Category A — Cheap / high reward

Must include the most valuable experiments.

Category B — Moderate cost / meaningful research value

Run if compute allows.

Category C — Expensive / optional

Do not run automatically.

The 4-generator LOGO benchmark remains the main research contribution, but it should only be committed to after the pilot demonstrates that the chosen scale is feasible.

11. Compute Contingency

If the real pilot shows that the intended 4-generator LOGO experiment is infeasible with available compute:

Do not silently shrink every dataset to an unreasonably tiny size.

Instead, apply the existing fallback strategy:

4 representative generators
        ↓ if infeasible
3 representative generators

Maintain diversity across generator families and document the reason.

The fallback must be recorded in:

docs/decisions.md
docs/experiment_protocol.md
12. Documentation Updates

Update:

docs/decisions.md
docs/experiment_protocol.md
README.md

to reflect:

Mock-data policy

Mock data is only activated explicitly.

Gate definitions

Infrastructure Gate ≠ Real-Data Pilot Gate.

Real-data requirement

Large-scale experiments cannot begin until the Real-Data Pilot Gate passes.

Experimental scale

Do not lock the final dataset scale until real pilot measurements exist.

13. Scope Boundaries

This is the final readiness step before the actual research experiments.

Do NOT:

redesign the LOTA architecture,
modify BGNIG without evidence,
add new models,
start CNN vs ViT,
add frequency-domain fusion,
add adversarial attacks,
expand the RAG system,
redesign the SQL system,
add authentication,
add SaaS features,
begin 4-generator LOGO,
begin the full ablation suite,
download the entire GenImage dataset automatically,
fabricate any experimental metrics.

Focus only on:

1. Remove silent mock fallbacks.
2. Separate infrastructure and real-data gates.
3. Prepare a minimal real GenImage subset workflow.
4. Validate the pipeline on real images.
5. Run one small real NBC pilot.
6. Measure actual compute.
7. Propose and document the final experiment scale.
Definition of Done

This iteration is complete when:

 Mock data can only be activated explicitly.
 Missing real data causes a clear error.
 Infrastructure Gate is separate from Real-Data Pilot Gate.
 A GenImage readiness check exists and reports dataset status.
 The expected real dataset structure is documented.
 A real-data pilot configuration exists.
 One real and one AI-generated GenImage image can be processed through the visualization pipeline once data is available.
 Synthetic and real-data artifacts remain clearly separated.
 A small real NBC pilot can run once the minimal dataset is available.
 The first real experiment is stored as experimental, never mock.
 The pilot reports actual throughput and training measurements.
 A proposed compute scale is generated based on measured results.
 The 4-generator → 3-generator fallback is documented.
 No full LOGO or large-scale experiment has started yet.
Final Output Required

After implementing these changes, stop and provide a concise implementation report containing:

1. Files changed

List every file created or modified.

2. Mock-data policy

Explain exactly how silent fallback was removed.

3. Gate status

Report separately:

INFRASTRUCTURE GATE: PASS/FAIL
REAL-DATA PILOT GATE: PASS/NOT READY/FAIL
4. Dataset status

Report whether GenImage is currently available and what user action is required if not.

5. Pilot status

If real data is available, report actual measurements.

If real data is unavailable, explicitly state:

Real-data pilot has not been run.
No experimental performance claims have been generated.
6. Proposed next action

Stop after this report.

Do not automatically start reproduction or LOGO training.

The next step will be reviewed after the real-data pilot results and compute measurements are available.