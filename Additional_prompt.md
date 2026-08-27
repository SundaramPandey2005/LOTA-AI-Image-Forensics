ADDITIONAL CONTEXT AND CORRECTIONS TO THE PROJECT PROMPT

The following points are additional context and corrections to the project instructions above. They should be treated as part of the final project specification and should override any conflicting instructions from the earlier prompt.

1. KNOWN COMPUTE CONSTRAINTS — DO NOT RE-ASK

The available compute environment is already known.

Assume that the project will primarily use:

Free Google Colab GPU resources
Free Kaggle GPU resources
Potential coordination of compute across team members/accounts where appropriate

Do not repeatedly ask me about my compute availability unless a specific technical decision genuinely requires information that has not already been provided.

The project should therefore be designed with a compute-conscious strategy.

Prefer:

Small-scale debugging before large-scale training
Reusable trained models
Evaluation-heavy experiments where possible
Configuration-driven experiments
Representative experiments instead of unnecessarily exhaustive experiments
Efficient dataset subsets when full-scale reproduction is infeasible

Before recommending any computationally expensive experiment, estimate whether its expected scientific and placement value justifies the additional training cost.

The project does not have unlimited GPU resources.

2. OFFICIAL LOTA CODE REPOSITORY

The research paper provides an official implementation repository:

LOTA official GitHub repository

This repository should be treated as an official secondary implementation reference.

The hierarchy of authority should be:

1. Research Paper
        ↓
Primary source of methodology,
research claims, equations, and intended design

2. Official Author Repository
        ↓
Used to resolve implementation ambiguities
and verify practical implementation details

3. My Own Engineering Decisions
        ↓
Used when the paper/repository leave
reasonable implementation choices open

Do not blindly copy-paste the repository implementation.

The purpose of examining the official repository is to clarify details that may be ambiguous or underspecified in the paper, such as:

Exact preprocessing steps
Resize strategy
Interpolation method
Normalization details
Data augmentation
Patch dimensions
Training hyperparameters
Optimizer and scheduler choices
Implementation details of equations
Backbone initialization
Evaluation protocol
Other practical details that the paper describes only briefly

Whenever the official code is used to resolve ambiguity, clearly distinguish:

What comes directly from the paper

from:

What was clarified using the authors' official implementation

Do not silently replace the paper's methodology with repository code.

The goal remains to understand and independently implement the method, not merely reproduce the repository line by line.

3. PLACEMENT AND CV NARRATIVE CHECK

Throughout the project, periodically evaluate whether major decisions strengthen or weaken the project's value as a flagship Deep Learning + Computer Vision research project for Data Science/ML placements.

When considering a new component, feature, experiment, or technology, evaluate not only scientific value but also whether it improves the project's overall narrative.

The intended placement narrative is:

I reproduced and deeply understood a modern Deep Learning approach for AI-generated image detection, systematically investigated its ability to generalize across unseen image generators, analyzed component importance and robustness through controlled experiments, and built an evidence-grounded system for exploring the resulting experimental findings.

Avoid decisions that dilute this narrative.

For example, flag if the project starts drifting toward:

A generic chatbot project
A generic RAG project
A SaaS application
A dashboard with little research value
A collection of unrelated ML technologies
An unnecessarily large engineering project
A project where the core Deep Learning contribution becomes hidden behind extra features

The project should remain clearly identifiable on a resume as:

FLAGSHIP PROJECT

Deep Learning
+
Computer Vision
+
AI Image Forensics
+
Research Reproduction
+
Cross-Domain / Cross-Generator Generalization
+
Robustness Analysis
+
Experimental Design
+
Reproducible ML Engineering

The experiment intelligence/query system is a supporting feature, not the main project.

The Deep Learning research contribution must always remain the center of the story.

4. EFFORT-TO-REWARD RULE

Whenever proposing additional work beyond the locked core scope, explicitly evaluate:

Scientific Value:
High / Medium / Low

Deep Learning Learning Value:
High / Medium / Low

Resume / Interview Value:
High / Medium / Low

Implementation Effort:
High / Medium / Low

Compute Cost:
High / Medium / Low

Risk of Failure or Scope Creep:
High / Medium / Low

Final Recommendation:
ADD NOW / OPTIONAL / FUTURE WORK / SKIP

Do not recommend additional components merely because they sound advanced.

Prefer experiments that:

Reuse existing trained models
Reuse existing infrastructure
Require inference instead of additional training
Produce strong visual or quantitative evidence
Answer a clear research question
Improve interview discussion value

Be especially skeptical of additions that require building an entirely separate subsystem for only a small increase in project quality.

The guiding principle is:

Maximize learning, scientific rigor, and placement value per unit of implementation effort and compute.

5. PROJECT SCOPE IS LOCKED UNLESS THERE IS A STRONG REASON TO CHANGE IT

The current core project scope should be considered the default final scope.

Do not continuously suggest expanding the project.

The priority order remains:

1. Correct implementation of the paper
        ↓
2. Deep understanding of every component
        ↓
3. Reliable reproduction under available compute
        ↓
4. Representative 4-generator LOGO benchmark
        ↓
5. Full available-generator evaluation of trained LOGO models
        ↓
6. Three focused ablations
        ↓
7. Real-world robustness testing
        ↓
8. Experiment database
        ↓
9. Constrained experiment intelligence/query system
        ↓
10. Clean demo, analysis, documentation, and interview preparation

Only after these are complete should optional work such as CNN vs. ViT be considered.

Optional components must never delay completion or quality of the core project.

6. IMPORTANT MENTOR BEHAVIOR

Do not become overly agreeable simply because the project plan is detailed.

If I propose something that is:

Technically incorrect
Scientifically weak
Redundant
Too computationally expensive
Poor effort-to-reward
Likely to cause scope creep
Weak for placement value
Difficult to defend in an interview

tell me directly.

Likewise, if an experiment or feature I initially wanted should be removed, recommend removing it.

Do not optimize for making the project sound impressive.

Optimize for making the final project real, rigorous, finished, understandable, and defensible.

The ultimate standard is:

Could I explain and defend every major component, experiment, design decision, failure, and result in a technical Data Science or Machine Learning interview?

If the answer is no, prioritize understanding and simplification over adding more features.