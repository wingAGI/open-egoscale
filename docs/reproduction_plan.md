# Open EgoScale Reproduction Plan

This document narrows the project goal from "full EgoScale replication" to a staged reproduction path that can be executed with limited data and compute.

The key constraint is that the original paper's strongest claim combines:

- Stage I large-scale human-only pretraining
- Stage II aligned human-plus-robot mid-training
- Stage III task-specific robot finetuning

Under limited conditions, the correct strategy is to separate the work into two milestones. Stage I can be validated without any real robot. Stage II and Stage III are only added after the Stage I scaling trend is stable.

## Phase 1

### Goal

Reproduce the Stage I scaling trend:

- x-axis: human pretraining data size
- y-axis: human validation loss

This phase does not require any robot data, robot embodiment adapter, or real-world rollout.

### Why this phase matters

This is the cheapest way to test the core Stage I claim:

- human egocentric manipulation data provides dense action supervision
- larger human data improves action prediction quality
- validation loss should decrease roughly linearly with log data size

This phase only establishes a scaling trend on human data. It does not prove robot transfer.

### Scope

Use only Stage I:

- train on human-retargeted samples
- evaluate on a fixed held-out human validation set
- compare several training data scales with the same architecture and recipe

Suggested scales for a practical reproduction:

- 0.1x
- 0.25x
- 0.5x
- 1.0x

If the dataset is large enough, report the actual hours or clip counts for each point. The x-axis should be the actual human data size, and the plot should additionally include `log(data_size)`.

### Data requirements

Each Stage I sample should remain human-only and match the Stage I assumptions already encoded in the current schema:

- `embodiment_id = sharpa` for the current synthetic reference path
- `data_source = human_retargeted`
- `has_proprio = false`
- action semantics shared with the target robot action space

For a future SO101-oriented branch, the semantic requirement should be interpreted as:

- human trajectories are retargeted into the same low-dimensional action space used by the target robot
- no real robot data is required in this phase

### Experimental controls

To make the scaling plot meaningful, keep all of the following fixed across runs:

- model architecture
- action semantics
- train and validation split
- optimizer and learning-rate schedule
- batch size and max steps or token budget
- random seeds, or at minimum evaluate with multiple seeds

Only the Stage I human data size should vary.

### Deliverables

Phase 1 is complete only when the project can produce:

1. A fixed Stage I validation set that is never used in training.
2. At least 4 Stage I training subsets at increasing data scales.
3. A per-scale validation loss report.
4. A plot of `human data size -> human validation loss`.
5. A plot of `log(human data size) -> human validation loss`.

### Minimum acceptance criteria

The phase should be considered successful if:

- validation loss decreases monotonically or near-monotonically as data size increases
- the trend is stable across seeds or reruns
- overfitting is lower for larger datasets

### Code implications

The current repository already contains the Stage I training path:

- [`train_stage1.py`](/Users/hex/workspace/knowledge_space/embodied_AI/VLA/egovla/egoscale/open-egoscale/egoscale_research/scripts/train_stage1.py)
- [`stage1.yaml`](/Users/hex/workspace/knowledge_space/embodied_AI/VLA/egovla/egoscale/open-egoscale/egoscale_research/configs/stage1.yaml)
- [`schema.py`](/Users/hex/workspace/knowledge_space/embodied_AI/VLA/egovla/egoscale/open-egoscale/egoscale_research/src/egoscale/data/schema.py)

The main missing piece for a proper Phase 1 reproduction is a dedicated validation workflow:

- fixed train and validation manifests
- a Stage I evaluation script that computes held-out loss
- a small runner that trains multiple data scales and aggregates results

## Phase 2

### Goal

Add a minimal Stage II and Stage III to test whether lower Stage I validation loss predicts better robot performance.

Two acceptable downstream metrics:

- robot open-loop action error
- robot task success rate

Open-loop error is cheaper and should be the first target. Success rate can be added after the pipeline is stable.

### Why this phase matters

Phase 1 only shows that the model gets better at predicting human pseudo-actions.

Phase 2 tests the stronger claim:

- a better Stage I human prior transfers into better robot-grounded behavior after a small amount of alignment and task finetuning

### Scope

Keep Stage II and Stage III intentionally small.

Recommended plan:

- use one target embodiment only
- use a small aligned human-plus-robot dataset for Stage II
- use a small task-specific robot dataset for Stage III
- reuse the same Stage II and Stage III data for every Stage I checkpoint

This keeps the comparison fair. The only changing variable should remain the Stage I pretraining scale.

### Stage II data

Stage II should mix:

- aligned human demonstrations
- robot demonstrations from the target embodiment

The aligned human data should match robot deployment views as closely as possible. This is the point of Stage II: grounding the human prior into the robot sensing and control interface.

### Stage III data

Stage III should use:

- small task-specific robot demonstrations

This stage is not for learning the general prior. It is only for task specialization.

### Experimental controls

For a valid comparison:

- the Stage II dataset must be identical across all runs
- the Stage III dataset must be identical across all runs
- downstream evaluation tasks must be identical across all runs
- only the Stage I checkpoint should change

### Deliverables

Phase 2 is complete only when the project can produce:

1. A fixed set of Stage I checkpoints at increasing human-data scales.
2. One fixed Stage II aligned dataset.
3. One fixed Stage III robot dataset.
4. A downstream evaluation table for each Stage I checkpoint.
5. A plot of `Stage I validation loss -> robot open-loop error` or `Stage I validation loss -> task success rate`.

### Minimum acceptance criteria

The phase should be considered successful if:

- lower Stage I validation loss usually leads to lower robot open-loop error, or
- lower Stage I validation loss usually leads to higher task success rate

The trend does not need to be perfectly linear. A clear ranking correlation is already useful.

## Execution Order

Recommended project order:

1. Finish Phase 1 first.
2. Do not collect robot data until the Stage I scaling plot is stable.
3. Add the smallest possible Stage II dataset.
4. Add the smallest possible Stage III dataset.
5. Start with open-loop evaluation before trying full real-robot success rate.

## Non-Goals

The staged reproduction explicitly does not aim to:

- match the original paper's absolute success numbers
- match the original paper's 20k-hour scale
- replicate the exact Sharpa hardware stack
- claim full human-to-robot scaling law reproduction from Phase 1 alone

## Summary

The project should be judged in two steps:

- Phase 1 answers: does larger human-only pretraining data reduce held-out human validation loss?
- Phase 2 answers: does a better Stage I checkpoint transfer better after small aligned robot grounding?

This is the smallest defensible route to reproducing the EgoScale recipe under realistic resource limits.
