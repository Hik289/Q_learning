# Q-Learning Pricing Experiments

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-pending-0E7C66.svg" alt="License"></a> <a href="#paper-or-reference"><img src="https://img.shields.io/badge/paper-reference-1F4E79.svg" alt="Paper or reference"></a> <img src="https://img.shields.io/badge/language-Jupyter%20Notebook-F37626.svg" alt="Jupyter Notebook">
</p>

<p align="center">
  <strong>Reinforcement-learning notebooks for pricing, mixed strategy games, and auction references.</strong>
</p>

<p align="center">
  <img src="assets/readme-figure.png" alt="Q-Learning Pricing Experiments overview" width="100%">
</p>

**Figure 1.** The overview figure describes the RL loop used by the project: define states and rewards, update action values, improve the policy, and track convergence against pricing objectives.

## Scope

This repository is organized as a conference-style research artifact for Q-learning pricing games and auction experiments. Q-Learning Pricing Experiments keeps notebooks, reference papers, and report artifacts for studying pricing and mixed-strategy games through reinforcement learning. The code folder separates reusable Q-learning assets from exploratory notebooks and written references.

The README is structured for fast inspection by reviewers and future collaborators: it states the artifact scope, the main entry points, the reproduction path, and the outputs that should be checked after a run.

## Artifact Contents

| Component | Role |
| --- | --- |
| `Q_learning_code/` | core implementation folder for reusable Q-learning code. |
| `AM_model_thirdversion.ipynb`, `betrand_cost.ipynb`, `cycle.ipynb` | experiment notebooks. |
| `graph/` | saved plots and graph artifacts. |
| `report1/`, `report2/` | written result folders. |
| `q-learninng_for_pricing_mixed_strategy_game.docx` | project write-up for the pricing game experiment. |

## Reproduction Guide

1. `git clone git@github.com:Hik289/Q_learning.git`
2. `python -m venv .venv && source .venv/bin/activate`
3. `python -m pip install -U pip jupyter numpy pandas scipy matplotlib`
4. Start with the notebooks, then move reusable code into `Q_learning_code/` when an experiment stabilizes.

For a full rerun, record the data window, random seed, software versions, machine type, and command used for each experiment. Keep raw datasets outside Git unless they are small public fixtures.

## Experimental Workflow

| Stage | What to Check |
| --- | --- |
| Setup | Confirm local data paths, environment packages, and any MATLAB or notebook paths before running experiments. |
| Run | Execute the smallest script or notebook first, then scale to the full experiment once outputs match expectations. |
| Inspect | Compare generated figures, logs, tables, and saved result folders against the intended analysis. |
| Extend | Add new experiments as separate scripts or notebooks with explicit names instead of overwriting existing artifacts. |

## Expected Outputs

- Recreated figures, tables, notebooks, reports, or saved result files from the listed entry points.
- A clear mapping from each experiment command to its generated output location.
- Updated notes when a script depends on local data, private paths, or external software.

## Paper or Reference

No external paper link is currently attached to this project. For now, the code, notebooks, and notes in this repository are the primary reference artifact.

## Citation

If this repository supports academic work, cite the linked paper when available. Otherwise cite the repository version used in your experiment.

```bibtex
@misc{q_learning_artifact_2026,
  title = {{Q-Learning Pricing Experiments}},
  author = {Hik289},
  year = {2026},
  howpublished = {\url{https://github.com/Hik289/Q_learning}},
  note = {Research artifact}
}
```

## License

No explicit license file is included yet. Add one before public reuse, redistribution, or package release.

## Reviewer Notes

| Item | Status |
| --- | --- |
| Code | Included in this repository. |
| Data | Expected to be configured locally unless a small fixture is committed. |
| Environment | Base dependencies are listed in the reproduction guide; pin a lockfile for archival release. |
| Results | Store generated artifacts under the existing result, report, log, or output folders. |
