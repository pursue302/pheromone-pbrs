# MuJoCo AntMaze

This experiment extends the PointMaze study to native continuous control in
[AntMaze U-Maze](https://robotics.farama.org/envs/maze/ant_maze/).
All arms use the same eight-dimensional torque action space and the same
CPU Soft Actor-Critic (SAC) learner. The learner observes the ant's pose,
velocity, planar position and goal position. The maze goal is in cell
(1, 1), with small start and goal position perturbations. The near start
is set 0.65 units left of its sampled goal. The middle and far starts are
in cells (1, 2) and (3, 1), respectively. Training samples these tiers
with probabilities 60%, 25% and 15%. Evaluations weight the tiers equally.

## Protocol

1. For each seed, train a **shared warm start** for 120,000 interactions,
   with one SAC update per four interactions. The warm start uses forward
   progress along a waypoint-based distance estimate, a small step cost
   and a terminal success bonus. This phase uses layout information and is
   charged to the budget of every arm. Only real, successful trajectories
   populate the shaping memory. The original AntMaze reward remains the
   evaluation signal throughout.
2. At 20,000-interaction intervals, check the shared policy using a reset
   seed bank reserved for checkpoint selection. For each seed choose its
   best checkpoint by success rate, breaking ties by discounted return.
   Candidate checkpoints need at least five completed successes. The
   comparison starts only if every seed has a checkpoint and the selected
   checkpoints average at least 25% success in the selection reset bank.
   All 120,000 interactions are charged even when an earlier checkpoint
   is selected. Final policy metrics use a separate reset seed bank.
   Keep the full paired cohort, including seeds with low individual rates.
3. Fork each seed's actor, critics, optimizers, replay buffer and trajectory
   history into four arms. Train each arm for another 100,000 interactions,
   with one SAC update per eight interactions. The waypoint training hint
   ends at the split. Potential-based arms recompute shaping on replay
   samples with the currently published potential. Actual successes update
   the potential memories, and updated potentials are published every ten
   completed episodes.
4. At each checkpoint, use the frozen actor's mean action. Evaluate from
   six fixed reset seeds per tier. Report final success, final original
   discounted return, and area under the success curve after the split.
   Return-Gate versus fixed mixing on post-split AUC is the primary
   comparison. Other paired comparisons are descriptive and their
   randomization p-values are unadjusted.

| Arm | Reward during the comparison phase |
|---|---|
| Sparse SAC | Original sparse AntMaze reward |
| Return-D (all) | Original reward plus one updated potential from the full history |
| Fixed mix | Original reward plus shaping from an equal mix of the frozen and online potentials |
| Return-Gate | Original reward plus shaping with online weight 0.5n/(n+5), where n counts fresh successes visiting that position bin |

Both success and the 300-step time limit end an episode. A success delivers
one original reward of +1, so mean final undiscounted original return equals
final success rate. The discounted return also accounts for time to goal.

## Reproduction

Run from the repository root. The CPU PyTorch wheel can be installed into
the Conda environment with the requirements file.

```bash
python -m pip install -r requirements.txt -r workspace/06_antmaze/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python workspace/06_antmaze/experiment.py --pilot --seeds 2 --seed-start 91020 --workers 2 --out runs/antmaze-pilot
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python workspace/06_antmaze/experiment.py --workers 16 --out runs/antmaze-main
python workspace/06_antmaze/analyze.py runs/antmaze-main
```

The analysis writes paired statistics and success curves. Individual run
records retain success steps and timeouts by starting tier.

## Pilot progression

The [initial pilot](pilots/standard_starts/summary.json) used the standard
cell-centered near, middle and far starts in cells (1, 2), (2, 3) and (3, 1).
Both pilot seeds had zero final greedy evaluation success after 120,000
interactions. The 0.90-unit and 0.65-unit near-start pilots moved the middle
tier to cell (1, 2) and retained the far tier at (3, 1). Their last checkpoints
averaged 19.4% and 22.2% success across the three tiers. Both saw higher
success at intermediate checkpoints. This informed the separate checkpoint
selection bank used in the main protocol.

The [checkpoint-selection pilot](pilots/selected_checkpoint/summary.json)
passed the 25% selection threshold and scored 25.0% on the reporting reset bank.

## Main results

Eight new seeds (92000–92007) shared 120,000 warm-up interactions per seed
and then received 100,000 further interactions per arm. The selected
checkpoints scored 31.25% in the selection bank and 30.56% in the reporting
bank before the split, all from the near tier. At the end, each arm was
evaluated in 144 episodes, 48 per tier.

| Method | Final success | Near success | Final discounted return | Post-split success AUC |
| --- | ---: | ---: | ---: | ---: |
| Sparse SAC | 7.64% | 22.92% | 0.0556 | 11.74% |
| Return-D (all) | 4.17% | 12.50% | 0.0271 | 9.03% |
| Fixed mix | 9.72% | 29.17% | 0.0694 | 9.17% |
| Return-Gate | 9.72% | 29.17% | 0.0475 | 11.11% |

Middle and far final success were both zero for every arm. Return-Gate
and fixed mixing tied on final success. The primary AUC contrast favored
Return-Gate by 1.94 percentage points (paired 95% bootstrap interval
[-2.08, +5.56] points, randomization p = 0.388). Sparse SAC had the highest
AUC. All final arms fell below the selected shared starting checkpoint
on the reporting reset bank.

The first run exposed a JSON export error for a NumPy scalar in the
telescoping check. After converting that scalar to a Python float, missing
records were regenerated from the original per-seed checkpoints with the
same settings. All 32 arms and their original-reward metrics were verified.

[Statistics](results/summary.json) · [Learning curves](results/success_curves.png)
