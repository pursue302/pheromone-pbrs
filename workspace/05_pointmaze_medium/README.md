# MuJoCo PointMaze Medium

This experiment tests whether adaptive memory mixing improves learning on
continuous navigation dynamics. It uses the standard Medium layout from
[Gymnasium-Robotics](https://robotics.farama.org/envs/maze/point_maze/),
sparse rewards, and termination upon reaching the goal. The goal is fixed
at row 1, column 6, with zero-based indexing. Training starts are sampled
uniformly from eight fixed cells, and evaluation visits each of those cells
once per checkpoint. Position noise is set to zero. This protocol evaluates
learning and seed variability on the same task.

The learner is linear True Online SARSA with tile-coded position and velocity
and nine discrete force choices. Its sparse Dutch trace update is algebraically
equivalent to the repository's dense canonical update. The first quarter of
the training budget uses original rewards, through the first episode boundary
at or after that point. All methods share the seed-dependent warm-up. Successful
episodes deposit weighted last-visit returns into position bins. Unseen bins
use zero and potentials are scaled by 0.1. Published potentials are fixed for
each episode, including a zero endpoint potential at success or the 250-step
episode limit.

| Method | Potential after warm-up |
|---|---|
| Unshaped SARSA | Zero |
| Return-D (all) | A single dynamic memory retaining warm-up and later successful trajectories |
| Fixed mix | Equal weights for frozen warm-up memory and a separate online memory |
| Return-Gate | The same two memories, with online weight (0.5n/(n+5)) per position bin |

A three-seed pilot used seeds 80000–80002 and 30,000 training interactions
per method and seed. The main comparison uses fresh paired seeds 81000–81019
and 100,000 interactions per method and seed. The primary comparison is
Return-Gate versus Fixed mix on normalized AUC of greedy evaluation success
rate over training. Evaluations occur every 10,000 interactions, including
at zero, and use the original task. Final success rate and contrasts against
the other baselines are secondary and descriptive. Intervals are paired
bootstrap percentile intervals over training seeds. Reported randomization
p-values are two-sided and unadjusted.

Run from the repository root in the `finalproject` Conda environment:

```bash
python -m pip install -r requirements.txt -r workspace/05_pointmaze_medium/requirements.txt
python workspace/05_pointmaze_medium/experiment.py --out runs/pointmaze-medium --workers 12
python workspace/05_pointmaze_medium/analyze.py runs/pointmaze-medium
```

The training command requires a new output directory. It writes a manifest and
one JSON file per training seed. The analysis writes `summary.json` and
`success_curves.png`. The shaded region is one standard error across training
seeds, and the dashed line shows the mean end of warm-up.

The completed main run is available in [results](results), with the
[summary](results/summary.json) and [learning curves](results/success_curves.png).

## Results

| Method | Success-rate AUC | Final success |
|---|---:|---:|
| Unshaped SARSA | 17.47% | 33/160 (20.63%) |
| Return-D (all) | 19.28% | 37/160 (23.13%) |
| Fixed mix | 19.22% | 35/160 (21.88%) |
| Return-Gate | 20.00% | 38/160 (23.75%) |

The primary comparison (Return-Gate minus Fixed mix) yielded +0.78 percentage
points of AUC, with a paired 95% bootstrap interval of [-0.78, +2.72] points
and a two-sided randomization p-value of 0.466. The secondary comparison with
unshaped SARSA yielded +2.53 points of AUC [0.97, 4.88], with an unadjusted
p-value of 0.0006. The experiment supports an AUC gain over unshaped SARSA in
this setup. The added value of the adaptive gate over fixed mixing remains
uncertain.

All 20 seeds produced at least one successful warm-up trajectory. Every
method succeeded from the nearest start in all 20 final evaluations. The
number of final successes from the other seven starts ranged from 13 for
unshaped SARSA to 18 for Return-Gate, indicating persistent difficulty
across the layout.

## Trained policy replay

The [MuJoCo replay](../../paper/figures/return_gate_trained.gif) shows the frozen Return-Gate
policy for seed 81000 after 100,000 training interactions. The start at (4, 4)
reaches the goal in 128 steps. The start at (6, 1) times out at 250 steps.
Recreate the GIF from the repository root with an EGL-capable MuJoCo setup.

```bash
MUJOCO_GL=egl python workspace/05_pointmaze_medium/demo.py --out /tmp/return_gate_trained.gif
```
