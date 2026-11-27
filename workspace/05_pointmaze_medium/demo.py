"""Replay two rollouts of a trained Return-Gate policy in MuJoCo."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiment import (
    EVALUATION_STARTS, FEATURES, FORCES, Config, action_values, encode, make_environment, reset, train,
)

HERE = Path(__file__).resolve().parent
SEED = 81_000
START_CELLS = ((4, 4), (6, 1))
FRAME_SIZE = 480
FRAME_STRIDE = 2


def replay(environment, weights, start, cap):
    """Record a greedy evaluation using the same action rule as the experiment."""
    observation = reset(environment, start, seed=72_001)
    frames = [(0, "Navigating", environment.render())]
    for step in range(1, cap + 1):
        action = int(np.argmax(action_values(weights, encode(observation))))
        result, reward, terminated, truncated, _ = environment.step(FORCES[action])
        observation = result["observation"]
        if terminated or truncated:
            status = "Reached goal" if reward == 1 else "Time limit reached"
            frames.append((step, status, environment.render()))
            return frames, step if reward == 1 else None
        if step % FRAME_STRIDE == 0:
            frames.append((step, "Navigating", environment.render()))
    raise RuntimeError("Expected an episode boundary by the step cap.")


def make_comparison(replays, path):
    """Put two synchronized rollouts side by side and hold each final frame."""
    width = len(replays) * FRAME_SIZE
    header = 87
    font = ImageFont.load_default(size=18)
    frames = []
    for index in range(max(map(len, replays))):
        image = Image.new("RGB", (width, FRAME_SIZE + header), "#152132")
        draw = ImageDraw.Draw(image)
        draw.text((14, 10), "PointMaze Medium | Return-Gate | 100,000 training steps",
                  fill="white", font=font)
        for panel, rollout in enumerate(replays):
            step, status, pixels = rollout[min(index, len(rollout) - 1)]
            x = panel * FRAME_SIZE
            draw.text((x + 14, 38), f"Start {START_CELLS[panel]} | Step {step}",
                      fill="white", font=font)
            color = {"Navigating": "#c2d0e5", "Reached goal": "#85e3a8",
                     "Time limit reached": "#ffb98b"}[status]
            draw.text((x + 14, 61), status, fill=color, font=font)
            image.paste(Image.fromarray(pixels), (x, header))
        frames.append(image.quantize(colors=128, method=Image.Quantize.FASTOCTREE))
    # The simulator runs at 10 Hz. Sampling every two steps and displaying
    # a frame every 100 ms plays the action at roughly double speed.
    durations = [100] * len(frames)
    durations[-1] = 1600
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = HERE / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    config = Config(**manifest["config"])
    saved = json.loads((results / f"seed_{SEED}.json").read_text())
    expected = next(row["records"][-1] for row in saved if row["method"] == "return_gate")

    # The result bundle stores evaluation records. Repeat the official seed
    # to recover its final weights, then check that the saved scores match.
    weights = np.empty(len(FORCES) * FEATURES)
    with make_environment(config) as env, make_environment(config) as evaluation:
        result = train(env, evaluation, "return_gate", SEED, config, weights_out=weights)
    final = result["records"][-1]
    if final != expected:
        raise RuntimeError(f"Replayed training does not match the saved checkpoint: {final}")

    with make_environment(config, render_mode="rgb_array", width=FRAME_SIZE,
                          height=FRAME_SIZE) as viewer:
        replays = []
        for start in START_CELLS:
            frames, arrival_step = replay(viewer, weights, start, config.cap)
            if arrival_step != final["steps_by_start"][EVALUATION_STARTS.index(start)]:
                raise RuntimeError(f"Rendered rollout did not match evaluation at {start}.")
            replays.append(frames)
            print(f"Start {start}: {frames[-1][1]} at step {frames[-1][0]}.")
    make_comparison(replays, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
