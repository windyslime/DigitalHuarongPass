from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .agent import DecisionTrace, NeuralAgent
from .dataset import Example, generate_examples
from .encoding import Action, PuzzleState, encode_state, goal_state, legal_actions
from .environment import PuzzleEnvironment
from .model import HuarongNet, load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    report: Path
    metrics: Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _split_examples(
    examples: tuple[Example, ...], seed: int, validation_ratio: float = 0.2
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_ratio))
    return tuple(shuffled[validation_count:]), tuple(shuffled[:validation_count])


def _loader(examples: Iterable[Example], batch_size: int, shuffle: bool) -> DataLoader:
    examples = tuple(examples)
    features = torch.stack([encode_state(example.state) for example in examples])
    actions = torch.tensor([int(example.expert_action) for example in examples])
    steps = torch.tensor([-float(example.remaining_steps) for example in examples])
    return DataLoader(
        TensorDataset(features, actions, steps),
        batch_size=min(batch_size, len(examples)),
        shuffle=shuffle,
    )


def train_imitation(
    train_examples: tuple[Example, ...],
    validation_examples: tuple[Example, ...],
    *,
    epochs: int,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    device: str = "cpu",
) -> tuple[HuarongNet, list[dict[str, float]]]:
    model = HuarongNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    policy_loss = nn.CrossEntropyLoss()
    q_loss = nn.SmoothL1Loss()
    train_loader = _loader(train_examples, batch_size, shuffle=True)
    validation_loader = _loader(validation_examples, batch_size, shuffle=False)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for features, actions, target_q in train_loader:
            features, actions, target_q = features.to(device), actions.to(device), target_q.to(device)
            optimizer.zero_grad()
            logits, q_values = model(features)
            selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = policy_loss(logits, actions) + 0.2 * q_loss(selected_q, target_q)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(features)

        model.eval()
        correct = 0
        validation_total = 0.0
        validation_count = len(validation_examples)
        with torch.inference_mode():
            for features, actions, target_q in validation_loader:
                features, actions, target_q = features.to(device), actions.to(device), target_q.to(device)
                logits, q_values = model(features)
                selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                validation_total += float(
                    (policy_loss(logits, actions) + 0.2 * q_loss(selected_q, target_q)).item()
                ) * len(features)
                correct += int((logits.argmax(dim=1) == actions).sum().item())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / len(train_examples),
                "validation_loss": validation_total / len(validation_examples),
                "validation_accuracy": correct / validation_count,
            }
        )
    return model, history


def _trace_to_dict(trace: DecisionTrace) -> dict[str, Any]:
    return {
        "state": list(trace.state),
        "blank_position": list(trace.blank_position),
        "selected_action": trace.selected_action.name,
        "confidence": trace.confidence,
        "inference_ms": trace.inference_ms,
        "step": trace.step,
        "hidden_summary": list(trace.hidden_summary),
        "candidates": [
            {
                "action": candidate.action.name,
                "legal": candidate.legal,
                "probability": candidate.probability,
                "q_value": candidate.q_value,
                "heuristic_after": candidate.heuristic_after,
                "heuristic_delta": candidate.heuristic_delta,
            }
            for candidate in trace.candidates
        ],
    }


def evaluate_agent(
    model: HuarongNet,
    examples: tuple[Example, ...],
    *,
    max_steps: int = 120,
    device: str = "cpu",
) -> tuple[dict[str, Any], list[DecisionTrace]]:
    agent = NeuralAgent(model, device=device)
    solved = 0
    illegal = 0
    steps_taken: list[int] = []
    inference_times: list[float] = []
    traces: list[DecisionTrace] = []
    for example in examples:
        environment = PuzzleEnvironment(example.state, max_steps=max_steps)
        for step in range(max_steps):
            trace = agent.decide(environment.state, step=step)
            if len(traces) < 3:
                traces.append(trace)
            inference_times.append(trace.inference_ms)
            transition = environment.step(trace.selected_action)
            if not transition.legal:
                illegal += 1
            if transition.done:
                if transition.state == goal_state():
                    solved += 1
                    steps_taken.append(step + 1)
                break
    count = len(examples)
    return (
        {
            "games": count,
            "solved": solved,
            "solve_rate": solved / count if count else 0.0,
            "average_steps": sum(steps_taken) / len(steps_taken) if steps_taken else None,
            "illegal_actions": illegal,
            "illegal_action_rate": illegal / max(1, count),
            "average_inference_ms": sum(inference_times) / len(inference_times)
            if inference_times
            else 0.0,
            "astar_average_steps": sum(example.remaining_steps for example in examples) / count
            if count
            else None,
        },
        traces,
    )


def train_dqn(
    model: HuarongNet,
    starts: tuple[Example, ...],
    *,
    episodes: int,
    seed: int,
    device: str = "cpu",
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    target_model = HuarongNet()
    target_model.load_state_dict(model.state_dict())
    target_model.to(device).eval()
    replay: deque[tuple[PuzzleState, Action, float, PuzzleState, bool]] = deque(maxlen=20_000)
    history: list[dict[str, float]] = []
    gamma = 0.99
    epsilon = 1.0
    updates = 0
    for episode in range(episodes):
        environment = PuzzleEnvironment(rng.choice(starts).state, max_steps=120)
        episode_reward = 0.0
        for _ in range(120):
            state = environment.state
            legal = legal_actions(state)
            if rng.random() < epsilon:
                action = rng.choice(legal)
            else:
                with torch.inference_mode():
                    _, q_values = model(encode_state(state).unsqueeze(0).to(device))
                action = max(legal, key=lambda candidate: float(q_values[0, int(candidate)].item()))
            transition = environment.step(action)
            replay.append((state, action, transition.reward, transition.state, transition.done))
            episode_reward += transition.reward
            if len(replay) >= 64:
                batch = rng.sample(replay, 64)
                states, actions, rewards, next_states, dones = zip(*batch)
                state_tensor = torch.stack([encode_state(state) for state in states]).to(device)
                next_tensor = torch.stack([encode_state(state) for state in next_states]).to(device)
                action_tensor = torch.tensor([int(action) for action in actions], device=device)
                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                done_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
                _, q_values = model(state_tensor)
                selected = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                with torch.inference_mode():
                    next_q = target_model(next_tensor)[1]
                    targets = reward_tensor + gamma * (1.0 - done_tensor) * next_q.max(dim=1).values
                loss = nn.functional.smooth_l1_loss(selected, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                updates += 1
                if updates % 100 == 0:
                    target_model.load_state_dict(model.state_dict())
            if transition.done:
                break
        epsilon = max(0.05, epsilon * 0.995)
        if episode == 0 or (episode + 1) % max(1, episodes // 20) == 0:
            history.append({"episode": float(episode + 1), "reward": episode_reward, "epsilon": epsilon})
    return history


def write_training_report(
    metrics: dict[str, Any],
    traces: list[DecisionTrace],
    report_path: Path,
    metrics_path: Path,
) -> None:
    metrics = dict(metrics)
    metrics["decision_traces"] = [_trace_to_dict(trace) for trace in traces]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation = metrics.get("evaluation", {})
    lines = [
        "# DigitalHuarongPass Neural Solver Training Report",
        "",
        f"- Stage: `{metrics.get('stage', 'unknown')}`",
        f"- Git commit: `{metrics.get('git_sha', 'unknown')}`",
        f"- Device: `{metrics.get('device', 'cpu')}`",
        f"- Seed: `{metrics.get('seed', 'unknown')}`",
        "",
        "## Dataset And Configuration",
        "",
        "```json",
        json.dumps(metrics.get("config", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Evaluation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Games | {evaluation.get('games', 0)} |",
        f"| Solved | {evaluation.get('solved', 0)} |",
        f"| Solve rate | {evaluation.get('solve_rate', 0):.2%} |",
        f"| Average steps | {evaluation.get('average_steps', 'n/a')} |",
        f"| A* average steps | {evaluation.get('astar_average_steps', 'n/a')} |",
        f"| Illegal action rate | {evaluation.get('illegal_action_rate', 0):.2%} |",
        f"| Average inference time | {evaluation.get('average_inference_ms', 0):.3f} ms |",
        "",
        "## Training History",
        "",
        "```json",
        json.dumps(metrics.get("history", []), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Decision Trace Samples",
        "",
    ]
    for index, trace in enumerate(traces, start=1):
        lines.extend(
            [
                f"### Sample {index}",
                "",
                f"Selected `{trace.selected_action.name}`, confidence `{trace.confidence:.3f}`, inference `{trace.inference_ms:.3f} ms`.",
                "",
                "| Action | Legal | Probability | Q value | Heuristic delta |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for candidate in trace.candidates:
            lines.append(
                f"| {candidate.action.name} | {'yes' if candidate.legal else 'no'} | {candidate.probability:.3f} | {candidate.q_value:.3f} | {candidate.heuristic_delta if candidate.heuristic_delta is not None else 'n/a'} |"
            )
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_training(
    *,
    stage: str = "imitation",
    checkpoint: Path | None = None,
    samples: int = 2500,
    epochs: int = 30,
    episodes: int = 1000,
    eval_games: int = 100,
    seed: int = 42,
    output_root: Path = Path("."),
    device: str = "cpu",
) -> TrainingResult:
    _set_seed(seed)
    output_root = Path(output_root)
    model_dir = output_root / "models"
    report_dir = output_root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = report_dir / f"training-{timestamp}.md"
    metrics_path = report_dir / f"training-{timestamp}.metrics.json"

    eval_examples = generate_examples(eval_games, seed=seed + 1000, min_depth=2, max_depth=10)
    metrics: dict[str, Any] = {
        "stage": stage,
        "git_sha": _git_sha(),
        "device": device,
        "seed": seed,
        "config": {
            "samples": samples,
            "epochs": epochs,
            "episodes": episodes,
            "eval_games": eval_games,
            "board": "4x4",
        },
    }

    if stage == "imitation":
        examples = generate_examples(samples, seed=seed, min_depth=2, max_depth=12)
        train_examples, validation_examples = _split_examples(examples, seed)
        model, history = train_imitation(
            train_examples,
            validation_examples,
            epochs=epochs,
            device=device,
        )
        checkpoint_path = model_dir / "imitation.pt"
        save_checkpoint(checkpoint_path, model, {"stage": "imitation", "seed": seed})
        metrics["dataset"] = {"train": len(train_examples), "validation": len(validation_examples)}
        metrics["supervised"] = {"history": history}
    elif stage == "dqn":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for DQN stage")
        model, metadata = load_checkpoint(checkpoint, device=device)
        starts = generate_examples(samples, seed=seed, min_depth=2, max_depth=12)
        history = train_dqn(model, starts, episodes=episodes, seed=seed, device=device)
        checkpoint_path = model_dir / "dqn.pt"
        save_checkpoint(checkpoint_path, model, {**metadata, "stage": "dqn", "seed": seed})
    elif stage == "eval-only":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for eval-only stage")
        model, metadata = load_checkpoint(checkpoint, device=device)
        history = [{"loaded_checkpoint": str(checkpoint), "stage": metadata.get("stage", "unknown")}]
        checkpoint_path = Path(checkpoint)
    else:
        raise ValueError(f"unknown training stage: {stage}")

    evaluation, traces = evaluate_agent(model, eval_examples, device=device)
    metrics["evaluation"] = evaluation
    metrics["history"] = history
    metrics["selected_checkpoint"] = str(checkpoint_path)
    write_training_report(metrics, traces, report_path, metrics_path)
    return TrainingResult(checkpoint_path, report_path, metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the neural solver")
    parser.add_argument("--stage", choices=("imitation", "dqn", "eval-only"), default="imitation")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_training(**vars(args))
    print(f"checkpoint: {result.checkpoint}")
    print(f"report: {result.report}")
    print(f"metrics: {result.metrics}")


if __name__ == "__main__":
    main()
