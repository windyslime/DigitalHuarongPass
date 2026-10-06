# Neural Solver Visualization Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a hybrid imitation/DQN model for the 4x4 puzzle, generate a reproducible training report, and visualize every AI decision in the Pygame application.

**Architecture:** A pure puzzle environment and encoder sit below a small PyTorch dual-head network. Training and reporting use those modules offline, while `NeuralAgent` exposes an immutable `DecisionTrace` consumed by a separate Pygame panel.

**Tech Stack:** Python 3.13, PyTorch, Pygame, pytest, uv

## Global Constraints

- Preserve the current manual game when no model is available.
- Keep action order exactly `UP`, `DOWN`, `LEFT`, `RIGHT`, from the blank tile's perspective.
- Support the configured 4x4 board in the first release.
- Do not commit transient dataset caches or intermediate checkpoints.
- Generate Markdown and JSON reports for every completed training run.

---

### Task 1: Puzzle State Encoding And Environment

**Files:**
- Modify: `pyproject.toml`
- Create: `src/ai/__init__.py`
- Create: `src/ai/encoding.py`
- Create: `src/ai/environment.py`
- Create: `tests/ai/test_encoding.py`
- Create: `tests/ai/test_environment.py`

**Interfaces:**
- Produces: `Action`, `PuzzleState`, `encode_state(state)`, `legal_actions(state)`, `heuristic_score(state)`, `PuzzleEnvironment.reset()`, and `PuzzleEnvironment.step(action)`.
- Consumes: no rendering or mutable `Board` state.

- [ ] **Step 1: Add failing encoding and transition tests**

```python
def test_goal_encoding_has_stable_shape():
    encoded = encode_state(goal_state())
    assert encoded.shape == (260,)

def test_right_moves_blank_and_can_finish():
    env = PuzzleEnvironment((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, 15))
    transition = env.step(Action.RIGHT)
    assert transition.state == goal_state()
    assert transition.done is True
```

- [ ] **Step 2: Run the tests and confirm imports fail**

Run: `uv run pytest tests/ai/test_encoding.py tests/ai/test_environment.py -q`

- [ ] **Step 3: Implement the action contract and 260-value encoder**

```python
class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}
```

Encode each of 16 cells with a 16-value one-hot vector where blank maps to index 0, then append normalized blank row, blank column, Manhattan distance, and linear-conflict count.

- [ ] **Step 4: Implement deterministic transitions and rewards**

Legal moves swap the blank with its target. Illegal moves retain the state and return `-1.0`; completing the puzzle returns `10.0`; other legal moves return `0.1 * (old_heuristic - new_heuristic) - 0.01`.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/ai/test_encoding.py tests/ai/test_environment.py -q`
Expected: all pass.

- [ ] **Step 6: Commit the environment**

```bash
git add pyproject.toml uv.lock src/ai tests/ai/test_encoding.py tests/ai/test_environment.py
git commit -m "feat: add neural puzzle environment"
```

### Task 2: Network, Checkpoints, And Decision Traces

**Files:**
- Create: `src/ai/model.py`
- Create: `src/ai/agent.py`
- Create: `tests/ai/test_model.py`
- Create: `tests/ai/test_agent.py`

**Interfaces:**
- Consumes: `encode_state`, `legal_actions`, `apply_action`, and `heuristic_score` from Task 1.
- Produces: `HuarongNet.forward(x) -> (policy_logits, q_values)`, `save_checkpoint`, `load_checkpoint`, `ActionScore`, `DecisionTrace`, and `NeuralAgent.decide(state, step)`.

- [ ] **Step 1: Add failing shape and trace tests**

```python
def test_network_has_two_four_action_heads():
    policy, q_values = HuarongNet()(torch.zeros(2, 260))
    assert policy.shape == (2, 4)
    assert q_values.shape == (2, 4)

def test_agent_never_selects_illegal_action():
    trace = NeuralAgent(HuarongNet()).decide(goal_state(), step=0)
    assert trace.selected_action in legal_actions(goal_state())
    assert len(trace.candidates) == 4
```

- [ ] **Step 2: Run tests and confirm model imports fail**

Run: `uv run pytest tests/ai/test_model.py tests/ai/test_agent.py -q`

- [ ] **Step 3: Implement the dual-head model and versioned checkpoints**

```python
class HuarongNet(nn.Module):
    def __init__(self, input_size: int = 260):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.policy_head = nn.Linear(64, 4)
        self.q_head = nn.Linear(64, 4)
```

Checkpoint metadata must include schema version, board size, action order, stage, training config, and normalization metadata.

- [ ] **Step 4: Implement immutable decision traces**

Each candidate records action, legal flag, probability, Q value, and heuristic delta. Selection masks illegal actions, ranks legal actions using the average of normalized policy and Q scores, and measures inference time.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest tests/ai/test_model.py tests/ai/test_agent.py -q`

```bash
git add src/ai/model.py src/ai/agent.py tests/ai/test_model.py tests/ai/test_agent.py
git commit -m "feat: add neural agent decision traces"
```

### Task 3: Dataset, Hybrid Training, Evaluation, And Reports

**Files:**
- Create: `src/ai/dataset.py`
- Create: `src/ai/train.py`
- Create: `tests/ai/test_dataset.py`
- Create: `tests/ai/test_training.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `Board.get_road`, Task 1 environment, and Task 2 model/agent.
- Produces: `generate_examples`, `train_imitation`, `train_dqn`, `evaluate_agent`, `write_training_report`, and the `python -m ai.train` CLI.

- [ ] **Step 1: Add failing deterministic-data and artifact tests**

```python
def test_dataset_is_reproducible():
    first = generate_examples(12, seed=7, min_depth=2, max_depth=5)
    second = generate_examples(12, seed=7, min_depth=2, max_depth=5)
    assert first == second

def test_tiny_training_writes_checkpoint_and_report(tmp_path):
    result = run_training(samples=24, epochs=1, eval_games=4, output_root=tmp_path)
    assert result.checkpoint.exists()
    assert result.report.exists()
    assert result.metrics.exists()
```

- [ ] **Step 2: Run tests and confirm training imports fail**

Run: `uv run pytest tests/ai/test_dataset.py tests/ai/test_training.py -q`

- [ ] **Step 3: Implement A-star-labelled random-walk examples**

Generate unique states by legal random walks from the goal, map the first direction from `Board.get_road(weight=1.0)` to `Action`, and store state, expert action, and remaining path length. Split by state before building loaders.

- [ ] **Step 4: Implement supervised training and evaluation**

Optimize policy cross entropy plus `0.2 * smooth_l1(expert_q, -remaining_steps)`. Evaluate classification accuracy and actual greedy solve runs on held-out random-walk states.

- [ ] **Step 5: Implement DQN fine-tuning**

Use replay capacity 20,000, batch size 64, discount `0.99`, epsilon decay, and a target-network update every 100 optimization steps. Preserve the imitation checkpoint and select the DQN checkpoint only when held-out solve rate improves, or solve rate ties while average steps decreases.

- [ ] **Step 6: Implement Markdown and JSON reporting**

The report records Git SHA, device, seed, config, data-generation failures, epoch metrics, solve metrics, baseline comparison, selected checkpoint, and sample `DecisionTrace` tables.

- [ ] **Step 7: Ignore generated bulk artifacts while keeping the final report reviewable**

Ignore dataset caches and intermediate checkpoints. Keep the selected model and final report available to the user, but do not stage them until final validation.

- [ ] **Step 8: Run focused tests and commit**

Run: `uv run pytest tests/ai/test_dataset.py tests/ai/test_training.py -q`

```bash
git add src/ai/dataset.py src/ai/train.py tests/ai/test_dataset.py tests/ai/test_training.py .gitignore
git commit -m "feat: train and report neural solver"
```

### Task 4: Pygame Decision Visualization

**Files:**
- Create: `src/render/ai_panel.py`
- Modify: `src/render/window.py`
- Modify: `src/config/render.py`
- Create: `tests/render/test_ai_panel.py`
- Create: `tests/render/test_window_ai.py`

**Interfaces:**
- Consumes: `DecisionTrace`, current `Board`, and Pygame surfaces.
- Produces: `draw_ai_panel`, `draw_action_overlay`, and manual/AI runtime controls.

- [ ] **Step 1: Add failing headless render tests**

```python
def test_panel_draws_non_background_pixels(dummy_screen, sample_trace):
    draw_ai_panel(dummy_screen, pygame.Rect(800, 0, 380, 720), sample_trace, paused=True)
    assert pygame.image.tostring(dummy_screen, "RGB") != bytes(dummy_screen.get_width() * dummy_screen.get_height() * 3)
```

- [ ] **Step 2: Run tests with the dummy video driver**

Run: `SDL_VIDEODRIVER=dummy uv run pytest tests/render -q`

- [ ] **Step 3: Implement the panel and overlay**

Use a fixed 380-pixel panel above 980 pixels of window width and a responsive 32% panel below that. Render status, model metadata, four stable candidate rows, confidence, inference time, and recent actions. Draw the chosen blank-movement arrow over the board using Pygame lines and a polygon arrowhead.

- [ ] **Step 4: Integrate controls and safe fallback**

`A` toggles AI, `SPACE` pauses/steps, `T` toggles the panel, and `R` creates a new board. Inference runs only between move animations. A missing or incompatible checkpoint leaves manual mode active and displays the load error.

- [ ] **Step 5: Run rendering tests and commit**

Run: `SDL_VIDEODRIVER=dummy uv run pytest tests/render -q`

```bash
git add src/render/ai_panel.py src/render/window.py src/config/render.py tests/render
git commit -m "feat: visualize neural solver decisions"
```

### Task 5: Train, Document, And Validate The Complete Flow

**Files:**
- Modify: `README.md`
- Create: `models/best.pt`
- Create: `reports/training-<timestamp>.md`
- Create: `reports/training-<timestamp>.metrics.json`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a runnable default model, a human-readable training report, and reproducible commands.

- [ ] **Step 1: Document setup, training, controls, and report locations**

Include exact `uv sync`, imitation training, DQN fine-tuning, evaluation-only, and game-launch commands.

- [ ] **Step 2: Run meaningful imitation training**

Run: `uv run python -m ai.train --stage imitation --samples 2500 --epochs 30 --eval-games 100 --seed 42`

- [ ] **Step 3: Run DQN fine-tuning from the imitation checkpoint**

Run: `uv run python -m ai.train --stage dqn --checkpoint models/imitation.pt --episodes 1000 --eval-games 100 --seed 42`

- [ ] **Step 4: Generate the final evaluation report**

Run: `uv run python -m ai.train --stage eval-only --checkpoint models/best.pt --eval-games 200 --seed 42`

- [ ] **Step 5: Run all checks**

Run: `uv run pytest -q`
Run: `uv run python -m compileall -q src`
Run: `SDL_VIDEODRIVER=dummy uv run python -m render.window --smoke-test`

- [ ] **Step 6: Review the branch and commit deliverables**

Review the diff for data leakage, action-direction mismatches, hidden reliance on display state, report reproducibility, and accidental inclusion of temporary checkpoints.

```bash
git add README.md models/best.pt reports/
git commit -m "docs: add neural solver training results"
```

