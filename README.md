# omp-gym

**An evidence-driven prompt gym for [OMP](https://github.com/MTEnt/oh-my-pi).**

A **prompt gym** is a feedback loop for improving reusable AI instructions from evidence instead of guesswork. It turns real work into representative tasks, measures how the current instructions perform, proposes a bounded improvement, and tests that candidate on tasks it was not trained on. In OMP, those reusable instructions are typically a skill's `SKILL.md` file.

## Why use a prompt gym?

- **Learn from your work.** Improvements come from the projects and tasks where you actually use OMP, not a generic benchmark.
- **Measure changes repeatably.** Explicit checks make "better" more than a subjective impression.
- **Reduce overfitting.** Training tasks guide a candidate while held-out tasks independently decide whether it improved.
- **Keep humans in control.** The gym stages evidence and a reviewable diff; it never needs to rewrite a live skill automatically.

## How it works with OMP

OMP records the sessions where work happens. `omp-gym` is a project-scoped companion that turns those sessions into an improvement loop:

1. **Harvest** OMP sessions that started in the selected project.
2. **Mine and review** representative tasks and their deterministic success checks.
3. **Replay a baseline** through OMP using the current skill.
4. **Generate one bounded candidate** from training-task evidence.
5. **Validate on held-out tasks** and accept the candidate only when the deterministic gate improves.
6. **Stage a proposal** containing the candidate `SKILL.md`, diff, scores, and replay evidence for human review and explicit adoption.

This design is inspired by [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) (MIT), but `omp-gym` is a clean-room Rust project built for OMP rather than a port of SkillOpt's Python package.

> **Implementation boundary:** The workflow above describes the intended complete system. Today's user-facing CLI still stops after session harvesting, task mining, and mock proposal staging. The codebase contains foundational task-review, replay, evaluation, and proposal-safety components, but the CLI does not yet connect them into the full optimization and adoption loop. The status table below is the authoritative list of what users can run now.

## Current status

`omp-gym` v0.1 is a working **session-harvest and task-mining prototype**. It is not yet a working skill optimizer.

| Capability | Status |
|---|---|
| Read OMP session JSONL | working |
| Restrict sessions to the selected project | working |
| Best-effort prompt redaction | working |
| Cluster representative tasks | working, heuristic |
| `doctor`, `status`, `dry-run` | working |
| Mock proposal staging with `run` | working |
| OMP `/gym` extension | working |
| macOS launchd scheduling | working for one scheduled project |
| Replay tasks through `omp -p` | not implemented |
| Reflect and generate a skill edit | not implemented |
| Validate on held-out tasks | not implemented |
| Apply a proposal with `adopt` | not implemented |

`run` creates a mock review artifact. It never changes `SKILL.md`. `adopt` intentionally refuses the mock artifacts v0.1 can generate.

## Requirements

- Rust toolchain with Cargo
- OMP for the `/gym` extension and real OMP session history
- macOS only for the built-in launchd scheduler

The CLI reads `PI_CODING_AGENT_DIR` when OMP supplies a custom agent directory. Otherwise it defaults to `~/.omp/agent`.

## Install

### Prebuilt release

Download `SHA256SUMS` and the archive for your machine from the
[latest GitHub release](https://github.com/MTEnt/omp-gym/releases/latest):

| Platform | Archive target |
|---|---|
| macOS on Apple Silicon | `aarch64-apple-darwin` |
| macOS on Intel | `x86_64-apple-darwin` |
| Linux on ARM64 | `aarch64-unknown-linux-gnu` |
| Linux on x86-64 | `x86_64-unknown-linux-gnu` |

Verify the downloaded archive before extracting it:

```bash
# macOS
grep "  $(basename "$ARCHIVE")$" SHA256SUMS | shasum -a 256 --check

# Linux
grep "  $(basename "$ARCHIVE")$" SHA256SUMS | sha256sum --check
```

Install the binary and bundled OMP extension:

```bash
tar -xzf "$ARCHIVE"
cd "$(basename "$ARCHIVE" .tar.gz)"
install -m 755 omp-gym ~/.local/bin/omp-gym
mkdir -p ~/.omp/agent/extensions
cp extensions/omp/gym.ts ~/.omp/agent/extensions/gym.ts
```

Ensure `~/.local/bin` is on `PATH`, then start a new OMP session so it discovers `/gym`.

### Build from source

```bash
git clone https://github.com/MTEnt/omp-gym.git
cd omp-gym
cargo install --locked --path crates/omp-gym
mkdir -p ~/.omp/agent/extensions
cp extensions/omp/gym.ts ~/.omp/agent/extensions/gym.ts
```

Verify the installation:

```bash
omp-gym --version
omp-gym --project . doctor
```

If the binary is somewhere other than `~/.cargo/bin/omp-gym` or
`~/.local/bin/omp-gym`, set `OMP_GYM_BIN` to its absolute path before starting OMP.

## Run it today

Run these commands from the project whose sessions you want to inspect.

### 1. Check paths

```bash
omp-gym --project . doctor
```

Confirm that the project and sessions root are correct.

### 2. Harvest sessions and mine tasks

```bash
omp-gym \
  --project . \
  --lookback-hours 168 \
  --max-sessions 50 \
  --max-tasks 20 \
  dry-run
```

Inspect:

```text
<project>/.omp/gym/tasks.json
<project>/.omp/gym/state.json
```

Only sessions whose recorded `cwd` is the selected project or one of its descendants are included.

If it reports zero sessions:

1. Retry with `--lookback-hours 0 --max-sessions 500`.
2. Confirm `omp-gym doctor` points at the OMP profile you actually use.
3. Confirm those OMP sessions were started in this project. The session's recorded startup `cwd`, not paths mentioned later in chat, controls project matching.

### 3. Stage a mock proposal

```bash
omp-gym --project . run
omp-gym --project . status
```

This writes metadata under:

```text
<project>/.omp/gym/proposals/
```

It does **not** replay tasks, evaluate a skill, generate a patch, or modify a skill.

### 4. Use the same flow inside OMP

```text
/gym doctor
/gym dry-run
/gym run
/gym status
```

The extension uses the current OMP project directory.

## Optional overnight snapshots

Do not schedule this expecting autonomous skill improvement yet. In v0.1 it only repeats the mock harvest/mining run.

Install a daily macOS launchd job at 02:15 local time:

```bash
omp-gym --project /absolute/path/to/project schedule --hour 2 --minute 15
```

Or in OMP:

```text
/gym overnight
```

Check its state and logs:

```bash
omp-gym --project /absolute/path/to/project status
cat /absolute/path/to/project/.omp/gym/logs/launchd.out.log
cat /absolute/path/to/project/.omp/gym/logs/launchd.err.log
```

Remove it:

```bash
omp-gym --project /absolute/path/to/project schedule --off
```

Or:

```text
/gym overnight off
```

The current scheduler uses one launchd label (`com.mtent.omp-gym`), so scheduling another project replaces the previous job.

## Data handling

`tasks.json` contains excerpts copied from your OMP conversations. Redaction is best-effort, not a guarantee. Review the file before sharing it.

The generated `.omp/gym/` directory contains its own `.gitignore`, so transcript-derived artifacts are not committed accidentally.

## Development checks

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo build --workspace --release
```

## Planned optimizer

The intended next stages are:

1. Replay reviewed tasks through `omp -p`.
2. Generate a bounded candidate edit for a selected `SKILL.md`.
3. Evaluate current and candidate skills on held-out tasks.
4. Stage only a strict improvement with evidence.
5. Apply an accepted non-mock proposal with backup and explicit user action.

Until those stages exist, call this a harvester/miner prototype, not a SkillOpt replacement.

## License

MIT. SkillOpt is a separate MIT project by Microsoft; `omp-gym` does not vendor its Python source.
