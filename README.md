# omp-gym

**Overnight skill gym for [OMP](https://github.com/MTEnt/oh-my-pi)** (and stock Oh My Pi).

Rust reimplementation of the *SkillOpt-Sleep* idea: harvest real agent sessions, mine recurring tasks, improve skill markdown behind a validation gate, stage changes for review, adopt into OMP skill dirs.

Inspired by [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) (MIT). This is a **clean-room Rust** project focused on OMP — not a port of the Python package.

## Why

- Keep **stock OMP** updating normally  
- Put *your* learning loop **outside** the agent binary  
- `/gym` overnight without Python/venv in the agent path  
- Path to an upstream OMP PR later (extension + docs first)

## Status (v0.1)

| Piece | State |
|---|---|
| OMP session harvest (JSONL) | done |
| Task mining | done |
| `status` / `dry-run` / `run` (mock stage) | done |
| macOS overnight schedule (launchd) | done |
| `/gym` OMP extension | done |
| Real replay via `omp -p` | next |
| Reflect + gated skill edit + adopt | next |

Mock nights **never** rewrite `SKILL.md`. Adopt refuses mock proposals.

## Install

```bash
git clone https://github.com/MTEnt/omp-gym.git
cd omp-gym
cargo install --path crates/omp-gym

# OMP slash command
mkdir -p ~/.omp/agent/extensions
cp extensions/omp/gym.ts ~/.omp/agent/extensions/
```

Optional: `export OMP_GYM_BIN=~/.cargo/bin/omp-gym`

## CLI

```bash
omp-gym doctor
omp-gym status --project .
omp-gym dry-run --project .
omp-gym run --project .                 # stages mock proposal
omp-gym schedule --hour 2 --minute 15   # macOS launchd
omp-gym schedule --off
omp-gym adopt --target-skill ~/.agents/skills/foo/SKILL.md
```

State lands in `<project>/.omp/gym/`.

## In OMP

```text
/gym doctor
/gym status
/gym dry-run
/gym run
/gym overnight
/gym overnight off
/gym adopt -- --target-skill ~/.agents/skills/foo/SKILL.md
```

## Pipeline

```text
OMP sessions (~/.omp/agent/sessions)
        │ harvest
        ▼
   mined tasks.json
        │ replay (planned: omp -p)
        ▼
   reflect → bounded skill edit
        │ validate on held-out tasks
        ▼
   staged proposal  ──you──►  adopt → SKILL.md
        │
        └── OMP loads skills from
            ~/.omp/agent/skills + ~/.agents/skills
```

## Upstream OMP PR plan

1. Prove `/gym` + harvest useful on real OMP sessions  
2. PR **extension + docs** to OMP (optional binary install via cargo)  
3. Later: session export schema / optional in-tree subcommand if wanted  

See `docs/UPSTREAM_PR.md`.

## License

MIT. SkillOpt is a separate MIT project by Microsoft; omp-gym does not vendor its Python source.
