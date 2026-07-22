# omp-gym design

## Goal

Build an overnight skill gym for OMP without forking OMP or fine-tuning model weights.

## Implemented in v0.1

The current executable is the data-preparation layer:

1. Read OMP session JSONL.
2. Keep sessions recorded in the selected project or its descendants.
3. Apply best-effort redaction to user excerpts.
4. Cluster representative tasks with a token-similarity heuristic.
5. Write local, gitignored task and state artifacts.
6. Stage mock proposal metadata for testing the review workflow.

No model is invoked. No skill is evaluated or changed.

## Planned optimizer loop

1. **Harvest** project-specific OMP sessions.
2. **Review** mined tasks before model replay.
3. **Replay** tasks with the current skill.
4. **Reflect** on failures and propose a bounded markdown edit.
5. **Validate** current and candidate skills on held-out tasks.
6. **Stage** only a strict improvement with scores and evidence.
7. **Adopt** after explicit user approval, with a backup.

## Non-goals

- Full SkillOpt benchmark and WebUI compatibility.
- Automatic adoption without user action.
- Maintaining an OMP core fork.

## Trust boundaries

- Harvesting is local and read-only.
- `.omp/gym/` artifacts are locally gitignored.
- Redaction is best-effort; generated task files may still contain sensitive conversation data.
- Future real backends will require explicit disclosure before sending task text to a model.
- Mock proposals can never be adopted.

## OMP integration

- Rust binary: `omp-gym`
- OMP extension: `extensions/omp/gym.ts`
- OMP command: `/gym`
- Skills remain normal `SKILL.md` files loaded by stock OMP.

## Next implementation milestone

- Reviewed-task state with stable task identities.
- `omp -p` replay runner.
- Candidate skill generation.
- Deterministic evaluator and held-out gate.
- Unified-diff proposal format.
- Adopt with backup and atomic replacement.
