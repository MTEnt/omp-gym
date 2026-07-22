# Complete OMP Gym Optimizer Design

Date: 2026-07-23
Status: Approved

## Goal

Build the missing production loop for `omp-gym` without forking OMP or fine-tuning model weights:

```text
harvest -> review -> replay -> reflect -> validate -> stage -> adopt
```

The implementation remains a standalone Rust CLI plus a thin stock-OMP extension. OMP provides authenticated model execution. The gym provides task curation, deterministic evaluation, evidence, scheduling, and safe skill adoption.

## Scope decisions

- Model execution is OMP-native only: replay, optimization, and supplemental judging invoke `omp -p`.
- The first complete release supports text-only isolated tasks. Tool-enabled repository tasks are out of scope.
- Deterministic checks exclusively control acceptance. Model judging is supplemental evidence.
- The acceptance gate requires a configured mean improvement and zero held-out task regressions.
- Passing candidates are staged for manual adoption. Overnight runs never rewrite a skill automatically.
- Release CI must publish installable binaries; a source-only repository or CI compilation result is not a release build.

## Alternatives considered

### OMP-native text-only replay

Selected. It reuses the user's OMP authentication and model configuration, minimizes credential handling, and permits an isolated, reproducible subprocess boundary.

### Direct provider API execution

Rejected for this release. It would improve structured-output control but add provider-specific credentials, clients, retry policies, and configuration alongside OMP.

### Tool-enabled worktree replay

Rejected for this release. It would broaden task coverage but require a sandbox, repository cleanup guarantees, side-effect oracles, and a materially larger prompt-injection surface.

## Architecture

```text
OMP session JSONL
      |
      v
harvest and stable merge
      |
      v
pending task store -- explicit approve/reject --> reviewed task suite
                                                   |
                                      deterministic train/validation split
                                                   |
                 +---------------------------------+---------------------------------+
                 |                                                                   |
                 v                                                                   v
       baseline replay on all tasks                              training trajectories and failures
                 |                                                                   |
                 |                                                          OMP optimizer call
                 |                                                                   |
                 |                                                        bounded candidate skill
                 |                                                                   |
                 +---------------------------------+---------------------------------+
                                                   |
                                                   v
                                      candidate held-out replay
                                                   |
                              deterministic scoring and optional judge
                                                   |
                        strict improvement plus zero-regression gate
                                                   |
                                  accepted proposal, diff, evidence
                                                   |
                                    manual atomic adopt and backup
```

The Rust core owns all contracts and state transitions. The CLI exposes them. The TypeScript extension resolves the binary and forwards `/gym` arguments without duplicating business logic.

## Task model

`.omp/gym/tasks.json` is schema-versioned and written atomically. Each task contains:

- A stable ID derived from SHA-256 of its normalized representative prompt.
- Title and complete replay prompt.
- Source session IDs, frequency, and first/last-seen timestamps.
- Review state: `pending`, `approved`, or `rejected`.
- Review timestamp and optional reviewer note.
- One or more typed deterministic checks.
- An optional rubric for supplemental pairwise judging.

Supported deterministic checks for the text-only release are:

- Exact trimmed response.
- Required substring, with explicit case sensitivity.
- Forbidden substring, with explicit case sensitivity.
- Rust regular expression.

Harvesting merges by stable ID, unions source sessions, updates frequency and timestamps, and preserves review state, checks, and rubrics. New tasks are always pending. An approved task without at least one valid check is invalid and cannot enter a run.

Review commands:

```text
omp-gym tasks list [--status pending|approved|rejected]
omp-gym tasks show <id>
omp-gym tasks approve <id> --contains <text> [--regex <pattern> ...] [--rubric <text>]
omp-gym tasks reject <id> [--note <text>]
omp-gym tasks reopen <id>
omp-gym tasks validate
```

The extension forwards the same command tail after `/gym`.

## Train-validation split

A run requires at least five approved tasks: at least three training tasks and at least two held-out validation tasks. Task IDs are ranked by a versioned salted hash, producing a deterministic split for the same task set. The run persists its exact split and task-store hash.

The optimizer receives only training prompts, checks, baseline outputs, and training scores. It never receives current-run validation prompts, checks, outputs, or scores. Validation data is provided only to baseline/candidate replay and the optional judge after candidate generation.

## OMP runner

All model roles use one subprocess abstraction with role-specific optional model selectors and timeouts. Each replay runs in a newly created empty temporary directory with a temporary append-system-prompt file containing exactly the current or candidate skill.

Conceptual invocation:

```text
omp -p --mode json --no-session --no-tools --no-skills \
  --no-extensions --no-rules --cwd <empty-directory> \
  --append-system-prompt <skill-file> [--model <selector>] <task-prompt>
```

A temporary OMP configuration overlay disables startup plan mode and other optional autonomous features that could alter a single-shot replay. The parent process captures bounded stdout/stderr, exit status, duration, and parsed NDJSON events. The terminal assistant text is extracted from `message_end`, with `agent_end` as a compatibility fallback.

Every trajectory records the role, task ID, prompt hash, skill hash, model metadata available from OMP events, redacted bounded events, final text, timing, and process outcome. Timeouts and malformed terminal events are explicit failures, never empty successful outputs.

## Scoring

Each deterministic check contributes one pass/fail unit. A task score is passed checks divided by total checks. Process success and nonempty terminal assistant text are mandatory invariants; violating either fails the task regardless of textual checks.

Validation acceptance requires all of the following:

1. Candidate mean deterministic score is at least the baseline mean plus the configured minimum delta.
2. Candidate score is greater than or equal to baseline score for every validation task.
3. At least one deterministic check changes from failing to passing.
4. Every replay invariant succeeds.
5. The candidate differs from the base skill and satisfies all structural/edit bounds.

A supplemental OMP judge receives paired baseline/candidate validation responses in randomized A/B order plus the task rubric. It returns a winner or tie and rationale. Judge failure does not reject an otherwise deterministic improvement and judge preference can never accept a deterministic regression.

## Candidate generation

The optimizer receives the complete current skill and training evidence only. It is instructed to emit a short summary and a complete proposed skill between strict sentinel markers. The parser rejects missing, duplicate, or malformed markers.

Candidate bounds are configurable and enforced before validation spend:

- Maximum total bytes.
- Maximum growth relative to the base skill.
- Maximum changed lines in the unified diff.
- Required valid skill frontmatter and nonempty body.
- No identical candidate, placeholders, or unresolved conflict markers.

The candidate generator gets one call. A malformed or out-of-bounds result fails the run rather than triggering an unbounded repair loop.

## Runs, evidence, and proposals

Local artifacts are private and gitignored:

```text
.omp/gym/
  config.json
  tasks.json
  state.json
  run.lock
  runs/<run-id>/
    run.json
    evidence.jsonl
  proposals/<proposal-id>/
    proposal.json
    candidate.SKILL.md
    skill.diff
  backups/<proposal-id>/SKILL.md
```

Every run gets an immutable run directory, including rejected and failed runs. Only a candidate that passes the strict gate gets a proposal directory and updates the latest-proposal pointer.

Proposal metadata includes schema version, IDs and timestamps, target path, base and candidate hashes, task-store hash, exact split, baseline/candidate scores, gate decision, judge evidence, edit bounds, and artifact paths.

Redaction remains best-effort and all prompt/event/output fields are size-bounded. User-facing commands warn that reviewed transcript-derived data may remain sensitive.

## Adoption

`omp-gym adopt [--proposal <id>]` performs a compare-and-swap transition:

1. Load an accepted, not-yet-adopted proposal.
2. Hash the current target and require it to equal the recorded base hash.
3. Copy the current skill into the proposal-specific backup directory.
4. Write and synchronize a temporary file beside the target.
5. Atomically rename the temporary file over the target.
6. Re-hash the target and require the candidate hash.
7. Atomically mark the proposal adopted in metadata and state.

If the target already has the candidate hash, adoption reports success idempotently. Any other hash mismatch refuses to overwrite user changes. Mock or rejected artifacts can never be adopted.

## Configuration and commands

Project-local `.omp/gym/config.json` stores the canonicalized project, target skill, OMP binary override, optional role models, per-role timeout, validation ratio/minimums, score delta, candidate bounds, judge toggle, and scheduler settings. CLI flags and documented environment variables may override values for one invocation.

Primary commands:

```text
omp-gym configure
omp-gym doctor
omp-gym harvest                 # local-only; `dry-run` remains an alias
omp-gym tasks ...
omp-gym run                     # real overnight-equivalent optimizer run
omp-gym status
omp-gym proposal show [id]
omp-gym proposal diff [id]
omp-gym adopt [--proposal id]
omp-gym schedule [--hour H --minute M | --off]
```

`run` acquires an exclusive project lock before any model call. It validates configuration, skill structure, task checks, and split sufficiency before spending model calls.

## Scheduling and extension

macOS launchd uses a project-specific stable label derived from the canonical project path, so scheduling one project cannot replace another. The plist invokes `omp-gym run --project <path>` and relies on persisted project configuration. Logs remain project-local. Schedule removal addresses only that project's label.

The OMP extension remains thin. It resolves a configured binary or common install locations, executes with the active OMP project directory, XML-escapes output, and maps command aliases to the real CLI surface. Its help must not advertise nonexistent behavior.

## Failure handling

- Invalid target, checks, split, or configuration: fail before model spend.
- Existing project lock: refuse concurrent execution with owner/run context when available.
- OMP timeout or nonzero exit: record bounded evidence and fail the run.
- Malformed NDJSON or missing terminal response: fail explicitly.
- Optimizer parse/bounds failure: retain run evidence, stage no proposal.
- Gate rejection: retain scores and evidence, stage no proposal, exit successfully as a completed non-improving night.
- Atomic-write failure: leave the previous complete file in place.
- Adoption base-hash mismatch: refuse replacement and preserve both user file and candidate.

No failure silently resets persisted state. Corrupt state/config/task files return contextual parse errors.

## Release distribution

The repository must publish actual release assets, not merely compile in CI. A tag-driven release workflow will build and archive at least:

- macOS arm64.
- macOS x86_64.
- Linux x86_64 GNU.
- Linux arm64 GNU where a supported runner/toolchain is available.

Archives include the `omp-gym` binary, license, and concise installation information. The workflow creates checksums, uploads immutable assets to a GitHub Release, and exercises the packaged binary before upload. Normal pull-request CI continues formatting, Clippy, tests, and release compilation.

## Verification

Automated verification includes:

- Unit tests for stable IDs, merge preservation, check validation/scoring, deterministic splits, candidate parsing/bounds, strict gate cases, hashes, diffs, and corrupt-state failures.
- Integration tests against a deterministic fake `omp` executable covering success, timeout, nonzero exit, malformed NDJSON, and missing terminal output.
- CLI contract tests for configure, task review, run, proposal, and adoption.
- Compare-and-swap and idempotent adoption tests with backup verification.
- Extension autodiscovery and command-forwarding smoke tests.
- Scheduler label/configuration tests.
- CI validation of packaged release assets and checksums.

Final acceptance also requires a real OMP end-to-end scenario using a temporary deficient skill and at least five reviewed tasks. The baseline must fail a general deterministic requirement, the optimizer must infer a bounded general instruction from training evidence only, the held-out candidate must improve with zero regression, a proposal must stage, adoption must produce the candidate hash and backup, and restoration must recover the original skill.

## Implementation order

1. Correct release distribution immediately so users can install the existing honest prototype.
2. Add versioned config, task, run, trajectory, score, and proposal contracts.
3. Implement stable task merging and the explicit review CLI.
4. Implement isolated OMP subprocess execution and trajectory parsing.
5. Implement checks, splits, scoring, and strict gating.
6. Implement optimizer candidate parsing, structural bounds, and diffs.
7. Implement proposal staging and atomic adoption.
8. Wire the complete run, scheduler, and extension.
9. Add contract/integration coverage, real OMP smoke verification, documentation, and tagged release proof.
