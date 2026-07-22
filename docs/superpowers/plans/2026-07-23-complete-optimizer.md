# Complete OMP Gym Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mock harvester prototype with a complete, review-first OMP skill optimizer that replays approved text tasks through `omp -p`, accepts only deterministic held-out improvements, stages evidence, and adopts candidates safely.

**Architecture:** The Rust core owns versioned task/config/run/proposal contracts, isolated OMP subprocess execution, deterministic evaluation, candidate bounds, and atomic persistence. The CLI exposes review/run/proposal/adopt operations; the TypeScript extension remains a thin command adapter. Model judging is supplemental, while deterministic checks and zero-regression rules exclusively govern acceptance.

**Tech Stack:** Rust 2021, Clap, Serde/JSON/YAML, SHA-256, regex, `wait-timeout`, `fs2`, `similar`, OMP NDJSON print mode, macOS launchd, TypeScript OMP extension, GitHub Actions.

**Approved design:** `docs/plans/2026-07-23-complete-optimizer-design.md`

---

## File structure

### Core crate

- Create `crates/omp-gym-core/src/privacy.rs`: shared bounded redaction for transcript and model evidence.
- Create `crates/omp-gym-core/src/task_store.rs`: stable IDs, v1 migration, merge, review mutations, and suite validation.
- Create `crates/omp-gym-core/src/evaluation.rs`: check validation/scoring, deterministic split, and strict gate.
- Create `crates/omp-gym-core/src/runner.rs`: isolated OMP subprocess, timeout/process-group handling, NDJSON parsing, and trajectories.
- Create `crates/omp-gym-core/src/optimizer.rs`: optimizer/judge prompts, candidate parsing, frontmatter/edit bounds, and unified diffs.
- Modify `crates/omp-gym-core/src/types.rs`: versioned domain contracts.
- Modify `crates/omp-gym-core/src/config.rs`: persisted project configuration and validation.
- Modify `crates/omp-gym-core/src/state.rs`: atomic JSON/pointer writes and proposal/run loading.
- Modify `crates/omp-gym-core/src/paths.rs`: run/proposal/backup paths and atomic-write helper.
- Modify `crates/omp-gym-core/src/mine.rs`: deterministic provisional IDs.
- Modify `crates/omp-gym-core/src/harvest.rs`: use shared privacy helpers.
- Replace `crates/omp-gym-core/src/pipeline.rs`: complete orchestration and compare-and-swap adoption.
- Modify `crates/omp-gym-core/src/lib.rs`: honest module docs and exports.

### CLI and integration

- Replace `crates/omp-gym/src/main.rs`: persisted config overlays, nested task/proposal commands, real run/adopt, and project-specific scheduling.
- Modify `extensions/omp/gym.ts`: real help/aliases, quoted argument parsing, timeout, and release binary lookup.
- Create `crates/omp-gym/tests/cli_flow.rs`: binary-level task review and proposal/adopt contracts.
- Create `crates/omp-gym-core/tests/fake_omp_pipeline.rs`: subprocess and complete pipeline integration against a deterministic fake OMP.

### Documentation and release

- Modify `README.md`: prompt-gym definition, SkillOpt explanation, OMP Gym differences, real workflow, safety, and release install.
- Modify `docs/DESIGN.md`: implemented architecture and invariants.
- Modify `docs/UPSTREAM_PR.md`: current upstream integration boundary.
- Modify `Cargo.toml`, both crate manifests, and `Cargo.lock`: dependencies and final version.
- Modify `.github/workflows/ci.yml` and `.github/workflows/release.yml` only if new tests require additional commands.

---

### Task 1: Versioned contracts and atomic persistence

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/omp-gym-core/Cargo.toml`
- Modify: `crates/omp-gym-core/src/types.rs`
- Modify: `crates/omp-gym-core/src/paths.rs`
- Modify: `crates/omp-gym-core/src/state.rs`
- Test: unit tests in `types.rs`, `paths.rs`, and `state.rs`

- [ ] **Step 1: Write failing serialization and atomic-write tests**

Add tests that round-trip every schema, reject corrupt JSON contextually, and prove a second atomic write replaces the complete first value rather than appending or truncating. Use the observable contract:

```rust
#[test]
fn atomic_json_replaces_complete_document() {
    let root = unique_test_dir("atomic-json");
    let path = root.join("state.json");
    atomic_write_json(&path, &serde_json::json!({"value": 1})).unwrap();
    atomic_write_json(&path, &serde_json::json!({"value": 2})).unwrap();
    let value: serde_json::Value = load_json(&path, "state").unwrap();
    assert_eq!(value, serde_json::json!({"value": 2}));
    assert!(!root.join("state.json.tmp").exists());
}
```

- [ ] **Step 2: Run tests and verify the missing APIs fail**

Run: `cargo test -p omp-gym-core atomic_json_replaces_complete_document -- --nocapture`

Expected: compile failure because `atomic_write_json` and `load_json` do not exist.

- [ ] **Step 3: Add dependencies and exact contracts**

Add workspace dependencies `sha2 = "0.10"`, `fs2 = "0.4"`, `similar = "2"`, `tempfile = "3"`, `wait-timeout = "0.2"`, `serde_yaml = "0.9"`, and `libc = "0.2"`.

Define schema-version constant `SCHEMA_VERSION: u32 = 2` and these public shapes in `types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus { Pending, Approved, Rejected }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CheckSpec {
    Exact { value: String },
    Contains { value: String, case_sensitive: bool },
    NotContains { value: String, case_sensitive: bool },
    Regex { pattern: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinedTask {
    pub id: String,
    pub title: String,
    pub prompt: String,
    pub source_session_ids: Vec<String>,
    pub frequency: usize,
    pub status: ReviewStatus,
    pub checks: Vec<CheckSpec>,
    pub rubric: Option<String>,
    pub review_note: Option<String>,
    pub reviewed_at: Option<DateTime<Utc>>,
    pub first_seen_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
}
```

Also define `ModelRole`, `Trajectory`, `CheckResult`, `TaskScore`, `GateDecision`, `RunStatus`, `RunRecord`, `ProposalStatus`, `StagedProposal`, and `TasksFile`. Define `TaskSplit { train_ids: Vec<String>, validation_ids: Vec<String> }` so the persisted split never borrows task records. All persisted top-level records carry `schema_version`; proposals carry base/candidate/task-store hashes and the same exact train/validation IDs.

- [ ] **Step 4: Implement atomic file helpers**

Implement `atomic_write(path, bytes)` by creating a UUID-suffixed temporary file beside the destination, writing and `sync_all`-ing it, preserving existing permissions when present, renaming over the destination, and synchronizing the parent directory on Unix. Build typed `load_json` and `atomic_write_json` helpers with path-rich `anyhow::Context`.

- [ ] **Step 5: Run focused and crate tests**

Run: `cargo test -p omp-gym-core types:: paths:: state:: -- --nocapture`

Expected: all new round-trip, corrupt-state, and atomic-replacement tests pass.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock crates/omp-gym-core/Cargo.toml crates/omp-gym-core/src/{types,paths,state}.rs
git commit -m "feat: add versioned gym contracts"
```

### Task 2: Stable task store and explicit review workflow

**Files:**
- Create: `crates/omp-gym-core/src/task_store.rs`
- Modify: `crates/omp-gym-core/src/mine.rs`
- Modify: `crates/omp-gym-core/src/pipeline.rs`
- Modify: `crates/omp-gym-core/src/lib.rs`
- Test: unit tests in `task_store.rs` and `mine.rs`

- [ ] **Step 1: Write failing identity and merge tests**

Cover identical prompts producing identical IDs, reordering sessions not changing IDs, a longer representative prompt preserving an approved existing task via token similarity, source-ID union/deduplication, rejected tasks staying rejected, and newly mined tasks staying pending.

```rust
#[test]
fn merge_preserves_reviewed_task_contract() {
    let mut existing = approved_task("task-existing", "Fix login failures", contains("resolved"));
    existing.review_note = Some("owner reviewed".into());
    let incoming = pending_task(stable_task_id("Please fix login failures in auth"), "Please fix login failures in auth");
    let merged = merge_tasks(vec![existing], vec![incoming], Utc::now()).unwrap();
    assert_eq!(merged[0].id, "task-existing");
    assert_eq!(merged[0].status, ReviewStatus::Approved);
    assert_eq!(merged[0].checks, vec![contains("resolved")]);
    assert_eq!(merged[0].review_note.as_deref(), Some("owner reviewed"));
}
```

- [ ] **Step 2: Run and observe failure**

Run: `cargo test -p omp-gym-core task_store -- --nocapture`

Expected: compile failure because `task_store` is not defined.

- [ ] **Step 3: Implement stable identity and merge**

Expose:

```rust
pub fn stable_task_id(prompt: &str) -> String;
pub fn load_tasks(path: &Path, project: &Path) -> Result<TasksFile>;
pub fn merge_tasks(existing: Vec<MinedTask>, mined: Vec<MinedTask>, now: DateTime<Utc>) -> Result<Vec<MinedTask>>;
pub fn save_tasks(path: &Path, file: &TasksFile) -> Result<()>;
pub fn approve_task(file: &mut TasksFile, id: &str, checks: Vec<CheckSpec>, rubric: Option<String>, note: Option<String>) -> Result<()>;
pub fn reject_task(file: &mut TasksFile, id: &str, note: Option<String>) -> Result<()>;
pub fn reopen_task(file: &mut TasksFile, id: &str) -> Result<()>;
pub fn validate_reviewed_tasks(file: &TasksFile) -> Result<Vec<&MinedTask>>;
```

Normalize whitespace, URLs, paths, and long hexadecimal IDs exactly once in a shared function. Sort normalized excerpt candidates before clustering so session enumeration order cannot alter provisional IDs. Hash normalized UTF-8 with SHA-256 and emit `task-` plus 24 lowercase hex characters. Exact IDs merge first; otherwise match an existing task only when significant-token Jaccard similarity is at least `0.70`, choosing the unique highest score. Ambiguous ties remain new pending tasks.

Load v1 files through explicit `TasksFileV1`/`MinedTaskV1` structs. Migrate every legacy item to pending because v1 review flags had no deterministic checks.

- [ ] **Step 4: Wire harvest to merge instead of overwrite**

Replace direct `tasks.json` writes in `dry_run` with `load_tasks`, `merge_tasks`, and `save_tasks`. Keep review state across harvests. Update report notes with counts for new, preserved-approved, and total tasks.

- [ ] **Step 5: Run focused tests**

Run: `cargo test -p omp-gym-core task_store mine pipeline::tests::dry_run -- --nocapture`

Expected: stable-ID, migration, merge-preservation, and existing project-filter tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/omp-gym-core/src/{task_store,mine,pipeline,lib}.rs
git commit -m "feat: persist reviewed gym tasks"
```

### Task 3: Deterministic checks, held-out split, and gate

**Files:**
- Create: `crates/omp-gym-core/src/evaluation.rs`
- Modify: `crates/omp-gym-core/src/lib.rs`
- Test: unit tests in `evaluation.rs`

- [ ] **Step 1: Write the gate truth-table tests**

Test exact/contains/not-contains/regex behavior, invalid and empty checks, process/nonempty invariants, deterministic split stability, minimum three train/two validation tasks, mean-delta rejection, one-task regression rejection, and strict acceptance.

```rust
#[test]
fn strict_gate_requires_improvement_without_regression() {
    let baseline = vec![score("v1", 0.0), score("v2", 1.0)];
    let improved = vec![score("v1", 1.0), score("v2", 1.0)];
    assert!(gate(&baseline, &improved, 0.25).accepted);
    let regressed = vec![score("v1", 1.0), score("v2", 0.0)];
    let decision = gate(&baseline, &regressed, 0.25);
    assert!(!decision.accepted);
    assert!(decision.reasons.iter().any(|reason| reason.contains("v2 regressed")));
}
```

- [ ] **Step 2: Run and observe failure**

Run: `cargo test -p omp-gym-core evaluation -- --nocapture`

Expected: compile failure because the evaluation module does not exist.

- [ ] **Step 3: Implement evaluation APIs**

Expose:

```rust
pub fn validate_check(check: &CheckSpec) -> Result<()>;
pub fn score_trajectory(task: &MinedTask, trajectory: &Trajectory) -> TaskScore;
pub fn split_tasks(tasks: &[&MinedTask], validation_ratio: f64, min_validation: usize) -> Result<TaskSplit>;
pub fn gate(baseline: &[TaskScore], candidate: &[TaskScore], min_delta: f64) -> GateDecision;
```

Trim only for `Exact`; preserve raw text for other checks. Compile regexes during suite validation. Split by sorting `SHA256("omp-gym-split-v1\0" + task.id)` bytes, then allocate `max(min_validation, ceil(total * ratio))` validation tasks while retaining at least three train tasks. Gate task IDs one-to-one, require candidate mean minus baseline mean to meet `min_delta` (with `1e-9` tolerance), reject every per-task regression, require at least one newly passed check, and reject invariant failures.

- [ ] **Step 4: Run tests**

Run: `cargo test -p omp-gym-core evaluation -- --nocapture`

Expected: all gate and split tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/omp-gym-core/src/{evaluation,lib}.rs
git commit -m "feat: add deterministic held-out gate"
```

### Task 4: Persisted configuration and isolated OMP runner

**Files:**
- Modify: `crates/omp-gym-core/src/config.rs`
- Create: `crates/omp-gym-core/src/privacy.rs`
- Create: `crates/omp-gym-core/src/runner.rs`
- Modify: `crates/omp-gym-core/src/harvest.rs`
- Modify: `crates/omp-gym-core/src/lib.rs`
- Test: unit tests in `config.rs`, `privacy.rs`, and `runner.rs`

- [ ] **Step 1: Write failing runner parser/config tests**

Test project-relative target resolution, invalid thresholds/timeouts, redaction inside nested event JSON, `message_end` extraction, `agent_end` fallback, malformed NDJSON, missing assistant terminal event, output truncation, nonzero exit, and timeout.

Use a Unix fake executable that writes a valid OMP event:

```rust
write_executable(&fake, r#"#!/bin/sh
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"GYM_OK result"}],"stopReason":"stop"}}'
"#);
let trajectory = OmpRunner::new(config_with_bin(fake)).run(&request).unwrap();
assert!(trajectory.process_success);
assert_eq!(trajectory.final_text.as_deref(), Some("GYM_OK result"));
```

- [ ] **Step 2: Run and observe failure**

Run: `cargo test -p omp-gym-core runner config privacy -- --nocapture`

Expected: compile failure for missing modules/APIs.

- [ ] **Step 3: Implement persisted configuration**

`GymConfig::load(project)` starts from safe defaults, then reads `.omp/gym/config.json` if present. `GymConfig::save()` writes atomically. Include canonical project, sessions root, a target skill that is optional for harvesting but required by `validate_for_run`, `omp_bin` defaulting to `omp`, optional replay/optimizer/judge model selectors, replay/optimizer/judge timeout seconds, judge enabled, validation ratio `0.40`, minimum validation `2`, minimum score delta `0.05`, maximum output bytes `1_048_576`, candidate bytes `32_768`, growth ratio `1.5`, and changed lines `120`. `validate_for_run()` canonicalizes the target and rejects bad bounds before model spend.

- [ ] **Step 4: Implement shared privacy helpers**

Move transcript redaction into `privacy.rs` and add recursive JSON-string redaction plus Unicode-safe character bounding. Preserve the existing secret/key/email patterns and tests; add bearer tokens and common private-key headers.

- [ ] **Step 5: Implement the runner**

Define:

```rust
pub struct ModelRequest<'a> { pub role: ModelRole, pub prompt: &'a str, pub skill: &'a str }
pub trait ModelRunner { fn run(&self, request: &ModelRequest<'_>) -> Result<Trajectory>; }
pub struct OmpRunner { config: GymConfig }
```

For each call create a `tempfile::TempDir`, write `skill.md` and this overlay:

```yaml
advisor:
  enabled: false
prewalk:
  enabled: false
plan:
  defaultOnStartup: false
```

Spawn `omp -p --mode json --no-session --no-tools --no-skills --no-extensions --no-rules --no-prewalk --no-title --cwd <temp> --config <overlay> --append-system-prompt <skill> --max-time <seconds> [--model selector] <prompt>`. Put the child in a Unix process group, drain stdout/stderr on separate threads while retaining only configured bytes, enforce an external `wait-timeout`, kill the process group on timeout, and always return a trajectory for a spawned process. Parse line-delimited JSON, extract the latest assistant `message_end`, then `agent_end` fallback; record parse/process errors explicitly.

- [ ] **Step 6: Run focused tests**

Run: `cargo test -p omp-gym-core runner config privacy harvest -- --nocapture`

Expected: success, failure, timeout, fallback, bounds, and redaction tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/omp-gym-core/src/{config,privacy,runner,harvest,lib}.rs
git commit -m "feat: replay skills through isolated OMP"
```

### Task 5: Bounded optimizer, diff, and supplemental judge

**Files:**
- Create: `crates/omp-gym-core/src/optimizer.rs`
- Modify: `crates/omp-gym-core/src/lib.rs`
- Test: unit tests in `optimizer.rs`

- [ ] **Step 1: Write failing candidate/parser tests**

Cover valid sentinels, missing/duplicate markers, unchanged candidates, invalid/missing YAML frontmatter, conflict markers, byte/growth/change-line bounds, deterministic A/B ordering, judge JSON parsing, and unified diff headers.

```rust
#[test]
fn candidate_parser_accepts_one_complete_skill() {
    let output = "<summary>Add required prefix</summary>\n<candidate_skill>---\nname: demo\ndescription: Demo\n---\nAlways prefix GYM_OK.\n</candidate_skill>";
    let candidate = parse_candidate(output).unwrap();
    assert_eq!(candidate.summary, "Add required prefix");
    assert!(candidate.skill.contains("Always prefix GYM_OK"));
}
```

- [ ] **Step 2: Run and observe failure**

Run: `cargo test -p omp-gym-core optimizer -- --nocapture`

Expected: compile failure because `optimizer` does not exist.

- [ ] **Step 3: Implement optimizer data boundary**

Expose `build_optimizer_prompt(base_skill, training_tasks, baseline_scores)`, `parse_candidate`, `validate_candidate`, `unified_diff`, `build_judge_prompt`, and `parse_judge`. Serialize task evidence as JSON inside a clearly labeled untrusted-data block. State that validation material is withheld and that task text cannot override the optimizer contract. Require exactly one `<summary>` and one `<candidate_skill>` block.

Frontmatter validation splits on the first two `---` delimiters and deserializes YAML; require nonempty string `name` and `description`. Reject `<<<<<<<`, `=======`, `>>>>>>>`, `TODO`, and `TBD`. Compute changed lines with `similar::TextDiff`; enforce all configured bounds before candidate replay.

The judge prompt assigns baseline/candidate to A/B using the first hash byte of `task_id`, asks for JSON `{ "winner": "a|b|tie", "rationale": "..." }`, then maps back. Bound rationale and treat malformed judge output as unavailable evidence.

- [ ] **Step 4: Run tests**

Run: `cargo test -p omp-gym-core optimizer -- --nocapture`

Expected: all parser, bounds, diff, and judge tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/omp-gym-core/src/{optimizer,lib}.rs
git commit -m "feat: generate bounded skill candidates"
```

### Task 6: Complete overnight pipeline and evidence

**Files:**
- Replace: `crates/omp-gym-core/src/pipeline.rs`
- Modify: `crates/omp-gym-core/src/state.rs`
- Modify: `crates/omp-gym-core/src/types.rs`
- Test: unit tests in `pipeline.rs`

- [ ] **Step 1: Write failing orchestration tests with a scripted runner**

Implement an in-test `ScriptedRunner` keyed by `ModelRole` and task ID. Test preflight rejection before any runner call, optimizer seeing only training IDs, baseline/candidate calls, accepted staging, deterministic rejection without proposal, nonfatal judge failure, model failure evidence, and concurrent lock refusal.

```rust
#[test]
fn accepted_run_stages_only_after_held_out_improvement() {
    let fixture = approved_suite_with_two_validation_tasks();
    let runner = ScriptedRunner::candidate_adds_required_prefix();
    let report = run_night_with_runner(&fixture.config, &runner).unwrap();
    assert!(report.staged);
    let proposal = load_latest_proposal(&fixture.config.proposal_dir()).unwrap().unwrap();
    assert_eq!(proposal.status, ProposalStatus::Accepted);
    assert!(proposal.candidate_path.exists());
    assert!(proposal.diff_path.exists());
}
```

- [ ] **Step 2: Run and observe failure**

Run: `cargo test -p omp-gym-core pipeline -- --nocapture`

Expected: failures because the current pipeline is mock-only.

- [ ] **Step 3: Implement the exact run state machine**

`run_night` constructs `OmpRunner`; `run_night_with_runner` accepts any `ModelRunner`. Both:

1. Acquire an exclusive `.omp/gym/run.lock` with `fs2`.
2. Validate config/target/checks/split before a model call.
3. Harvest and merge tasks, then snapshot task-store and base-skill hashes.
4. Create immutable `runs/<uuid>/run.json` with status `running`.
5. Replay baseline on all approved tasks.
6. Build the optimizer request from training tasks/scores only.
7. Parse and bound the candidate.
8. Replay candidate on validation tasks only.
9. Score and apply the strict deterministic gate.
10. If enabled, judge held-out A/B pairs; store but never gate on them.
11. Write redacted bounded trajectories to `evidence.jsonl`.
12. On acceptance, atomically create `proposals/<id>/proposal.json`, `candidate.SKILL.md`, `skill.diff`, and `LATEST`.
13. Finalize run/state atomically as accepted, rejected, or failed.

A deterministic rejection returns `Ok(GymReport { staged: false, ... })`; infrastructure/model/parse failures persist evidence then return `Err`. Never update `LATEST` for a rejection or failure.

- [ ] **Step 4: Run pipeline tests**

Run: `cargo test -p omp-gym-core pipeline -- --nocapture`

Expected: all orchestration, evidence, lock, and staging tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/omp-gym-core/src/{pipeline,state,types}.rs
git commit -m "feat: complete overnight optimizer pipeline"
```

### Task 7: Atomic compare-and-swap adoption

**Files:**
- Modify: `crates/omp-gym-core/src/pipeline.rs`
- Modify: `crates/omp-gym-core/src/state.rs`
- Test: unit tests in `pipeline.rs`

- [ ] **Step 1: Write failing adoption tests**

Test accepted adoption, backup byte equality, final candidate hash, idempotent second adoption, base-hash mismatch preserving user edits, rejection/mock refusal, missing candidate refusal, and proposal metadata transition to adopted.

- [ ] **Step 2: Run and observe failure**

Run: `cargo test -p omp-gym-core adopt -- --nocapture`

Expected: current non-mock adoption failure.

- [ ] **Step 3: Implement adoption**

Expose `adopt(cfg, proposal_id: Option<&str>)`. Resolve explicit or latest accepted proposal, verify all artifact hashes, compare the current target hash to base hash, and allow candidate-hash idempotence. Copy current bytes to `.omp/gym/backups/<proposal-id>/SKILL.md`, write a synchronized temporary file beside the target with original permissions, atomically rename, verify candidate hash, then atomically update proposal/state. Any precondition failure leaves target bytes unchanged.

- [ ] **Step 4: Run focused tests**

Run: `cargo test -p omp-gym-core adopt -- --nocapture`

Expected: all compare-and-swap and backup tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/omp-gym-core/src/{pipeline,state}.rs
git commit -m "feat: adopt validated skills atomically"
```

### Task 8: Complete CLI review/config/proposal surface

**Files:**
- Replace: `crates/omp-gym/src/main.rs`
- Create: `crates/omp-gym/tests/cli_flow.rs`
- Modify: `crates/omp-gym/Cargo.toml`

- [ ] **Step 1: Write failing binary-level CLI tests**

Use `env!("CARGO_BIN_EXE_omp-gym")` and temp projects to test `configure`, `harvest`/`dry-run`, `tasks list/show/approve/reject/reopen/validate`, `status`, `proposal show/diff`, and `adopt --proposal`. Assert exit codes and stable user-facing phrases, not source layout.

- [ ] **Step 2: Run and observe failure**

Run: `cargo test -p omp-gym --test cli_flow -- --nocapture`

Expected: unknown-command and missing-config failures.

- [ ] **Step 3: Implement Clap command tree**

Global options become optional one-run overrides so persisted values are not overwritten by defaults. Commands:

```text
configure --target-skill PATH [--omp-bin PATH] [--replay-model M] [--optimizer-model M] [--judge-model M] [--no-judge]
harvest (visible alias: dry-run)
tasks list [--status S]
tasks show ID
tasks approve ID [--exact V] [--contains V] [--contains-ci V] [--not-contains V] [--not-contains-ci V] [--regex P] [--rubric R] [--note N]
tasks reject ID [--note N]
tasks reopen ID
tasks validate
run [--no-judge]
status
proposal show [ID]
proposal diff [ID]
adopt [--proposal ID]
schedule --hour H --minute M | --off
doctor
```

`configure` saves config atomically. Every mutating task command loads/saves through `task_store`. Remove `backend`, `stage`, and every mock-only phrase. Reports print run ID, gate status, proposal ID, target, and artifact directory.

- [ ] **Step 4: Run CLI and workspace tests**

Run: `cargo test -p omp-gym --test cli_flow -- --nocapture && cargo test --workspace --all-features`

Expected: all CLI contracts and workspace tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/omp-gym/Cargo.toml crates/omp-gym/src/main.rs crates/omp-gym/tests/cli_flow.rs
git commit -m "feat: add gym review and proposal CLI"
```

### Task 9: Harden scheduler and OMP extension

**Files:**
- Modify: `crates/omp-gym/src/main.rs`
- Modify: `extensions/omp/gym.ts`
- Test: CLI unit tests and extension smoke commands

- [ ] **Step 1: Write failing scheduler/tokenizer tests**

In Rust, test `schedule_label(project)` yields `com.mtent.omp-gym.<16-hex>` and differs by canonical project. In TypeScript, export a pure `parseArgs` and test quoted values, escaped spaces, empty quotes, and unmatched-quote errors using `bun test`.

- [ ] **Step 2: Run and observe failure**

Run: `cargo test -p omp-gym schedule_label -- --nocapture`

Run: `bun test extensions/omp/gym.test.ts`

Expected: missing label/tokenizer tests fail.

- [ ] **Step 3: Implement project-specific scheduling**

Derive label from SHA-256 of canonical project. The plist runs only `omp-gym --project <canonical> run`; persisted config supplies target/models/bounds. Store the exact label/plist in state. `--off` removes only that project's job. Validate hour `0..=23` and minute `0..=59` before writing.

- [ ] **Step 4: Implement extension command forwarding**

Resolve binaries in this order: `OMP_GYM_BIN`, `~/.local/bin/omp-gym`, `~/.cargo/bin/omp-gym`, development builds, then `PATH`. Parse quoted arguments without `eval`. Forward `configure`, `harvest`/`dry-run`, `tasks`, `run`, `status`, `proposal`, `adopt`, `doctor`, and `schedule`; retain `/gym overnight` as a schedule alias. Add a 30-minute `spawnSync` timeout and include signal/timeout errors. Replace all prototype/mock help text.

- [ ] **Step 5: Run scheduler, extension, and package tests**

Run: `cargo test -p omp-gym schedule -- --nocapture`

Run: `bun test extensions/omp/gym.test.ts && ./scripts/test-package-release.sh`

Expected: project labels, tokenizer, extension mapping, and bundled extension package pass.

- [ ] **Step 6: Commit**

```bash
git add crates/omp-gym/src/main.rs extensions/omp/gym.ts extensions/omp/gym.test.ts
git commit -m "feat: harden gym scheduling and extension"
```

### Task 10: Fake-OMP end-to-end integration coverage

**Files:**
- Create: `crates/omp-gym-core/tests/fake_omp_pipeline.rs`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Write deterministic fake OMP executable**

The fixture inspects the final prompt and append-skill file. For replay, output a valid assistant NDJSON event; include `GYM_OK` only when the skill instructs it. For optimizer prompts, return a valid complete candidate adding the general `GYM_OK` prefix rule. For judge prompts, return a tie JSON object. Add modes for nonzero exit, malformed JSON, missing terminal event, and sleep past timeout.

- [ ] **Step 2: Add complete integration cases**

Exercise five approved tasks (three train/two validation), baseline failure, candidate generation without validation leakage, candidate held-out improvement, proposal staging, adoption, backup, idempotence, and restore. Add separate error-path cases for timeout/nonzero/malformed/missing response.

- [ ] **Step 3: Run and fix only product defects exposed by the suite**

Run: `cargo test -p omp-gym-core --test fake_omp_pipeline -- --nocapture`

Expected: all end-to-end and failure-path cases pass with the real subprocess boundary.

- [ ] **Step 4: Keep CI authoritative**

Ensure CI runs `cargo test --locked --workspace --all-features` and `bun test extensions/omp/gym.test.ts` after installing Bun only if GitHub's runner does not already provide it. Do not duplicate packaging matrix work already covered by `release-builds`.

- [ ] **Step 5: Commit**

```bash
git add crates/omp-gym-core/tests/fake_omp_pipeline.rs .github/workflows/ci.yml
git commit -m "test: cover complete optimizer lifecycle"
```

### Task 11: Explain prompt gyms, SkillOpt, and OMP Gym

**Files:**
- Modify: `README.md`
- Modify: `docs/DESIGN.md`
- Modify: `docs/UPSTREAM_PR.md`
- Modify: `crates/omp-gym-core/src/lib.rs`
- Test: command/help smoke checks

- [ ] **Step 1: Replace prototype documentation with observed behavior**

Add these explicit README sections:

1. **What is a prompt or skill gym?** A local evaluation-and-improvement loop that converts recurring tasks into a reviewed suite, reruns an agent with a current instruction artifact, measures outcomes, proposes a bounded instruction change, and accepts it only behind a validation gate. It changes instructions, not model weights.
2. **How Microsoft SkillOpt works.** Explain rollout, reflection, candidate skill edits, held-out validation, and staged adoption; link the MIT project and paper/docs without claiming code reuse.
3. **How OMP Gym differs.** Rust/OMP-native execution; mines the owner's real OMP sessions; explicit human review and deterministic checks; text-only isolation; supplemental judge; strict zero-regression gate; manual atomic adoption; stock OMP extension; no Python runtime, no model training, and no vendored SkillOpt implementation.

Then document `configure -> harvest -> tasks approve -> run -> proposal diff -> adopt`, artifact sensitivity, scheduling, bounds, failure behavior, and release installation.

- [ ] **Step 2: Update design/upstream documents and crate docs**

Remove every `v0.1 mock`, `future`, `not implemented`, and false current-status statement. Preserve an honest limitations section: text-only replay, best-effort redaction, deterministic checks require human authoring, macOS-only built-in scheduler, and no automatic adoption.

- [ ] **Step 3: Verify commands against docs**

Run: `cargo run -p omp-gym -- --help`

Run: `cargo run -p omp-gym -- tasks --help`

Run: `cargo run -p omp-gym -- proposal --help`

Expected: documented commands and options appear; no mock/prototype claims remain in user-facing help.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/DESIGN.md docs/UPSTREAM_PR.md crates/omp-gym-core/src/lib.rs
git commit -m "docs: explain the OMP prompt gym"
```

### Task 12: Real OMP smoke, audit, and completed release

**Files:**
- Modify: `Cargo.toml` and `Cargo.lock` for version `0.2.0`
- Modify: any file with a defect proven by verification

- [ ] **Step 1: Run all static and automated checks**

Run:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
cargo build --locked --workspace --release
bun test extensions/omp/gym.test.ts
./scripts/test-package-release.sh
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12
```

Expected: every command exits zero.

- [ ] **Step 2: Run a real OMP end-to-end scenario**

Create a temporary project and valid deficient `SKILL.md` that says only “Answer concisely.” Create five approved tasks whose hidden deterministic check requires `GYM_OK`, with three train and two validation under the deterministic splitter. Configure a low-cost authenticated OMP model selector, run `omp-gym run`, and observe baseline failures, bounded candidate creation, held-out improvement, accepted proposal, and evidence. Run `proposal diff`, adopt, verify candidate hash/backup, run adopt again for idempotence, then restore the original from backup. If the optimizer produces no passing candidate, record the evidence, refine only the optimizer contract proven deficient, and rerun from a fresh temporary project.

- [ ] **Step 3: Audit every approved-design invariant**

Check the implementation against every heading in `docs/plans/2026-07-23-complete-optimizer-design.md`: review preservation, validation withholding, OMP-only execution, isolation flags, deterministic-only gate, zero regressions, immutable run evidence, passing-only proposals, compare-and-swap adoption, project-specific schedule, extension parity, and release packaging.

- [ ] **Step 4: Bump and verify version**

Set workspace version to `0.2.0`, update the lockfile, rebuild, and assert `target/release/omp-gym --version` prints `omp-gym 0.2.0`.

- [ ] **Step 5: Commit and push**

```bash
git add -A
git commit -m "feat: complete OMP skill optimizer"
git push origin main
```

- [ ] **Step 6: Verify remote CI before tagging**

Wait for both `ci` and the four-platform `release-builds` matrix on the pushed commit. Require formatting, Clippy, tests, packaging contract, Apple arm64/x86_64, and Linux arm64/x86_64 success.

- [ ] **Step 7: Publish and verify v0.2.0**

```bash
git tag -a v0.2.0 -m "omp-gym 0.2.0"
git push origin v0.2.0
```

Require the tag workflow to publish four archives, `SHA256SUMS`, and attestations. Download all assets, verify every checksum, run the downloaded native binary's `--version` and `doctor`, and confirm `extensions/omp/gym.ts` is bundled.

---

## Self-review record

- **Spec coverage:** Every approved design section maps to Tasks 1–12. Release CI was repaired and `v0.1.0` published before this plan; Task 12 proves the completed release.
- **Placeholder scan:** The plan contains no deferred implementation markers. Every test/implementation step names exact files, APIs, invariants, commands, and expected outcomes.
- **Type consistency:** `MinedTask`, `CheckSpec`, `Trajectory`, `TaskScore`, `GateDecision`, `RunRecord`, and `StagedProposal` are defined once in Task 1 and consumed under the same names throughout. `ModelRunner::run`, `run_night_with_runner`, and `adopt(cfg, proposal_id)` remain consistent across unit, integration, CLI, and smoke tasks.
