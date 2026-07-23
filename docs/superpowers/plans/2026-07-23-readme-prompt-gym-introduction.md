# README Prompt Gym Introduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the repository’s opaque opening with an accurate, plain-language explanation of prompt gyms, their benefits, and their relationship to OMP.

**Architecture:** This is a documentation-only change. `README.md` gains a vision-first introduction and an explicit current-state boundary before its existing status table; the table and all operational instructions remain authoritative and unchanged.

**Tech Stack:** GitHub-flavored Markdown, existing Rust/GitHub Actions verification.

---

### Task 1: Rewrite the repository opening

**Files:**
- Modify: `README.md:3-5`
- Reference: `docs/superpowers/specs/2026-07-23-readme-prompt-gym-introduction-design.md`

- [ ] **Step 1: Replace the existing two-line description**

Keep `# omp-gym` and replace the text between it and `## Current status` with this content:

```markdown
**An evidence-driven prompt gym for [OMP](https://github.com/MTEnt/oh-my-pi).**

A **prompt gym** is a feedback loop for improving reusable AI instructions from evidence instead of guesswork. It turns real work into representative tasks, measures how the current instructions perform, proposes a bounded improvement, and tests that candidate on tasks it was not trained on. In OMP, those reusable instructions are typically a skill’s `SKILL.md` file.

## Why use a prompt gym?

- **Learn from your work.** Improvements come from the projects and tasks where you actually use OMP, not a generic benchmark.
- **Measure changes repeatably.** Explicit checks make “better” more than a subjective impression.
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

This design is inspired by [microsoft/SkillOpt](https://github.com/microsoft/SkillOpt) (MIT), but `omp-gym` is a clean-room Rust project built for OMP rather than a port of SkillOpt’s Python package.

> **Implementation boundary:** The workflow above describes the intended complete system. The current release is still a session-harvest and task-mining prototype: it includes foundational task review, replay, evaluation, and proposal-safety components, but the CLI does not yet orchestrate the full optimization and adoption loop. The status table below is authoritative.
```

- [ ] **Step 2: Check the Markdown diff**

Run:

```bash
git diff --check -- README.md
git diff -- README.md
```

Expected: no whitespace errors; only the opening before `## Current status` changes.

- [ ] **Step 3: Commit the README change**

```bash
git add README.md
git commit -m "docs: explain the OMP prompt gym"
```

Expected: one documentation commit containing only `README.md`.

### Task 2: Verify and publish the documentation

**Files:**
- Verify: `README.md`
- Verify: `.github/workflows/ci.yml`
- Verify: `.github/workflows/release.yml`

- [ ] **Step 1: Verify repository contracts locally**

Run:

```bash
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12
./scripts/test-package-release.sh
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
cargo build --locked --workspace --release
```

Expected: every command exits successfully; the test command reports zero failures.

- [ ] **Step 2: Push the verified HEAD to `main`**

Run:

```bash
git fetch origin main
git merge-base --is-ancestor origin/main HEAD
git push origin HEAD:main
```

Expected: ancestry check and push both succeed without force.

- [ ] **Step 3: Verify GitHub Actions**

Find the workflow runs for the pushed SHA and wait for both required workflows:

```bash
SHA=$(git rev-parse HEAD)
CI_RUN=$(gh run list --repo MTEnt/omp-gym --workflow ci.yml --commit "$SHA" --limit 1 --json databaseId --jq '.[0].databaseId')
RELEASE_RUN=$(gh run list --repo MTEnt/omp-gym --workflow release.yml --commit "$SHA" --limit 1 --json databaseId --jq '.[0].databaseId')
test -n "$CI_RUN" && test -n "$RELEASE_RUN"
gh run watch "$CI_RUN" --repo MTEnt/omp-gym --exit-status
gh run watch "$RELEASE_RUN" --repo MTEnt/omp-gym --exit-status
```

Expected: `ci` and `release-builds` both finish with conclusion `success`; publishing remains tag-gated.
