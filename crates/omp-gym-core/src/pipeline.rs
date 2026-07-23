use crate::config::GymConfig;
use crate::evaluation::split_tasks;
use crate::harvest::harvest_sessions;
use crate::mine::mine_tasks;
use crate::paths::ensure_private_dir;
use crate::state::{load_latest_proposal, load_state, save_state};
use crate::task_store::{
    load_tasks, merge_tasks, save_tasks, validate_reviewed_tasks,
};
use crate::types::{MinedTask, ReviewStatus, TaskSplit, TasksFile};
use anyhow::{bail, Context, Result};
use chrono::Utc;
use fs2::FileExt;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct GymReport {
    pub sessions: usize,
    pub tasks: usize,
    pub backend: String,
    pub staged: bool,
    pub proposal_id: Option<String>,
    pub target_skill: Option<PathBuf>,
    pub notes: Vec<String>,
    pub gym_dir: PathBuf,
}

#[derive(Debug)]
struct RunLease {
    _file: File,
}

impl RunLease {
    fn acquire(cfg: &GymConfig) -> Result<Self> {
        ensure_private_dir(&cfg.gym_dir())?;
        let path = cfg.run_lock_path();
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        options.mode(0o600);
        let file = options
            .open(&path)
            .with_context(|| format!("open optimizer run lock {}", path.display()))?;
        #[cfg(unix)]
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("set optimizer run lock permissions {}", path.display()))?;
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Self { _file: file }),
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                bail!(
                    "optimizer run already in progress for {}",
                    cfg.project.display()
                )
            }
            Err(error) => Err(error)
                .with_context(|| format!("acquire optimizer run lock {}", path.display())),
        }
    }
}

/// Validated, stable inputs for an optimizer run.
///
/// The exclusive project run lease remains held until this value is dropped.
/// This preflight context intentionally contains no model output or persisted
/// run/proposal artifacts.
#[derive(Debug)]
pub struct PreparedRun {
    _lease: RunLease,
    approved_tasks: Vec<MinedTask>,
    split: TaskSplit,
    target_skill: PathBuf,
    base_skill: String,
    task_store_hash: String,
    base_skill_hash: String,
    session_count: usize,
    task_count: usize,
    notes: Vec<String>,
}

impl PreparedRun {
    pub fn approved_tasks(&self) -> &[MinedTask] {
        &self.approved_tasks
    }

    pub fn approved_task_count(&self) -> usize {
        self.approved_tasks.len()
    }

    pub fn split(&self) -> &TaskSplit {
        &self.split
    }

    pub fn target_skill(&self) -> &Path {
        &self.target_skill
    }

    pub fn base_skill(&self) -> &str {
        &self.base_skill
    }

    pub fn task_store_hash(&self) -> &str {
        &self.task_store_hash
    }

    pub fn base_skill_hash(&self) -> &str {
        &self.base_skill_hash
    }

    pub fn session_count(&self) -> usize {
        self.session_count
    }

    pub fn task_count(&self) -> usize {
        self.task_count
    }

    pub fn notes(&self) -> &[String] {
        &self.notes
    }
}

#[derive(Debug)]
struct RefreshResult {
    session_count: usize,
    task_count: usize,
    notes: Vec<String>,
    tasks_file: TasksFile,
    task_store_bytes: Vec<u8>,
}

fn refresh_tasks_with_state_saver<F>(cfg: &GymConfig, state_saver: F) -> Result<RefreshResult>
where
    F: FnOnce(&Path, &crate::state::GymState) -> Result<()>,
{
    ensure_private_dir(&cfg.gym_dir())?;
    let sessions = harvest_sessions(
        &cfg.sessions_root,
        &cfg.project,
        cfg.lookback_hours,
        cfg.max_sessions,
    )?;
    let mined = mine_tasks(&sessions, cfg.max_tasks);
    let now = Utc::now();
    let mut tasks_file = load_tasks(&cfg.tasks_path(), &cfg.project)?;
    let mut state = load_state(&cfg.state_path())?;
    let existing_ids: BTreeSet<String> = tasks_file
        .tasks
        .iter()
        .map(|task| task.id.clone())
        .collect();
    let approved_ids: BTreeSet<String> = tasks_file
        .tasks
        .iter()
        .filter(|task| task.status == ReviewStatus::Approved)
        .map(|task| task.id.clone())
        .collect();
    tasks_file.tasks = merge_tasks(tasks_file.tasks, mined, now)?;
    tasks_file.generated_at = now;
    let new_count = tasks_file
        .tasks
        .iter()
        .filter(|task| !existing_ids.contains(&task.id))
        .count();
    let preserved_approved_count = tasks_file
        .tasks
        .iter()
        .filter(|task| task.status == ReviewStatus::Approved && approved_ids.contains(&task.id))
        .count();
    let task_count = tasks_file.tasks.len();
    let task_store_bytes = serde_json::to_vec_pretty(&tasks_file)
        .with_context(|| format!("serialize refreshed task store {}", cfg.tasks_path().display()))?;
    save_tasks(&cfg.tasks_path(), &tasks_file)?;

    state.last_harvest_at = Some(now);
    state.last_session_ids = sessions.iter().map(|session| session.id.clone()).collect();
    let state_save_warning = state_saver(&cfg.state_path(), &state)
        .err()
        .map(|error| format!("Warning: tasks saved but state update failed: {error:#}"));

    let mut notes = vec![
        format!(
            "Harvested {} session(s) from {}",
            sessions.len(),
            cfg.sessions_root.display()
        ),
        format!(
            "Mined and merged {} task(s) → {}",
            task_count,
            cfg.tasks_path().display()
        ),
        format!(
            "Task store: {new_count} new, {preserved_approved_count} preserved approved, {task_count} total"
        ),
        "No skill files were modified.".into(),
    ];
    if let Some(warning) = state_save_warning {
        notes.push(warning);
    }
    Ok(RefreshResult {
        session_count: sessions.len(),
        task_count,
        notes,
        tasks_file,
        task_store_bytes,
    })
}

/// Harvest + mine + merge into tasks.json without changing review decisions.
pub fn dry_run(cfg: &GymConfig) -> Result<GymReport> {
    dry_run_with_state_saver(cfg, save_state)
}

fn dry_run_with_state_saver<F>(cfg: &GymConfig, state_saver: F) -> Result<GymReport>
where
    F: FnOnce(&Path, &crate::state::GymState) -> Result<()>,
{
    let refresh = refresh_tasks_with_state_saver(cfg, state_saver)?;
    let mut notes = refresh.notes;
    if cfg.backend != "mock" {
        notes.push(format!(
            "backend={} requested; v0.1 dry-run stays offline (mock harvest/mine only).",
            cfg.backend
        ));
    }
    if let Some(skill) = &cfg.target_skill {
        notes.push(format!(
            "reserved target skill (not used by v0.1): {}",
            skill.display()
        ));
    }

    Ok(GymReport {
        sessions: refresh.session_count,
        tasks: refresh.task_count,
        backend: cfg.backend.clone(),
        staged: false,
        proposal_id: None,
        target_skill: cfg.target_skill.clone(),
        notes,
        gym_dir: cfg.gym_dir(),
    })
}

/// Acquires the exclusive run lease and prepares every model-free run input.
///
/// Configuration and the target skill are validated before harvesting. The
/// refreshed task store is then loaded and validated, approved tasks are
/// deterministically split, and hashes are computed from the exact raw bytes
/// that future run persistence must reference. No run or proposal artifacts
/// are created and no model boundary is crossed.
pub fn prepare_run(cfg: &GymConfig) -> Result<PreparedRun> {
    let lease = RunLease::acquire(cfg)?;
    let target_skill = cfg.validate_for_run()?;
    let base_bytes = std::fs::read(&target_skill)
        .with_context(|| format!("read complete target skill {}", target_skill.display()))?;
    let base_skill_hash = format!("{:x}", Sha256::digest(&base_bytes));
    let base_skill = String::from_utf8(base_bytes)
        .with_context(|| format!("target skill is not UTF-8: {}", target_skill.display()))?;

    let RefreshResult {
        session_count,
        task_count,
        notes,
        tasks_file,
        task_store_bytes,
    } = refresh_tasks_with_state_saver(cfg, save_state)?;
    let task_store_hash = format!("{:x}", Sha256::digest(&task_store_bytes));

    let mut approved_tasks = validate_reviewed_tasks(&tasks_file)?
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    approved_tasks.sort_by(|left, right| left.id.cmp(&right.id));
    let approved_refs = approved_tasks.iter().collect::<Vec<_>>();
    let split = split_tasks(
        &approved_refs,
        cfg.validation_ratio,
        cfg.min_validation_tasks,
    )?;

    Ok(PreparedRun {
        _lease: lease,
        approved_tasks,
        split,
        target_skill,
        base_skill,
        task_store_hash,
        base_skill_hash,
        session_count,
        task_count,
        notes,
    })
}

/// Harvest/mine only; proposal staging awaits real replay, evaluation, and gating.
/// No model runs and no skill file is evaluated or changed.
pub fn run_night(cfg: &GymConfig, stage: bool) -> Result<GymReport> {
    if cfg.backend != "mock" {
        bail!(
            "backend '{}' is not implemented; use --backend mock for the harvest-only prototype",
            cfg.backend
        );
    }
    let mut report = dry_run(cfg)?;
    if !stage {
        report
            .notes
            .push("run with stage=false completed harvest/mine only.".into());
        return Ok(report);
    }
    if report.tasks == 0 {
        bail!(
            "no tasks were mined for project {}; refusing to stage an empty proposal",
            cfg.project.display()
        );
    }

    bail!(
        "mock proposal staging is disabled until replay, evaluation, and gate evidence are implemented"
    )
}

pub fn status(cfg: &GymConfig) -> Result<String> {
    let state = load_state(&cfg.state_path())?;
    let proposal = load_latest_proposal(&cfg.proposal_dir())?;
    let mut lines = vec![
        format!("project:     {}", cfg.project.display()),
        format!("gym dir:     {}", cfg.gym_dir().display()),
        format!("sessions:    {}", cfg.sessions_root.display()),
        format!("backend:     {}", cfg.backend),
        format!(
            "target:      {}",
            cfg.target_skill
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(unset)".into())
        ),
        format!("mock runs:   {}", state.nights_completed),
        format!(
            "last harvest:{}",
            state
                .last_harvest_at
                .map(|t| t.to_rfc3339())
                .unwrap_or_else(|| "never".into())
        ),
        format!(
            "last run:    {}",
            state
                .last_run_at
                .map(|t| t.to_rfc3339())
                .unwrap_or_else(|| "never".into())
        ),
    ];
    if let Some(sched) = &state.schedule {
        lines.push(format!(
            "schedule:    {} {:02}:{:02} ({})",
            if sched.enabled { "on" } else { "off" },
            sched.hour_local,
            sched.minute_local,
            sched.label
        ));
    } else {
        lines.push("schedule:    not configured".into());
    }
    match proposal {
        Some(p) => {
            lines.push(format!("proposal:    {} (status={:?})", p.id, p.status));
            lines.push(format!("  summary:   {}", p.summary));
            lines.push(format!("  accepted:  {}", p.gate.accepted));
        }
        None => lines.push("proposal:    none staged".into()),
    }
    Ok(lines.join("\n"))
}

/// Adopt is intentionally strict in v0.1: refuse mock proposals.
pub fn adopt(cfg: &GymConfig) -> Result<String> {
    let Some(proposal) = load_latest_proposal(&cfg.proposal_dir())? else {
        bail!("no staged proposal to adopt");
    };
    if proposal.candidate_path.as_os_str().is_empty() {
        bail!(
            "latest proposal {} is mock-only; refusing to modify SKILL.md. \
             Wait for a real backend night or implement replay/validate first.",
            proposal.id
        );
    }
    let Some(target) = cfg.target_skill.as_ref() else {
        bail!("--target-skill is required to adopt");
    };
    if !target.exists() {
        bail!("target skill does not exist: {}", target.display());
    }
    // Future: apply staged skill patch with backup.
    bail!(
        "non-mock adopt not implemented yet (proposal {}, target {})",
        proposal.id,
        target.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CheckSpec, MinedTask, ReviewStatus, TaskSplit, TasksFile, SCHEMA_VERSION,
    };
    use chrono::Utc;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::{tempdir, TempDir};

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("omp-gym-{name}-{nanos}"))
    }

    fn write_session(path: &std::path::Path, id: &str, cwd: &std::path::Path, prompt: &str) {
        let session = serde_json::json!({
            "type": "session",
            "id": id,
            "cwd": cwd,
            "timestamp": "2026-07-23T12:00:00Z"
        });
        let message = serde_json::json!({
            "type": "message",
            "message": {
                "role": "user",
                "content": [{ "type": "text", "text": prompt }]
            }
        });
        std::fs::write(path, format!("{session}\n{message}\n")).expect("write session fixture");
    }

    #[test]
    fn dry_run_uses_only_sessions_from_the_selected_project() {
        let root = unique_test_dir("project-filter");
        let project = root.join("project");
        let other_project = root.join("other-project");
        let sessions = root.join("sessions");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::create_dir_all(&other_project).expect("create other project");
        std::fs::create_dir_all(&sessions).expect("create sessions root");
        write_session(
            &sessions.join("selected.jsonl"),
            "selected",
            &project,
            "Fix authentication in the selected project",
        );
        write_session(
            &sessions.join("other.jsonl"),
            "other",
            &other_project,
            "Replace billing in another project",
        );

        let mut config = GymConfig::for_project(&project).expect("build config");
        config.sessions_root = sessions;
        config.lookback_hours = 0;
        config.max_sessions = 10;
        config.max_tasks = 10;

        let report = dry_run(&config).expect("dry run");
        let tasks: TasksFile = serde_json::from_str(
            &std::fs::read_to_string(config.tasks_path()).expect("read generated tasks"),
        )
        .expect("parse generated tasks");

        assert_eq!(report.sessions, 1);
        assert_eq!(tasks.tasks.len(), 1);
        assert_eq!(tasks.tasks[0].source_session_ids, ["selected"]);
        assert!(tasks.tasks[0].prompt.contains("selected project"));

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn dry_run_preserves_approved_tasks_and_checks_across_reharvest() {
        let root = tempdir().expect("create temporary directory");
        let project = root.path().join("project");
        let sessions = root.path().join("sessions");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::create_dir_all(&sessions).expect("create sessions root");
        write_session(
            &sessions.join("selected.jsonl"),
            "selected",
            &project,
            "Fix recurring login authentication failures in the auth module",
        );

        let mut config = GymConfig::for_project(&project).expect("build config");
        config.sessions_root = sessions;
        config.lookback_hours = 0;
        dry_run(&config).expect("initial dry run");

        let mut tasks =
            crate::task_store::load_tasks(&config.tasks_path(), &project).expect("load tasks");
        assert!(tasks.tasks[0].first_seen_at <= tasks.tasks[0].last_seen_at);
        let task_id = tasks.tasks[0].id.clone();
        crate::task_store::approve_task(
            &mut tasks,
            &task_id,
            vec![CheckSpec::Contains {
                value: "resolved".into(),
                case_sensitive: false,
            }],
            Some("Explain the fix".into()),
            Some("owner reviewed".into()),
        )
        .expect("approve task");
        crate::task_store::save_tasks(&config.tasks_path(), &tasks).expect("save reviewed tasks");

        let report = dry_run(&config).expect("repeat dry run");
        let tasks =
            crate::task_store::load_tasks(&config.tasks_path(), &project).expect("reload tasks");

        assert_eq!(tasks.tasks.len(), 1);
        assert_eq!(tasks.tasks[0].id, task_id);
        assert_eq!(tasks.tasks[0].status, ReviewStatus::Approved);
        assert_eq!(
            tasks.tasks[0].checks,
            [CheckSpec::Contains {
                value: "resolved".into(),
                case_sensitive: false,
            }]
        );
        assert_eq!(tasks.tasks[0].rubric.as_deref(), Some("Explain the fix"));
        assert_eq!(
            tasks.tasks[0].review_note.as_deref(),
            Some("owner reviewed")
        );
        assert!(report
            .notes
            .iter()
            .any(|note| note == "Task store: 0 new, 1 preserved approved, 1 total"));
    }

    #[test]
    fn dry_run_warns_but_succeeds_when_state_save_fails_after_tasks_save() {
        let root = tempdir().expect("create temporary directory");
        let project = root.path().join("project");
        let sessions = root.path().join("sessions");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::create_dir_all(&sessions).expect("create sessions root");
        write_session(
            &sessions.join("selected.jsonl"),
            "selected",
            &project,
            "Fix recurring login authentication failures in the auth module",
        );
        let mut config = GymConfig::for_project(&project).expect("build config");
        config.sessions_root = sessions;
        config.lookback_hours = 0;

        let report = dry_run_with_state_saver(&config, |_, _| {
            anyhow::bail!("simulated state write failure")
        })
        .expect("task store remains authoritative");

        assert!(config.tasks_path().exists());
        assert!(!config.state_path().exists());
        assert!(report.notes.iter().any(|note| {
            note.contains("Warning:")
                && note.contains("state")
                && note.contains("simulated state write failure")
        }));
    }

    #[test]
    fn corrupt_state_aborts_before_tasks_are_mutated() {
        let root = tempdir().expect("create temporary directory");
        let project = root.path().join("project");
        let sessions = root.path().join("sessions");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::create_dir_all(&sessions).expect("create sessions root");
        write_session(
            &sessions.join("selected.jsonl"),
            "selected",
            &project,
            "Fix recurring login authentication failures in the auth module",
        );
        let mut config = GymConfig::for_project(&project).expect("build config");
        config.sessions_root = sessions.clone();
        config.lookback_hours = 0;
        dry_run(&config).expect("initial dry run");
        let original_tasks = std::fs::read(config.tasks_path()).expect("read initial tasks");
        std::fs::write(config.state_path(), b"{corrupt").expect("corrupt state");
        write_session(
            &sessions.join("new.jsonl"),
            "new",
            &project,
            "Document the deployment rollback procedure carefully",
        );

        let error = dry_run(&config).expect_err("corrupt state must abort");

        assert!(error.to_string().contains("state"));
        assert_eq!(
            std::fs::read(config.tasks_path()).expect("read unchanged tasks"),
            original_tasks
        );
    }

    #[test]
    fn dry_run_gitignores_transcript_derived_artifacts() {
        let root = unique_test_dir("artifact-ignore");
        let project = root.join("project");
        let sessions = root.join("sessions");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::create_dir_all(&sessions).expect("create sessions root");
        write_session(
            &sessions.join("selected.jsonl"),
            "selected",
            &project,
            "Review private customer workflow",
        );

        let mut config = GymConfig::for_project(&project).expect("build config");
        config.sessions_root = sessions;
        config.lookback_hours = 0;
        dry_run(&config).expect("dry run");

        let ignore = std::fs::read_to_string(config.gym_dir().join(".gitignore"))
            .expect("gym artifacts should be ignored");
        assert_eq!(ignore, "*\n!.gitignore\n");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            for (path, expected_mode) in [
                (config.gym_dir(), 0o700),
                (config.gym_dir().join(".gitignore"), 0o600),
                (config.tasks_path(), 0o600),
                (config.state_path(), 0o600),
            ] {
                let mode = std::fs::metadata(&path)
                    .expect("private artifact metadata")
                    .permissions()
                    .mode()
                    & 0o777;
                assert_eq!(
                    mode,
                    expected_mode,
                    "unexpected mode for {}",
                    path.display()
                );
            }
        }

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn run_rejects_unimplemented_backends() {
        let root = unique_test_dir("backend-gate");
        let project = root.join("project");
        std::fs::create_dir_all(&project).expect("create project");

        let mut config = GymConfig::for_project(&project).expect("build config");
        config.backend = "omp".to_owned();
        config.sessions_root = root.join("missing-sessions");

        let error = run_night(&config, true).expect_err("omp backend is not implemented");
        assert!(error.to_string().contains("not implemented"));

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn mock_run_never_stages_an_accepted_proposal() {
        let root = tempdir().expect("create temporary directory");
        let project = root.path().join("project");
        let sessions = root.path().join("sessions");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::create_dir_all(&sessions).expect("create sessions root");
        write_session(
            &sessions.join("selected.jsonl"),
            "selected",
            &project,
            "Improve authentication error handling",
        );

        let mut config = GymConfig::for_project(&project).expect("build config");
        config.sessions_root = sessions;
        config.lookback_hours = 0;

        let error =
            run_night(&config, true).expect_err("mock run must not create an accepted proposal");

        assert!(error
            .to_string()
            .contains("mock proposal staging is disabled"));
        assert!(!config.proposal_dir().join("LATEST").exists());
    }

    #[test]
    fn run_does_not_stage_a_proposal_without_tasks() {
        let root = unique_test_dir("empty-run");
        let project = root.join("project");
        std::fs::create_dir_all(&project).expect("create project");

        let mut config = GymConfig::for_project(&project).expect("build config");
        config.sessions_root = root.join("missing-sessions");

        let error = run_night(&config, true).expect_err("empty run should not stage");
        assert!(error.to_string().contains("no tasks"));
        assert!(!config.proposal_dir().join("LATEST").exists());

        std::fs::remove_dir_all(root).ok();
    }

    struct PreflightFixture {
        _root: TempDir,
        config: GymConfig,
    }

    fn preflight_task(id: &str, status: ReviewStatus, checks: Vec<CheckSpec>) -> MinedTask {
        let now = Utc::now();
        MinedTask {
            id: id.to_owned(),
            title: format!("Task {id}"),
            prompt: format!("Complete task {id}"),
            source_session_ids: vec![format!("session-{id}")],
            source_occurrences: BTreeMap::from([(format!("session-{id}"), 1)]),
            frequency: 1,
            status,
            checks,
            rubric: None,
            review_note: None,
            reviewed_at: Some(now),
            first_seen_at: now,
            last_seen_at: now,
        }
    }

    fn valid_check(id: &str) -> CheckSpec {
        CheckSpec::Contains {
            value: format!("done-{id}"),
            case_sensitive: true,
        }
    }

    fn preflight_fixture(statuses: &[ReviewStatus]) -> PreflightFixture {
        let root = tempdir().expect("create temporary directory");
        let project = root.path().join("project");
        let sessions = root.path().join("sessions");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::create_dir_all(&sessions).expect("create sessions");
        let target = project.join("skills/demo/SKILL.md");
        std::fs::create_dir_all(target.parent().expect("target parent")).expect("create skills");
        std::fs::write(&target, b"---\nname: demo\n---\n\nComplete the task.\n")
            .expect("write base skill");

        let mut config = GymConfig::for_project(&project).expect("build config");
        config.sessions_root = sessions;
        config.lookback_hours = 0;
        config.target_skill = Some(PathBuf::from("skills/demo/SKILL.md"));
        let tasks = statuses
            .iter()
            .enumerate()
            .rev()
            .map(|(index, status)| {
                let id = format!("task-{index}");
                let checks = if *status == ReviewStatus::Approved {
                    vec![valid_check(&id)]
                } else {
                    Vec::new()
                };
                preflight_task(&id, status.clone(), checks)
            })
            .collect();
        crate::task_store::save_tasks(
            &config.tasks_path(),
            &TasksFile {
                schema_version: SCHEMA_VERSION,
                generated_at: Utc::now(),
                project: project.canonicalize().expect("canonical project"),
                tasks,
            },
        )
        .expect("save tasks");

        PreflightFixture {
            _root: root,
            config,
        }
    }

    fn five_approved_fixture() -> PreflightFixture {
        preflight_fixture(&[
            ReviewStatus::Approved,
            ReviewStatus::Approved,
            ReviewStatus::Approved,
            ReviewStatus::Approved,
            ReviewStatus::Approved,
        ])
    }

    #[test]
    fn prepared_run_holds_exclusive_lock_and_releases_it_on_drop() {
        let fixture = five_approved_fixture();
        let first = prepare_run(&fixture.config).expect("first preflight");

        let conflict = prepare_run(&fixture.config)
            .expect_err("held lease must prevent a concurrent preflight")
            .to_string();
        assert!(conflict.contains("already in progress"), "{conflict}");

        drop(first);
        prepare_run(&fixture.config).expect("lock should release when prepared run drops");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(fixture.config.run_lock_path())
                    .expect("lock metadata")
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn invalid_config_and_target_fail_before_harvest_mutates_tasks() {
        for defect in ["config", "target"] {
            let mut fixture = five_approved_fixture();
            let before = std::fs::read(fixture.config.tasks_path()).expect("read tasks before");
            write_session(
                &fixture.config.sessions_root.join("new.jsonl"),
                "new-session",
                &fixture.config.project,
                "Document a newly harvested deployment rollback procedure",
            );
            if defect == "config" {
                fixture.config.validation_ratio = 1.0;
            } else {
                fixture.config.target_skill = Some(PathBuf::from("skills/missing/SKILL.md"));
            }

            prepare_run(&fixture.config).expect_err("invalid preflight must fail");

            assert_eq!(
                std::fs::read(fixture.config.tasks_path()).expect("read unchanged tasks"),
                before,
                "{defect} defect mutated task store"
            );
            assert!(!fixture.config.state_path().exists());
        }
    }

    #[test]
    fn refresh_is_preserved_before_approved_selection_rejects_small_suite() {
        let fixture = preflight_fixture(&[
            ReviewStatus::Approved,
            ReviewStatus::Approved,
            ReviewStatus::Approved,
            ReviewStatus::Approved,
        ]);
        write_session(
            &fixture.config.sessions_root.join("new.jsonl"),
            "new-session",
            &fixture.config.project,
            "Document a newly harvested deployment rollback procedure",
        );

        let error = prepare_run(&fixture.config)
            .expect_err("new pending task must not satisfy approved minimum")
            .to_string();
        let tasks = load_tasks(
            &fixture.config.tasks_path(),
            &fixture.config.project,
        )
        .expect("load refreshed tasks");

        assert!(error.contains("at least 5 approved tasks"), "{error}");
        assert_eq!(tasks.tasks.len(), 5);
        assert_eq!(
            tasks
                .tasks
                .iter()
                .filter(|task| task.status == ReviewStatus::Approved)
                .count(),
            4
        );
        assert!(tasks
            .tasks
            .iter()
            .any(|task| task.source_session_ids == ["new-session"]));
        assert!(fixture.config.state_path().exists());
    }

    #[test]
    fn pending_and_rejected_tasks_are_excluded_from_stable_prepared_inputs() {
        let fixture = preflight_fixture(&[
            ReviewStatus::Rejected,
            ReviewStatus::Approved,
            ReviewStatus::Pending,
            ReviewStatus::Approved,
            ReviewStatus::Approved,
            ReviewStatus::Approved,
            ReviewStatus::Approved,
        ]);

        let prepared = prepare_run(&fixture.config).expect("prepare approved task suite");
        let ids = prepared
            .approved_tasks()
            .iter()
            .map(|task| task.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(ids, ["task-1", "task-3", "task-4", "task-5", "task-6"]);
        assert_eq!(prepared.task_count(), 7);
        assert_eq!(prepared.approved_task_count(), 5);
        assert_eq!(prepared.split().train_ids.len(), 3);
        assert_eq!(prepared.split().validation_ids.len(), 2);
        assert_eq!(
            prepared.split().train_ids.len() + prepared.split().validation_ids.len(),
            5
        );
    }

    #[test]
    fn approved_tasks_require_nonempty_valid_checks() {
        for checks in [
            Vec::new(),
            vec![CheckSpec::Regex {
                pattern: "(".into(),
            }],
        ] {
            let fixture = five_approved_fixture();
            let mut tasks =
                load_tasks(&fixture.config.tasks_path(), &fixture.config.project).unwrap();
            tasks.tasks[0].checks = checks;
            crate::task_store::save_tasks(&fixture.config.tasks_path(), &tasks).unwrap();

            let error = prepare_run(&fixture.config)
                .expect_err("invalid approved checks must fail preflight")
                .to_string();
            assert!(error.contains("invalid checks"), "{error}");
        }
    }

    #[test]
    fn prepared_run_uses_raw_hashes_complete_base_and_canonical_target() {
        let fixture = five_approved_fixture();
        let target = fixture
            .config
            .project
            .join("skills/demo/SKILL.md")
            .canonicalize()
            .expect("canonical target");
        let expected_base = std::fs::read(&target).expect("read raw base");
        let prepared = prepare_run(&fixture.config).expect("prepare run");
        let expected_tasks =
            std::fs::read(fixture.config.tasks_path()).expect("read raw refreshed tasks");

        assert_eq!(prepared.target_skill(), target);
        assert_eq!(prepared.base_skill().as_bytes(), expected_base);
        assert_eq!(
            prepared.base_skill_hash(),
            format!("{:x}", Sha256::digest(&expected_base))
        );
        assert_eq!(
            prepared.task_store_hash(),
            format!("{:x}", Sha256::digest(&expected_tasks))
        );
        assert_eq!(prepared.session_count(), 0);
        assert!(!prepared.notes().is_empty());
    }

    #[test]
    fn corrupt_or_missing_task_store_fails_without_run_artifacts() {
        let corrupt = five_approved_fixture();
        std::fs::write(corrupt.config.tasks_path(), b"{corrupt").expect("corrupt tasks");
        let corrupt_error = prepare_run(&corrupt.config)
            .expect_err("corrupt tasks must fail")
            .to_string();
        assert!(corrupt_error.contains("parse tasks JSON"), "{corrupt_error}");
        assert!(!corrupt.config.runs_dir().exists());
        assert!(!corrupt.config.proposal_dir().join("LATEST").exists());

        let missing = five_approved_fixture();
        std::fs::remove_file(missing.config.tasks_path()).expect("remove tasks");
        let missing_error = prepare_run(&missing.config)
            .expect_err("missing reviewed task store must not start a run")
            .to_string();
        assert!(
            missing_error.contains("at least 5 approved tasks"),
            "{missing_error}"
        );
        assert!(!missing.config.runs_dir().exists());
        assert!(!missing.config.proposal_dir().join("LATEST").exists());
    }

    #[test]
    fn successful_preflight_creates_no_run_or_proposal_artifacts() {
        let fixture = five_approved_fixture();

        let prepared = prepare_run(&fixture.config).expect("prepare run");

        assert_eq!(
            prepared.split(),
            &TaskSplit {
                train_ids: prepared.split().train_ids.clone(),
                validation_ids: prepared.split().validation_ids.clone(),
            }
        );
        assert!(!fixture.config.runs_dir().exists());
        assert!(!fixture.config.proposal_dir().exists());
        assert!(!fixture.config.proposal_dir().join("LATEST").exists());
    }

    #[test]
    fn refresh_returns_the_exact_owned_task_store_snapshot_it_persisted() {
        let fixture = five_approved_fixture();

        let refresh =
            refresh_tasks_with_state_saver(&fixture.config, save_state).expect("refresh tasks");

        assert_eq!(
            refresh.task_store_bytes,
            std::fs::read(fixture.config.tasks_path()).expect("read persisted snapshot")
        );
        assert_eq!(refresh.tasks_file.tasks.len(), refresh.task_count);
    }
}
