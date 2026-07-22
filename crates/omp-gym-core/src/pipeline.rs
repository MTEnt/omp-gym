use crate::config::GymConfig;
use crate::harvest::harvest_sessions;
use crate::mine::mine_tasks;
use crate::paths::{atomic_write_json, ensure_private_dir};
use crate::state::{load_latest_proposal, load_state, save_state};
use crate::types::{TasksFile, SCHEMA_VERSION};
use anyhow::{bail, Result};
use chrono::Utc;
use std::path::PathBuf;

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

/// Harvest + mine + write tasks.json. No skill mutation.
pub fn dry_run(cfg: &GymConfig) -> Result<GymReport> {
    ensure_private_dir(&cfg.gym_dir())?;
    let sessions = harvest_sessions(
        &cfg.sessions_root,
        &cfg.project,
        cfg.lookback_hours,
        cfg.max_sessions,
    )?;
    let tasks = mine_tasks(&sessions, cfg.max_tasks);
    let task_count = tasks.len();

    let tasks_file = TasksFile {
        schema_version: SCHEMA_VERSION,
        generated_at: Utc::now(),
        project: cfg.project.clone(),
        tasks,
    };
    atomic_write_json(&cfg.tasks_path(), &tasks_file)?;

    let mut state = load_state(&cfg.state_path())?;
    state.last_harvest_at = Some(Utc::now());
    state.last_session_ids = sessions.iter().map(|s| s.id.clone()).collect();
    save_state(&cfg.state_path(), &state)?;

    let mut notes = vec![
        format!(
            "Harvested {} session(s) from {}",
            sessions.len(),
            cfg.sessions_root.display()
        ),
        format!(
            "Mined {} task(s) → {}",
            task_count,
            cfg.tasks_path().display()
        ),
        "No skill files were modified.".into(),
    ];
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
        sessions: sessions.len(),
        tasks: task_count,
        backend: cfg.backend.clone(),
        staged: false,
        proposal_id: None,
        target_skill: cfg.target_skill.clone(),
        notes,
        gym_dir: cfg.gym_dir(),
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
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::tempdir;

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
                assert_eq!(mode, expected_mode, "unexpected mode for {}", path.display());
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
}
