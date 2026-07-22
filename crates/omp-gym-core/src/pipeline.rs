use crate::config::GymConfig;
use crate::harvest::harvest_sessions;
use crate::mine::mine_tasks;
use crate::paths::ensure_dir;
use crate::state::{load_latest_proposal, load_state, save_proposal, save_state};
use crate::types::{StagedProposal, TasksFile};
use anyhow::{bail, Result};
use chrono::Utc;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

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
    ensure_dir(&cfg.gym_dir())?;
    let sessions = harvest_sessions(&cfg.sessions_root, cfg.lookback_hours, cfg.max_sessions)?;
    let tasks = mine_tasks(&sessions, cfg.max_tasks);

    let tasks_file = TasksFile {
        generated_at: Utc::now(),
        project: cfg.project.clone(),
        reviewed: false,
        tasks: tasks.clone(),
    };
    fs::write(cfg.tasks_path(), serde_json::to_string_pretty(&tasks_file)?)?;

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
        format!("Mined {} task(s) → {}", tasks.len(), cfg.tasks_path().display()),
        "dry-run does not stage skill changes.".into(),
    ];
    if cfg.backend != "mock" {
        notes.push(format!(
            "backend={} requested; v0.1 dry-run stays offline (mock harvest/mine only).",
            cfg.backend
        ));
    }
    if let Some(skill) = &cfg.target_skill {
        notes.push(format!("target skill: {}", skill.display()));
    } else {
        notes.push("no --target-skill set; adopt will require one later.".into());
    }

    Ok(GymReport {
        sessions: sessions.len(),
        tasks: tasks.len(),
        backend: cfg.backend.clone(),
        staged: false,
        proposal_id: None,
        target_skill: cfg.target_skill.clone(),
        notes,
        gym_dir: cfg.gym_dir(),
    })
}

/// Full night cycle. v0.1: harvest/mine + stage a mock proposal (no live skill edit).
/// Real replay/reflect/validate backends land next; gate remains conservative.
pub fn run_night(cfg: &GymConfig, stage: bool) -> Result<GymReport> {
    let mut report = dry_run(cfg)?;
    if !stage {
        report
            .notes
            .push("run with stage=false completed harvest/mine only.".into());
        return Ok(report);
    }

    if cfg.backend != "mock" {
        // Keep safe until omp/openai backends exist
        report.notes.push(format!(
            "backend={} not fully implemented; staging mock proposal only (no skill mutation).",
            cfg.backend
        ));
    }

    let proposal = StagedProposal {
        id: Uuid::new_v4().to_string(),
        created_at: Utc::now(),
        target_skill: cfg
            .target_skill
            .clone()
            .unwrap_or_else(|| PathBuf::from("(unset)")),
        summary: format!(
            "Mock gym night: {} sessions → {} tasks. Replay/reflect/validate backends pending.",
            report.sessions, report.tasks
        ),
        task_count: report.tasks,
        session_count: report.sessions,
        mock: true,
        accepted: false,
        notes: vec![
            "v0.1 stages a review artifact only.".into(),
            "No SKILL.md changes until a non-mock backend passes the validation gate and you run adopt.".into(),
        ],
    };

    let path = save_proposal(&cfg.proposal_dir(), &proposal)?;
    let mut state = load_state(&cfg.state_path())?;
    state.last_run_at = Some(Utc::now());
    state.nights_completed = state.nights_completed.saturating_add(1);
    state.last_proposal_id = Some(proposal.id.clone());
    save_state(&cfg.state_path(), &state)?;

    report.staged = true;
    report.proposal_id = Some(proposal.id);
    report.notes.push(format!("Staged proposal → {}", path.display()));
    Ok(report)
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
        format!("nights:      {}", state.nights_completed),
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
            lines.push(format!("proposal:    {} (mock={})", p.id, p.mock));
            lines.push(format!("  summary:   {}", p.summary));
            lines.push(format!("  accepted:  {}", p.accepted));
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
    if proposal.mock {
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
