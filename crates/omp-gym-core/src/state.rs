use crate::paths::{atomic_write, atomic_write_json, load_json};
use crate::types::{RunStatus, StagedProposal, SCHEMA_VERSION};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct GymState {
    pub schema_version: u32,
    pub last_harvest_at: Option<DateTime<Utc>>,
    pub last_run_at: Option<DateTime<Utc>>,
    pub last_run_id: Option<String>,
    pub last_run_status: Option<RunStatus>,
    pub last_session_ids: Vec<String>,
    pub nights_completed: u64,
    pub last_proposal_id: Option<String>,
    pub schedule: Option<ScheduleState>,
}

impl Default for GymState {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            last_harvest_at: None,
            last_run_at: None,
            last_run_id: None,
            last_run_status: None,
            last_session_ids: Vec::new(),
            nights_completed: 0,
            last_proposal_id: None,
            schedule: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleState {
    pub enabled: bool,
    pub hour_local: u32,
    pub minute_local: u32,
    pub label: String,
}

pub fn load_state(path: &Path) -> Result<GymState> {
    if !path.exists() {
        return Ok(GymState::default());
    }
    load_json(path, "state")
}

pub fn save_state(path: &Path, state: &GymState) -> Result<()> {
    atomic_write_json(path, state)
}

pub fn save_proposal(dir: &Path, proposal: &StagedProposal) -> Result<std::path::PathBuf> {
    crate::paths::ensure_dir(dir)?;
    let path = dir.join(format!("{}.json", proposal.id));
    atomic_write_json(&path, proposal)?;
    atomic_write(&dir.join("LATEST"), proposal.id.as_bytes())?;
    Ok(path)
}

pub fn load_latest_proposal(dir: &Path) -> Result<Option<StagedProposal>> {
    let latest = dir.join("LATEST");
    if !latest.exists() {
        return Ok(None);
    }
    let id = std::fs::read_to_string(&latest)
        .with_context(|| format!("read latest proposal pointer {}", latest.display()))?
        .trim()
        .to_string();
    let path = dir.join(format!("{id}.json"));
    if !path.exists() {
        return Ok(None);
    }
    load_json(&path, "proposal").map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RunStatus, SCHEMA_VERSION};
    use tempfile::tempdir;

    #[test]
    fn load_state_reports_corrupt_json_with_path_context() {
        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("state.json");
        std::fs::write(&path, "{not-json").expect("write corrupt state");

        let error = load_state(&path).expect_err("corrupt state must not reset silently");
        let message = format!("{error:#}");
        assert!(message.contains("parse state"));
        assert!(message.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn load_latest_proposal_reports_corrupt_json_with_path_context() {
        let root = tempdir().expect("create temporary directory");
        std::fs::write(root.path().join("LATEST"), "proposal-1").expect("write latest pointer");
        let proposal_path = root.path().join("proposal-1.json");
        std::fs::write(&proposal_path, "{not-json").expect("write corrupt proposal");

        let error = load_latest_proposal(root.path())
            .expect_err("corrupt proposal must not reset silently");
        let message = format!("{error:#}");
        assert!(message.contains("parse proposal"));
        assert!(message.contains(proposal_path.to_string_lossy().as_ref()));
    }

    #[test]
    fn save_state_atomically_replaces_complete_document() {
        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("nested").join("state.json");
        let mut state = GymState::default();
        state.last_session_ids = vec!["obsolete-session-with-a-long-id".into()];
        save_state(&path, &state).expect("write first state");

        state.last_session_ids.clear();
        state.last_run_id = Some("run-2".into());
        state.last_run_status = Some(RunStatus::Rejected);
        save_state(&path, &state).expect("replace state");

        assert_eq!(load_state(&path).expect("load state"), state);
        let names: Vec<_> = std::fs::read_dir(path.parent().expect("state parent"))
            .expect("read state parent")
            .map(|entry| entry.expect("directory entry").file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from("state.json")]);
    }

    #[test]
    fn default_state_uses_current_schema_version() {
        assert_eq!(GymState::default().schema_version, SCHEMA_VERSION);
    }
}
