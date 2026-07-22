use crate::paths::{atomic_write, atomic_write_json};
use crate::types::{RunStatus, StagedProposal, SCHEMA_VERSION};
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Deserialize)]
struct SchemaVersion {
    schema_version: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyGymState {
    last_harvest_at: Option<DateTime<Utc>>,
    last_run_at: Option<DateTime<Utc>>,
    last_session_ids: Vec<String>,
    nights_completed: u64,
    last_proposal_id: Option<String>,
    schedule: Option<ScheduleState>,
}

pub fn load_state(path: &Path) -> Result<GymState> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(GymState::default());
        }
        Err(error) => {
            return Err(error).with_context(|| format!("read state JSON {}", path.display()));
        }
    };
    let version: SchemaVersion = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse state JSON {}", path.display()))?;

    match version.schema_version {
        Some(observed) if observed != SCHEMA_VERSION => bail!(
            "state schema_version mismatch: observed {observed}, supported {SCHEMA_VERSION}, path {}",
            path.display()
        ),
        Some(_) => serde_json::from_slice(&bytes)
            .with_context(|| format!("parse state JSON {}", path.display())),
        None => {
            let legacy: LegacyGymState = serde_json::from_slice(&bytes).with_context(|| {
                format!(
                    "state schema_version mismatch: observed missing, supported {SCHEMA_VERSION}, path {}; unversioned value does not match the supported legacy state shape",
                    path.display()
                )
            })?;
            Ok(GymState {
                schema_version: SCHEMA_VERSION,
                last_harvest_at: legacy.last_harvest_at,
                last_run_at: legacy.last_run_at,
                last_run_id: None,
                last_run_status: None,
                last_session_ids: legacy.last_session_ids,
                nights_completed: legacy.nights_completed,
                last_proposal_id: legacy.last_proposal_id,
                schedule: legacy.schedule,
            })
        }
    }
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
    let raw_id = match std::fs::read_to_string(&latest) {
        Ok(raw_id) => raw_id,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read latest proposal pointer {}", latest.display()));
        }
    };
    let id = raw_id.as_str();
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        bail!(
            "unsafe proposal ID in latest proposal pointer {}: expected nonempty ASCII alphanumeric, '_' or '-'",
            latest.display()
        );
    }

    let path = dir.join(format!("{id}.json"));
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => bail!(
            "proposal store integrity error: latest pointer {} references missing proposal JSON {}",
            latest.display(),
            path.display()
        ),
        Err(error) => {
            return Err(error).with_context(|| format!("read proposal JSON {}", path.display()));
        }
    };
    let version: SchemaVersion = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse proposal JSON {}", path.display()))?;
    match version.schema_version {
        Some(observed) if observed != SCHEMA_VERSION => bail!(
            "proposal schema_version mismatch: observed {observed}, supported {SCHEMA_VERSION}, path {}",
            path.display()
        ),
        None => bail!(
            "proposal schema_version mismatch: observed missing, supported {SCHEMA_VERSION}, path {}",
            path.display()
        ),
        Some(_) => serde_json::from_slice(&bytes)
            .with_context(|| format!("parse proposal JSON {}", path.display()))
            .map(Some),
    }
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
    fn load_state_migrates_legacy_unversioned_state() {
        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("state.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "last_harvest_at": null,
                "last_run_at": null,
                "last_session_ids": ["legacy-session"],
                "nights_completed": 7,
                "last_proposal_id": "proposal-7",
                "schedule": null
            }))
            .expect("serialize legacy state"),
        )
        .expect("write legacy state");

        let state = load_state(&path).expect("migrate legacy state");

        assert_eq!(state.schema_version, SCHEMA_VERSION);
        assert_eq!(state.last_session_ids, vec!["legacy-session"]);
        assert_eq!(state.nights_completed, 7);
        assert_eq!(state.last_proposal_id.as_deref(), Some("proposal-7"));
        assert_eq!(state.last_run_id, None);
        assert_eq!(state.last_run_status, None);
    }

    #[test]
    fn load_state_rejects_unsupported_schema_version_with_context() {
        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("state.json");
        let observed = SCHEMA_VERSION + 1;
        std::fs::write(&path, format!(r#"{{"schema_version":{observed}}}"#))
            .expect("write future state");

        let error = load_state(&path).expect_err("future state must be rejected");
        let message = format!("{error:#}");

        assert!(message.contains(&format!("observed {observed}")));
        assert!(message.contains(&format!("supported {SCHEMA_VERSION}")));
        assert!(message.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn load_state_rejects_unversioned_nonlegacy_shape() {
        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("state.json");
        std::fs::write(&path, "{}").expect("write incomplete unversioned state");

        let error = load_state(&path).expect_err("unknown unversioned state must be rejected");
        let message = format!("{error:#}");

        assert!(message.contains("unversioned"));
        assert!(message.contains("legacy"));
        assert!(message.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn load_state_preserves_non_not_found_read_errors() {
        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("state.json");
        std::fs::create_dir(&path).expect("create directory at state path");

        let error = load_state(&path).expect_err("state read error must be preserved");
        let message = format!("{error:#}");

        assert!(message.contains("read state JSON"));
        assert!(message.contains(path.to_string_lossy().as_ref()));
    }

    #[test]
    fn load_latest_proposal_rejects_unsupported_schema_version_with_context() {
        let root = tempdir().expect("create temporary directory");
        let latest = root.path().join("LATEST");
        let proposal_path = root.path().join("proposal-1.json");
        let observed = SCHEMA_VERSION + 1;
        std::fs::write(&latest, "proposal-1").expect("write latest pointer");
        std::fs::write(
            &proposal_path,
            format!(r#"{{"schema_version":{observed}}}"#),
        )
        .expect("write future proposal");

        let error =
            load_latest_proposal(root.path()).expect_err("future proposal must be rejected");
        let message = format!("{error:#}");

        assert!(message.contains(&format!("observed {observed}")));
        assert!(message.contains(&format!("supported {SCHEMA_VERSION}")));
        assert!(message.contains(proposal_path.to_string_lossy().as_ref()));
    }

    #[test]
    fn load_latest_proposal_rejects_unversioned_metadata() {
        let root = tempdir().expect("create temporary directory");
        let proposal_path = root.path().join("proposal-1.json");
        std::fs::write(root.path().join("LATEST"), "proposal-1").expect("write latest pointer");
        std::fs::write(&proposal_path, r#"{"id":"proposal-1"}"#)
            .expect("write unversioned proposal");

        let error =
            load_latest_proposal(root.path()).expect_err("unversioned proposal must be rejected");
        let message = format!("{error:#}");

        assert!(message.contains("observed missing"));
        assert!(message.contains(&format!("supported {SCHEMA_VERSION}")));
        assert!(message.contains(proposal_path.to_string_lossy().as_ref()));
    }

    #[test]
    fn load_latest_proposal_rejects_unsafe_pointer_ids() {
        for id in [
            "",
            "../escape",
            "nested/id",
            "proposal.1",
            "prøposal",
            " proposal-1 ",
            "proposal-1\n",
        ] {
            let root = tempdir().expect("create temporary directory");
            let latest = root.path().join("LATEST");
            std::fs::write(&latest, id).expect("write latest pointer");

            let error = load_latest_proposal(root.path())
                .expect_err("unsafe latest pointer must be rejected");
            let message = format!("{error:#}");

            assert!(message.contains("unsafe proposal ID"), "message: {message}");
            assert!(
                message.contains(latest.to_string_lossy().as_ref()),
                "message: {message}"
            );
        }
    }

    #[test]
    fn load_latest_proposal_reports_missing_referenced_json_as_integrity_error() {
        let root = tempdir().expect("create temporary directory");
        let latest = root.path().join("LATEST");
        let proposal_path = root.path().join("proposal-1.json");
        std::fs::write(&latest, "proposal-1").expect("write latest pointer");

        let error = load_latest_proposal(root.path())
            .expect_err("broken latest pointer must be an integrity error");
        let message = format!("{error:#}");

        assert!(message.contains("integrity"));
        assert!(message.contains(latest.to_string_lossy().as_ref()));
        assert!(message.contains(proposal_path.to_string_lossy().as_ref()));
    }

    #[test]
    fn load_latest_proposal_preserves_non_not_found_pointer_read_errors() {
        let root = tempdir().expect("create temporary directory");
        let latest = root.path().join("LATEST");
        std::fs::create_dir(&latest).expect("create directory at pointer path");

        let error =
            load_latest_proposal(root.path()).expect_err("pointer read error must be preserved");
        let message = format!("{error:#}");

        assert!(message.contains("read latest proposal pointer"));
        assert!(message.contains(latest.to_string_lossy().as_ref()));
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
