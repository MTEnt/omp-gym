use crate::types::StagedProposal;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GymState {
    pub last_harvest_at: Option<DateTime<Utc>>,
    pub last_run_at: Option<DateTime<Utc>>,
    pub last_session_ids: Vec<String>,
    pub nights_completed: u64,
    pub last_proposal_id: Option<String>,
    pub schedule: Option<ScheduleState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read state {}", path.display()))?;
    Ok(serde_json::from_str(&raw).unwrap_or_default())
}

pub fn save_state(path: &Path, state: &GymState) -> Result<()> {
    if let Some(parent) = path.parent() {
        crate::paths::ensure_dir(parent)?;
    }
    let raw = serde_json::to_string_pretty(state)?;
    std::fs::write(path, raw).with_context(|| format!("write state {}", path.display()))?;
    Ok(())
}

pub fn save_proposal(dir: &Path, proposal: &StagedProposal) -> Result<std::path::PathBuf> {
    crate::paths::ensure_dir(dir)?;
    let path = dir.join(format!("{}.json", proposal.id));
    let raw = serde_json::to_string_pretty(proposal)?;
    std::fs::write(&path, raw)?;
    // also write latest pointer
    std::fs::write(dir.join("LATEST"), proposal.id.as_bytes())?;
    Ok(path)
}

pub fn load_latest_proposal(dir: &Path) -> Result<Option<StagedProposal>> {
    let latest = dir.join("LATEST");
    if !latest.exists() {
        return Ok(None);
    }
    let id = std::fs::read_to_string(latest)?.trim().to_string();
    let path = dir.join(format!("{id}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path)?;
    Ok(Some(serde_json::from_str(&raw)?))
}
