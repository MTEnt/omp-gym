use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub path: PathBuf,
    pub title: Option<String>,
    pub cwd: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub user_turns: usize,
    pub assistant_turns: usize,
    pub tool_calls: usize,
    /// Truncated user prompts (redaction applied best-effort).
    pub user_excerpts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinedTask {
    pub id: String,
    pub title: String,
    pub prompt: String,
    pub source_session_ids: Vec<String>,
    pub frequency: usize,
    /// false until human marks reviewed for real-backend replay
    pub reviewed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagedProposal {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub target_skill: PathBuf,
    pub summary: String,
    pub task_count: usize,
    pub session_count: usize,
    pub mock: bool,
    pub accepted: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasksFile {
    pub generated_at: DateTime<Utc>,
    pub project: PathBuf,
    pub reviewed: bool,
    pub tasks: Vec<MinedTask>,
}
