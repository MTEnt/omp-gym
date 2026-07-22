use crate::paths::{omp_sessions_root, project_gym_dir};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GymConfig {
    pub project: PathBuf,
    /// OMP sessions root override
    pub sessions_root: PathBuf,
    /// Skill file to improve (SKILL.md)
    pub target_skill: Option<PathBuf>,
    /// Hours of history to harvest (0 = all, subject to max_sessions)
    pub lookback_hours: u64,
    pub max_sessions: usize,
    pub max_tasks: usize,
    /// mock | omp | openai_compatible (v0.1: mock fully implemented)
    pub backend: String,
    pub auto_adopt: bool,
}

impl GymConfig {
    pub fn for_project(project: impl Into<PathBuf>) -> Result<Self> {
        let project = project.into();
        let project = if project.is_absolute() {
            project
        } else {
            std::env::current_dir()?.join(project)
        };
        Ok(Self {
            project,
            sessions_root: omp_sessions_root()?,
            target_skill: None,
            lookback_hours: 72,
            max_sessions: 20,
            max_tasks: 10,
            backend: "mock".into(),
            auto_adopt: false,
        })
    }

    pub fn gym_dir(&self) -> PathBuf {
        project_gym_dir(&self.project)
    }

    pub fn state_path(&self) -> PathBuf {
        self.gym_dir().join("state.json")
    }

    pub fn tasks_path(&self) -> PathBuf {
        self.gym_dir().join("tasks.json")
    }

    pub fn proposal_dir(&self) -> PathBuf {
        self.gym_dir().join("proposals")
    }

    pub fn with_target_skill(mut self, path: impl AsRef<Path>) -> Self {
        self.target_skill = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = backend.into();
        self
    }

    pub fn with_limits(mut self, max_sessions: usize, max_tasks: usize) -> Self {
        self.max_sessions = max_sessions;
        self.max_tasks = max_tasks;
        self
    }
}
