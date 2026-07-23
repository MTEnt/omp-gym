use crate::paths::{atomic_write_json, load_json, omp_sessions_root, project_gym_dir};
use crate::types::{ModelRole, SCHEMA_VERSION};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GymConfig {
    pub schema_version: u32,
    pub project: PathBuf,
    pub sessions_root: PathBuf,
    pub target_skill: Option<PathBuf>,
    pub lookback_hours: u64,
    pub max_sessions: usize,
    pub max_tasks: usize,
    pub backend: String,
    pub omp_bin: PathBuf,
    pub replay_model: Option<String>,
    pub optimizer_model: Option<String>,
    pub judge_model: Option<String>,
    pub replay_timeout_secs: u64,
    pub optimizer_timeout_secs: u64,
    pub judge_timeout_secs: u64,
    pub judge_enabled: bool,
    pub validation_ratio: f64,
    pub min_validation_tasks: usize,
    pub min_score_delta: f64,
    pub max_output_bytes: usize,
    pub max_candidate_bytes: usize,
    pub max_growth_ratio: f64,
    pub max_changed_lines: usize,
}

impl GymConfig {
    pub fn for_project(project: impl Into<PathBuf>) -> Result<Self> {
        let project = absolute_project(project.into())?;
        Self::defaults(project)
    }

    pub fn load(project: impl AsRef<Path>) -> Result<Self> {
        let project = absolute_project(project.as_ref().to_path_buf())?;
        let path = project_gym_dir(&project).join("config.json");
        if !path.exists() {
            return Self::for_project(project);
        }

        let mut defaults = serde_json::to_value(Self::defaults(project.clone())?)
            .context("serialize default gym config")?;
        let persisted: serde_json::Value = load_json(&path, "gym config")?;
        let defaults = defaults
            .as_object_mut()
            .context("default gym config must serialize as an object")?;
        let persisted = persisted.as_object().with_context(|| {
            format!("gym config JSON root must be an object: {}", path.display())
        })?;
        if let Some(version) = persisted
            .get("schema_version")
            .and_then(|value| value.as_u64())
        {
            if version != u64::from(SCHEMA_VERSION) {
                bail!("unsupported gym config schema version {version}");
            }
        }
        for (key, value) in persisted {
            if key != "project" {
                defaults.insert(key.clone(), value.clone());
            }
        }
        defaults.insert(
            "project".into(),
            serde_json::to_value(project).context("serialize canonical project path")?,
        );
        serde_json::from_value(serde_json::Value::Object(defaults.clone()))
            .with_context(|| format!("parse gym config JSON {}", path.display()))
    }

    pub fn save(&self) -> Result<()> {
        atomic_write_json(&self.gym_dir().join("config.json"), self)
            .with_context(|| format!("save gym config for {}", self.project.display()))
    }

    pub fn validate_for_run(&self) -> Result<PathBuf> {
        if self.replay_timeout_secs == 0
            || self.optimizer_timeout_secs == 0
            || self.judge_timeout_secs == 0
        {
            bail!("model timeout seconds must be greater than zero");
        }
        if !self.validation_ratio.is_finite()
            || self.validation_ratio <= 0.0
            || self.validation_ratio >= 1.0
        {
            bail!("validation ratio must be finite and between zero and one");
        }
        if self.min_validation_tasks < 2 {
            bail!("minimum validation task count must be at least two");
        }
        if !self.min_score_delta.is_finite()
            || self.min_score_delta <= 0.0
            || self.min_score_delta > 1.0
        {
            bail!("minimum score delta must be finite, positive, and at most one");
        }
        if self.max_output_bytes < 2 {
            bail!("output byte bound must be at least two");
        }
        if self.max_candidate_bytes == 0 {
            bail!("candidate byte bound must be greater than zero");
        }
        if !self.max_growth_ratio.is_finite() || self.max_growth_ratio < 1.0 {
            bail!("maximum growth ratio must be finite and at least one");
        }
        if self.max_changed_lines == 0 {
            bail!("maximum changed lines must be greater than zero");
        }

        let target = self
            .target_skill
            .as_ref()
            .context("target skill is required to run")?;
        let target = if target.is_absolute() {
            target.clone()
        } else {
            self.project.join(target)
        };
        target
            .canonicalize()
            .with_context(|| format!("canonicalize target skill {}", target.display()))
    }

    pub fn model_for(&self, role: &ModelRole) -> Option<&str> {
        match role {
            ModelRole::Replay => self.replay_model.as_deref(),
            ModelRole::Optimizer => self.optimizer_model.as_deref(),
            ModelRole::Judge => self.judge_model.as_deref(),
        }
    }

    pub fn timeout_secs_for(&self, role: &ModelRole) -> u64 {
        match role {
            ModelRole::Replay => self.replay_timeout_secs,
            ModelRole::Optimizer => self.optimizer_timeout_secs,
            ModelRole::Judge => self.judge_timeout_secs,
        }
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

    fn defaults(project: PathBuf) -> Result<Self> {
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            project,
            sessions_root: omp_sessions_root()?,
            target_skill: None,
            lookback_hours: 72,
            max_sessions: 20,
            max_tasks: 10,
            backend: "mock".into(),
            omp_bin: PathBuf::from("omp"),
            replay_model: None,
            optimizer_model: None,
            judge_model: None,
            replay_timeout_secs: 300,
            optimizer_timeout_secs: 600,
            judge_timeout_secs: 300,
            judge_enabled: true,
            validation_ratio: 0.40,
            min_validation_tasks: 2,
            min_score_delta: 0.05,
            max_output_bytes: 1_048_576,
            max_candidate_bytes: 32_768,
            max_growth_ratio: 1.5,
            max_changed_lines: 120,
        })
    }
}

fn absolute_project(project: PathBuf) -> Result<PathBuf> {
    let absolute = if project.is_absolute() {
        project
    } else {
        std::env::current_dir()?.join(project)
    };
    absolute
        .canonicalize()
        .with_context(|| format!("canonicalize project {}", absolute.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_resolves_project_relative_target_and_round_trips() {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(project.join("skills/demo")).unwrap();
        std::fs::write(
            project.join("skills/demo/SKILL.md"),
            "---\nname: demo\n---\n",
        )
        .unwrap();

        let mut config = GymConfig::load(&project).unwrap();
        config.target_skill = Some(PathBuf::from("skills/demo/SKILL.md"));
        config.save().unwrap();

        let loaded = GymConfig::load(&project).unwrap();
        assert_eq!(
            loaded.validate_for_run().unwrap(),
            project.canonicalize().unwrap().join("skills/demo/SKILL.md")
        );
        assert_eq!(loaded.validation_ratio, 0.40);
        assert_eq!(loaded.min_validation_tasks, 2);
        assert_eq!(loaded.max_output_bytes, 1_048_576);
    }

    #[test]
    fn validate_for_run_rejects_invalid_bounds_and_timeouts() {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(&project).unwrap();
        let skill = project.join("SKILL.md");
        std::fs::write(&skill, "skill").unwrap();
        let mut config = GymConfig::load(&project).unwrap();
        config.target_skill = Some(PathBuf::from("SKILL.md"));

        config.replay_timeout_secs = 0;
        assert!(config
            .validate_for_run()
            .unwrap_err()
            .to_string()
            .contains("timeout"));
        config.replay_timeout_secs = 60;
        config.validation_ratio = 1.0;
        assert!(config
            .validate_for_run()
            .unwrap_err()
            .to_string()
            .contains("validation ratio"));
        config.validation_ratio = 0.4;
        config.min_score_delta = f64::NAN;
        assert!(config
            .validate_for_run()
            .unwrap_err()
            .to_string()
            .contains("score delta"));
        config.min_score_delta = 0.05;
        config.max_growth_ratio = 0.5;
        assert!(config
            .validate_for_run()
            .unwrap_err()
            .to_string()
            .contains("growth ratio"));
        config.max_growth_ratio = 1.5;
        config.max_output_bytes = 1;
        assert!(config
            .validate_for_run()
            .unwrap_err()
            .to_string()
            .contains("output"));
    }

    #[test]
    fn load_overlays_partial_persisted_config_on_safe_defaults() {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(project.join(".omp/gym")).unwrap();
        std::fs::write(
            project.join(".omp/gym/config.json"),
            r#"{"replay_timeout_secs":17,"judge_enabled":false}"#,
        )
        .unwrap();

        let config = GymConfig::load(&project).unwrap();
        assert_eq!(config.replay_timeout_secs, 17);
        assert!(!config.judge_enabled);
        assert_eq!(config.optimizer_timeout_secs, 600);
        assert_eq!(config.omp_bin, PathBuf::from("omp"));
        assert_eq!(config.schema_version, crate::types::SCHEMA_VERSION);
    }

    #[test]
    fn load_rejects_unsupported_config_schema() {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(project.join(".omp/gym")).unwrap();
        std::fs::write(
            project.join(".omp/gym/config.json"),
            r#"{"schema_version":999}"#,
        )
        .unwrap();

        let error = GymConfig::load(&project).unwrap_err().to_string();
        assert!(error.contains("unsupported gym config schema version 999"));
    }

    #[test]
    fn load_reports_corrupt_persisted_config() {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(project.join(".omp/gym")).unwrap();
        std::fs::write(project.join(".omp/gym/config.json"), "{broken").unwrap();

        let error = GymConfig::load(&project).unwrap_err().to_string();
        assert!(error.contains("parse gym config JSON"));
        assert!(error.contains("config.json"));
    }
}
