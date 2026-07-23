//! Project-scoped harvesting, review, evaluation, and isolated OMP replay.
//!
//! The core harvests sessions, persists deterministic reviewable tasks,
//! evaluates replay trajectories, runs text-only model requests through an
//! isolated bounded OMP subprocess, and validates bounded skill candidates.
//! Proposal staging and skill adoption remain outside this layer.

pub mod config;
pub mod evaluation;
pub mod harvest;
pub mod mine;
pub mod optimizer;
pub mod paths;
pub mod pipeline;
pub mod privacy;
pub mod runner;
pub mod state;
pub mod task_store;
pub mod types;

pub use config::GymConfig;
pub use evaluation::{gate, score_trajectory, split_tasks, validate_check};
pub use optimizer::{
    build_judge_prompt, build_optimizer_prompt, parse_candidate, parse_judge, unified_diff,
    validate_candidate, CandidateLimits, JudgeInput, JudgeOrder, JudgePrompt, JudgeSide,
    OptimizerTrainingInput, ParsedCandidate,
};
pub use pipeline::{dry_run, run_night, GymReport};
pub use privacy::{bound_chars, redact_json_strings, redact_text};
pub use runner::{ModelRequest, ModelRunner, OmpRunner};
pub use state::{load_state, save_state, GymState};
pub use task_store::{
    approve_task, load_tasks, merge_tasks, reject_task, reopen_task, save_tasks, stable_task_id,
    validate_reviewed_tasks,
};
pub use types::{
    CandidateBounds, CheckResult, CheckSpec, JudgeEvidence, MinedTask, ModelRole, ReviewStatus,
    SessionSummary, StagedProposal, TaskScore, TasksFile, Trajectory,
};
