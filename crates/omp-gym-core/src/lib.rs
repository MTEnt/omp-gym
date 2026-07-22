//! Project-scoped OMP session harvesting and task-mining primitives.
//!
//! The local data-preparation layer harvests and mines sessions, persists
//! deterministic reviewable tasks, and preserves explicit review decisions.
//! It does not replay tasks, optimize a skill, validate a candidate, or apply
//! skill changes.
//!
//! Planned loop: harvest → review → replay → reflect → validate → stage → adopt.

pub mod config;
pub mod evaluation;
pub mod harvest;
pub mod mine;
pub mod paths;
pub mod pipeline;
pub mod state;
pub mod task_store;
pub mod types;

pub use config::GymConfig;
pub use evaluation::{gate, score_trajectory, split_tasks, validate_check};
pub use pipeline::{dry_run, run_night, GymReport};
pub use state::{load_state, save_state, GymState};
pub use task_store::{
    approve_task, load_tasks, merge_tasks, reject_task, reopen_task, save_tasks, stable_task_id,
    validate_reviewed_tasks,
};
pub use types::{CheckSpec, MinedTask, ReviewStatus, SessionSummary, StagedProposal, TasksFile};
