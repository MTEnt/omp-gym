//! Project-scoped OMP session harvesting and task-mining primitives.
//!
//! v0.1 implements the local data-preparation layer: harvest, redact, mine,
//! status, dry-run, and mock proposal metadata. It does not replay tasks,
//! optimize a skill, validate a candidate, or apply skill changes.
//!
//! Planned loop: harvest → review → replay → reflect → validate → stage → adopt.

pub mod config;
pub mod harvest;
pub mod mine;
pub mod paths;
pub mod pipeline;
pub mod state;
pub mod types;

pub use config::GymConfig;
pub use pipeline::{dry_run, run_night, GymReport};
pub use state::{load_state, save_state, GymState};
pub use types::{MinedTask, SessionSummary, StagedProposal};
