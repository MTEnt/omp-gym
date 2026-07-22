//! omp-gym-core — SkillOpt-Sleep-inspired gym loop for OMP skills.
//!
//! Pipeline:
//! harvest OMP sessions → mine recurring tasks → replay → reflect
//! → validate (held-out gate) → stage → (user) adopt
//!
//! v0.1 implements harvest/mine/status/dry-run/stage scaffolding.
//! Replay/reflect/validate backends land next; mock path works offline.

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
