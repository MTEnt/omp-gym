use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus {
    Pending,
    Approved,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CheckSpec {
    Exact { value: String },
    Contains { value: String, case_sensitive: bool },
    NotContains { value: String, case_sensitive: bool },
    Regex { pattern: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MinedTask {
    pub id: String,
    pub title: String,
    pub prompt: String,
    pub source_session_ids: Vec<String>,
    #[serde(default)]
    pub source_occurrences: BTreeMap<String, usize>,
    pub frequency: usize,
    pub status: ReviewStatus,
    pub checks: Vec<CheckSpec>,
    pub rubric: Option<String>,
    pub review_note: Option<String>,
    pub reviewed_at: Option<DateTime<Utc>>,
    pub first_seen_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Replay,
    Optimizer,
    Judge,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Trajectory {
    pub schema_version: u32,
    pub id: String,
    pub role: ModelRole,
    pub task_id: Option<String>,
    pub started_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub prompt_hash: String,
    pub skill_hash: String,
    pub model: Option<String>,
    pub process_success: bool,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub response_nonempty: bool,
    pub final_text: Option<String>,
    pub events: Vec<serde_json::Value>,
    pub stderr: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckResult {
    pub check: CheckSpec,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskScore {
    pub task_id: String,
    pub passed_checks: usize,
    pub total_checks: usize,
    pub score: f64,
    pub invariants_passed: bool,
    pub check_results: Vec<CheckResult>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskSplit {
    pub train_ids: Vec<String>,
    pub validation_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GateDecision {
    pub accepted: bool,
    pub baseline_mean: f64,
    pub candidate_mean: f64,
    pub delta: f64,
    pub improved_checks: usize,
    pub regressions: Vec<String>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Accepted,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    pub schema_version: u32,
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub project: PathBuf,
    pub target_skill: PathBuf,
    pub status: RunStatus,
    pub task_store_hash: String,
    pub base_skill_hash: String,
    pub candidate_skill_hash: Option<String>,
    pub split: Option<TaskSplit>,
    pub baseline_scores: Vec<TaskScore>,
    pub candidate_scores: Vec<TaskScore>,
    pub gate: Option<GateDecision>,
    pub trajectory_ids: Vec<String>,
    pub evidence_path: PathBuf,
    pub proposal_id: Option<String>,
    pub error: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Accepted,
    Adopted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JudgeEvidence {
    pub task_id: String,
    pub winner: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateBounds {
    pub base_bytes: usize,
    pub candidate_bytes: usize,
    pub growth_ratio: f64,
    pub changed_lines: usize,
    pub max_candidate_bytes: usize,
    pub max_growth_ratio: f64,
    pub max_changed_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StagedProposal {
    pub schema_version: u32,
    pub id: String,
    pub run_id: String,
    pub created_at: DateTime<Utc>,
    pub adopted_at: Option<DateTime<Utc>>,
    pub target_skill: PathBuf,
    pub status: ProposalStatus,
    pub summary: String,
    pub base_skill_hash: String,
    pub candidate_skill_hash: String,
    pub task_store_hash: String,
    pub split: TaskSplit,
    pub baseline_scores: Vec<TaskScore>,
    pub candidate_scores: Vec<TaskScore>,
    pub gate: GateDecision,
    pub edit_bounds: CandidateBounds,
    pub candidate_path: PathBuf,
    pub diff_path: PathBuf,
    pub evidence_path: PathBuf,
    pub backup_path: Option<PathBuf>,
    pub judge_evidence: Vec<JudgeEvidence>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TasksFile {
    pub schema_version: u32,
    pub generated_at: DateTime<Utc>,
    pub project: PathBuf,
    pub tasks: Vec<MinedTask>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn timestamp() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 23, 12, 0, 0)
            .single()
            .expect("valid timestamp")
    }

    fn task_score() -> TaskScore {
        TaskScore {
            task_id: "task-1".into(),
            passed_checks: 1,
            total_checks: 1,
            score: 1.0,
            invariants_passed: true,
            check_results: vec![CheckResult {
                check: CheckSpec::Contains {
                    value: "done".into(),
                    case_sensitive: false,
                },
                passed: true,
                detail: "matched".into(),
            }],
            reasons: vec![],
        }
    }

    fn gate() -> GateDecision {
        GateDecision {
            accepted: true,
            baseline_mean: 0.0,
            candidate_mean: 1.0,
            delta: 1.0,
            improved_checks: 1,
            regressions: vec![],
            reasons: vec!["improved".into()],
        }
    }

    #[test]
    fn schema_contracts_round_trip() {
        let now = timestamp();
        let task = MinedTask {
            id: "task-1".into(),
            title: "Example".into(),
            prompt: "Return done".into(),
            source_session_ids: vec!["session-1".into()],
            source_occurrences: BTreeMap::from([("session-1".into(), 2)]),
            frequency: 2,
            status: ReviewStatus::Approved,
            checks: vec![CheckSpec::Exact {
                value: "done".into(),
            }],
            rubric: Some("Answer exactly".into()),
            review_note: Some("reviewed".into()),
            reviewed_at: Some(now),
            first_seen_at: now,
            last_seen_at: now,
        };
        let tasks = TasksFile {
            schema_version: SCHEMA_VERSION,
            generated_at: now,
            project: PathBuf::from("/project"),
            tasks: vec![task],
        };
        let trajectory = Trajectory {
            schema_version: SCHEMA_VERSION,
            id: "trajectory-1".into(),
            role: ModelRole::Replay,
            task_id: Some("task-1".into()),
            started_at: now,
            duration_ms: 42,
            prompt_hash: "prompt-hash".into(),
            skill_hash: "skill-hash".into(),
            model: Some("model".into()),
            process_success: true,
            exit_code: Some(0),
            timed_out: false,
            response_nonempty: true,
            final_text: Some("done".into()),
            events: vec![serde_json::json!({"type": "result"})],
            stderr: String::new(),
            error: None,
        };
        let split = TaskSplit {
            train_ids: vec!["task-1".into()],
            validation_ids: vec!["task-2".into()],
        };
        let run = RunRecord {
            schema_version: SCHEMA_VERSION,
            id: "run-1".into(),
            created_at: now,
            finished_at: Some(now),
            project: PathBuf::from("/project"),
            target_skill: PathBuf::from("/project/SKILL.md"),
            status: RunStatus::Accepted,
            task_store_hash: "tasks-hash".into(),
            base_skill_hash: "base-hash".into(),
            candidate_skill_hash: Some("candidate-hash".into()),
            split: Some(split.clone()),
            baseline_scores: vec![task_score()],
            candidate_scores: vec![task_score()],
            gate: Some(gate()),
            trajectory_ids: vec!["trajectory-1".into()],
            evidence_path: PathBuf::from("/project/evidence.jsonl"),
            proposal_id: Some("proposal-1".into()),
            error: None,
            notes: vec!["complete".into()],
        };
        let proposal = StagedProposal {
            schema_version: SCHEMA_VERSION,
            id: "proposal-1".into(),
            run_id: "run-1".into(),
            created_at: now,
            adopted_at: None,
            target_skill: PathBuf::from("/project/SKILL.md"),
            status: ProposalStatus::Accepted,
            summary: "Improves exact output".into(),
            base_skill_hash: "base-hash".into(),
            candidate_skill_hash: "candidate-hash".into(),
            task_store_hash: "tasks-hash".into(),
            split,
            baseline_scores: vec![task_score()],
            candidate_scores: vec![task_score()],
            gate: gate(),
            edit_bounds: CandidateBounds {
                base_bytes: 1_000,
                candidate_bytes: 1_100,
                growth_ratio: 1.1,
                changed_lines: 8,
                max_candidate_bytes: 2_000,
                max_growth_ratio: 1.25,
                max_changed_lines: 20,
            },
            candidate_path: PathBuf::from("/project/candidate.SKILL.md"),
            diff_path: PathBuf::from("/project/skill.diff"),
            evidence_path: PathBuf::from("/project/evidence.jsonl"),
            backup_path: None,
            judge_evidence: vec![JudgeEvidence {
                task_id: "task-1".into(),
                winner: "candidate".into(),
                rationale: "more precise".into(),
            }],
            notes: vec!["review before adoption".into()],
        };

        let tasks_json = serde_json::to_string(&tasks).expect("serialize tasks");
        let trajectory_json = serde_json::to_string(&trajectory).expect("serialize trajectory");
        let run_json = serde_json::to_string(&run).expect("serialize run");
        let proposal_json = serde_json::to_string(&proposal).expect("serialize proposal");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&proposal_json).expect("valid proposal json")
                ["edit_bounds"],
            serde_json::json!({
                "base_bytes": 1_000,
                "candidate_bytes": 1_100,
                "growth_ratio": 1.1,
                "changed_lines": 8,
                "max_candidate_bytes": 2_000,
                "max_growth_ratio": 1.25,
                "max_changed_lines": 20
            })
        );

        assert_eq!(
            serde_json::from_str::<TasksFile>(&tasks_json).expect("deserialize tasks"),
            tasks
        );
        assert_eq!(
            serde_json::from_str::<Trajectory>(&trajectory_json).expect("deserialize trajectory"),
            trajectory
        );
        assert_eq!(
            serde_json::from_str::<RunRecord>(&run_json).expect("deserialize run"),
            run
        );
        assert_eq!(
            serde_json::from_str::<StagedProposal>(&proposal_json).expect("deserialize proposal"),
            proposal
        );
        for json in [tasks_json, trajectory_json, run_json, proposal_json] {
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&json).expect("valid json")
                    ["schema_version"],
                SCHEMA_VERSION
            );
        }
    }

    fn assert_json_round_trip<T>(value: T, expected: serde_json::Value)
    where
        T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let encoded = serde_json::to_value(&value).expect("serialize variant");
        assert_eq!(encoded, expected);
        let decoded: T = serde_json::from_value(encoded).expect("deserialize variant");
        assert_eq!(decoded, value);
    }

    #[test]
    fn review_status_variants_round_trip() {
        for (variant, name) in [
            (ReviewStatus::Pending, "pending"),
            (ReviewStatus::Approved, "approved"),
            (ReviewStatus::Rejected, "rejected"),
        ] {
            assert_json_round_trip(variant, serde_json::json!(name));
        }
    }

    #[test]
    fn check_spec_variants_round_trip() {
        let cases = [
            (
                CheckSpec::Exact {
                    value: "done".into(),
                },
                serde_json::json!({"kind": "exact", "value": "done"}),
            ),
            (
                CheckSpec::Contains {
                    value: "done".into(),
                    case_sensitive: true,
                },
                serde_json::json!({
                    "kind": "contains",
                    "value": "done",
                    "case_sensitive": true
                }),
            ),
            (
                CheckSpec::NotContains {
                    value: "secret".into(),
                    case_sensitive: false,
                },
                serde_json::json!({
                    "kind": "not_contains",
                    "value": "secret",
                    "case_sensitive": false
                }),
            ),
            (
                CheckSpec::Regex {
                    pattern: "^done$".into(),
                },
                serde_json::json!({"kind": "regex", "pattern": "^done$"}),
            ),
        ];

        for (variant, expected) in cases {
            assert_json_round_trip(variant, expected);
        }
    }

    #[test]
    fn model_role_variants_round_trip() {
        for (variant, name) in [
            (ModelRole::Replay, "replay"),
            (ModelRole::Optimizer, "optimizer"),
            (ModelRole::Judge, "judge"),
        ] {
            assert_json_round_trip(variant, serde_json::json!(name));
        }
    }

    #[test]
    fn run_status_variants_round_trip() {
        for (variant, name) in [
            (RunStatus::Running, "running"),
            (RunStatus::Accepted, "accepted"),
            (RunStatus::Rejected, "rejected"),
            (RunStatus::Failed, "failed"),
        ] {
            assert_json_round_trip(variant, serde_json::json!(name));
        }
    }

    #[test]
    fn proposal_status_variants_round_trip() {
        for (variant, name) in [
            (ProposalStatus::Accepted, "accepted"),
            (ProposalStatus::Adopted, "adopted"),
        ] {
            assert_json_round_trip(variant, serde_json::json!(name));
        }
    }
}
