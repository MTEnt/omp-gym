use crate::types::{
    CheckResult, CheckSpec, GateDecision, MinedTask, ReviewStatus, TaskScore, TaskSplit, Trajectory,
};
use anyhow::{bail, Context, Result};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use unicode_casefold::UnicodeCaseFold;

const SCORE_EPSILON: f64 = 1e-9;
const SPLIT_DOMAIN: &[u8] = b"omp-gym-split-v1\0";
const SCORE_VALIDATION_EPSILON: f64 = 1e-12;

pub fn validate_check(check: &CheckSpec) -> Result<()> {
    match check {
        CheckSpec::Exact { value }
        | CheckSpec::Contains { value, .. }
        | CheckSpec::NotContains { value, .. } => {
            if value.trim().is_empty() {
                bail!("task checks cannot contain an empty value");
            }
        }
        CheckSpec::Regex { pattern } => {
            if pattern.trim().is_empty() {
                bail!("task regex check cannot be empty");
            }
            Regex::new(pattern).with_context(|| format!("invalid task regex check {pattern:?}"))?;
        }
    }
    Ok(())
}

pub fn score_trajectory(task: &MinedTask, trajectory: &Trajectory) -> TaskScore {
    let final_text = trajectory.final_text.as_deref().unwrap_or_default();
    let mut reasons = Vec::new();
    if trajectory.task_id.as_deref() != Some(task.id.as_str()) {
        reasons.push("trajectory is not the requested validation replay".to_owned());
    }
    if trajectory.role != crate::types::ModelRole::Replay {
        reasons.push("trajectory role is not a validation replay".to_owned());
    }
    if !trajectory.process_success {
        reasons.push("trajectory process failed".to_owned());
    }
    if trajectory.timed_out {
        reasons.push("trajectory timed out".to_owned());
    }
    if !trajectory.response_nonempty {
        reasons.push("trajectory response marked empty".to_owned());
    }
    match trajectory.final_text.as_deref() {
        None => reasons.push("trajectory final response missing".to_owned()),
        Some(text) if text.trim().is_empty() => {
            reasons.push("trajectory final response empty".to_owned());
        }
        Some(_) => {}
    }
    if trajectory.error.is_some() {
        reasons.push("trajectory reported an error".to_owned());
    }

    let mut check_results = Vec::with_capacity(task.checks.len());
    for (index, check) in task.checks.iter().enumerate() {
        let (passed, detail) = match validate_check(check) {
            Ok(()) => evaluate_check(check, final_text),
            Err(_) => {
                reasons.push(format!("invalid check {index}: validation failed"));
                (false, "invalid check".to_owned())
            }
        };
        check_results.push(CheckResult {
            check: check.clone(),
            passed,
            detail,
        });
    }
    let invariants_passed = reasons.is_empty();

    let passed_checks = check_results.iter().filter(|result| result.passed).count();
    let total_checks = check_results.len();
    let score = if invariants_passed && total_checks != 0 {
        passed_checks as f64 / total_checks as f64
    } else {
        0.0
    };

    TaskScore {
        task_id: task.id.clone(),
        passed_checks,
        total_checks,
        score,
        invariants_passed,
        check_results,
        reasons,
    }
}

fn evaluate_check(check: &CheckSpec, final_text: &str) -> (bool, String) {
    match check {
        CheckSpec::Exact { value } => {
            let passed = final_text.trim() == value;
            (passed, outcome_detail("exact", passed))
        }
        CheckSpec::Contains {
            value,
            case_sensitive,
        } => {
            let passed = contains(final_text, value, *case_sensitive);
            (passed, outcome_detail("contains", passed))
        }
        CheckSpec::NotContains {
            value,
            case_sensitive,
        } => {
            let passed = !contains(final_text, value, *case_sensitive);
            (passed, outcome_detail("not_contains", passed))
        }
        CheckSpec::Regex { pattern } => {
            let passed = Regex::new(pattern)
                .map(|regex| regex.is_match(final_text))
                .unwrap_or(false);
            (passed, outcome_detail("regex", passed))
        }
    }
}

fn contains(text: &str, value: &str, case_sensitive: bool) -> bool {
    if case_sensitive {
        text.contains(value)
    } else {
        text.case_fold()
            .collect::<String>()
            .contains(&value.case_fold().collect::<String>())
    }
}

fn outcome_detail(kind: &str, passed: bool) -> String {
    format!("{kind} check {}", if passed { "passed" } else { "failed" })
}

fn diagnostic_id(id: &str) -> String {
    const MAX_BYTES: usize = 64;
    let mut escaped = String::with_capacity(MAX_BYTES + 3);
    let mut truncated = false;
    for character in id.chars() {
        let escaped_character = character.escape_default();
        let piece_len = escaped_character.clone().count();
        if escaped.len() + piece_len > MAX_BYTES {
            truncated = true;
            break;
        }
        escaped.extend(escaped_character);
    }
    if truncated {
        escaped.push_str("...");
    }
    escaped
}

pub fn split_tasks(
    tasks: &[&MinedTask],
    validation_ratio: f64,
    min_validation: usize,
) -> Result<TaskSplit> {
    if tasks.len() < 5 {
        bail!("task split requires at least 5 approved tasks");
    }
    if !validation_ratio.is_finite() || validation_ratio <= 0.0 || validation_ratio >= 1.0 {
        bail!("validation ratio must be finite and between 0 and 1");
    }
    if min_validation < 2 {
        bail!("minimum validation task count must be at least 2");
    }
    if tasks
        .iter()
        .any(|task| task.status != ReviewStatus::Approved)
    {
        bail!("task split accepts only approved tasks");
    }

    let mut ids = HashSet::with_capacity(tasks.len());
    let mut ranked = Vec::with_capacity(tasks.len());
    for task in tasks {
        if !ids.insert(task.id.as_str()) {
            bail!(
                "duplicate task ID in split input: {}",
                diagnostic_id(&task.id)
            );
        }
        let mut hasher = Sha256::new();
        hasher.update(SPLIT_DOMAIN);
        hasher.update(task.id.as_bytes());
        let hash: [u8; 32] = hasher.finalize().into();
        ranked.push((hash, task.id.clone()));
    }
    ranked.sort_unstable_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let ratio_count = (tasks.len() as f64 * validation_ratio).ceil() as usize;
    let validation_count = min_validation.max(ratio_count).min(tasks.len() - 3);
    if validation_count < 2 {
        bail!("task split must retain at least 2 validation tasks");
    }

    let mut validation_ids = ranked[..validation_count]
        .iter()
        .map(|(_, id)| id.clone())
        .collect::<Vec<_>>();
    let mut train_ids = ranked[validation_count..]
        .iter()
        .map(|(_, id)| id.clone())
        .collect::<Vec<_>>();
    validation_ids.sort_unstable();
    train_ids.sort_unstable();

    Ok(TaskSplit {
        train_ids,
        validation_ids,
    })
}

pub fn gate(baseline: &[TaskScore], candidate: &[TaskScore], min_delta: f64) -> GateDecision {
    let mut reasons = Vec::new();
    let mut regressions = Vec::new();

    if !min_delta.is_finite() || min_delta < 0.0 {
        reasons.push("min_delta must be finite and non-negative".to_owned());
    }
    if baseline.is_empty() || candidate.is_empty() {
        reasons.push("gate requires at least one baseline and candidate task score".to_owned());
    }

    validate_scores(baseline, "baseline", &mut reasons);
    validate_scores(candidate, "candidate", &mut reasons);
    let baseline_mean = derived_mean(baseline);
    let candidate_mean = derived_mean(candidate);
    let delta = candidate_mean - baseline_mean;
    let baseline_by_id = scores_by_id(baseline, "baseline", &mut reasons);
    let candidate_by_id = scores_by_id(candidate, "candidate", &mut reasons);
    if baseline_by_id.keys().ne(candidate_by_id.keys()) {
        reasons.push("baseline and candidate task ID sets differ".to_owned());
    }
    for (task_id, score) in &baseline_by_id {
        if !score.invariants_passed {
            reasons.push(format!(
                "baseline task {} invariants failed",
                diagnostic_id(task_id)
            ));
        }
    }
    for (task_id, score) in &candidate_by_id {
        if !score.invariants_passed {
            reasons.push(format!(
                "candidate task {} invariants failed",
                diagnostic_id(task_id)
            ));
        }
    }

    let mut improved_checks = 0;
    for (task_id, baseline_score) in &baseline_by_id {
        let Some(candidate_score) = candidate_by_id.get(task_id) else {
            continue;
        };
        let task_id = diagnostic_id(task_id);
        let baseline_derived = derived_score(baseline_score);
        let candidate_derived = derived_score(candidate_score);
        if candidate_derived + SCORE_EPSILON < baseline_derived {
            regressions.push(format!(
                "task {task_id} regressed: score {baseline_derived:.6} -> {candidate_derived:.6}"
            ));
        }
        if baseline_score.invariants_passed && !candidate_score.invariants_passed {
            regressions.push(format!("task {task_id} invariants regressed"));
        }
        if baseline_score.check_results.len() != candidate_score.check_results.len()
            || baseline_score
                .check_results
                .iter()
                .zip(&candidate_score.check_results)
                .any(|(baseline_result, candidate_result)| {
                    baseline_result.check != candidate_result.check
                })
        {
            reasons.push(format!("task {task_id} check results are not aligned"));
            continue;
        }
        for (index, (baseline_result, candidate_result)) in baseline_score
            .check_results
            .iter()
            .zip(&candidate_score.check_results)
            .enumerate()
        {
            match (baseline_result.passed, candidate_result.passed) {
                (true, false) => {
                    regressions.push(format!("task {task_id} check {index} regressed"));
                }
                (false, true) => improved_checks += 1,
                _ => {}
            }
        }
    }
    reasons.extend(regressions.iter().cloned());

    if min_delta.is_finite() && min_delta >= 0.0 && delta + SCORE_EPSILON < min_delta {
        reasons.push(format!(
            "mean improvement {delta:.6} is below required {min_delta:.6}"
        ));
    }
    if improved_checks == 0 {
        reasons.push("candidate must turn at least one failed check into a pass".to_owned());
    }
    reasons.sort_unstable();
    reasons.dedup();
    regressions.sort_unstable();
    regressions.dedup();

    GateDecision {
        accepted: reasons.is_empty() && regressions.is_empty(),
        baseline_mean,
        candidate_mean,
        delta,
        improved_checks,
        regressions,
        reasons,
    }
}

fn validate_scores(scores: &[TaskScore], label: &str, reasons: &mut Vec<String>) {
    for score in scores {
        validate_task_score(score, label, reasons);
    }
}

fn derived_mean(scores: &[TaskScore]) -> f64 {
    let mut ordered = scores.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| {
        left.task_id
            .cmp(&right.task_id)
            .then_with(|| derived_score(left).total_cmp(&derived_score(right)))
            .then_with(|| left.score.total_cmp(&right.score))
    });
    let mut mean = 0.0;
    for (index, score) in ordered.into_iter().enumerate() {
        let derived = derived_score(score);
        mean += (derived - mean) / (index + 1) as f64;
    }
    mean
}

fn validate_task_score(score: &TaskScore, label: &str, reasons: &mut Vec<String>) {
    let actual_passed = score
        .check_results
        .iter()
        .filter(|result| result.passed)
        .count();
    let mut invalid = |detail: &str| {
        reasons.push(format!(
            "{label} task {} is invalid: {detail}",
            diagnostic_id(&score.task_id)
        ));
    };

    if !score.score.is_finite() || !(0.0..=1.0).contains(&score.score) {
        invalid("score must be finite and between 0 and 1");
    }
    if score.total_checks == 0 {
        invalid("total_checks must be nonzero");
    }
    if score.total_checks != score.check_results.len() {
        invalid("total_checks does not match check results");
    }
    if score.passed_checks != actual_passed {
        invalid("passed_checks does not match check results");
    }
    if score
        .check_results
        .iter()
        .any(|result| validate_check(&result.check).is_err())
    {
        invalid("contains an invalid check");
    }
    let expected = derived_score(score);
    if !score.score.is_finite() || (score.score - expected).abs() > SCORE_VALIDATION_EPSILON {
        invalid("score does not match deterministic check results");
    }
}

fn derived_score(score: &TaskScore) -> f64 {
    if !score.invariants_passed || score.check_results.is_empty() {
        return 0.0;
    }
    let passed = score
        .check_results
        .iter()
        .filter(|result| result.passed)
        .count();
    passed as f64 / score.check_results.len() as f64
}

fn scores_by_id<'a>(
    scores: &'a [TaskScore],
    label: &str,
    reasons: &mut Vec<String>,
) -> BTreeMap<&'a str, &'a TaskScore> {
    let mut by_id = BTreeMap::new();
    for score in scores {
        if by_id.insert(score.task_id.as_str(), score).is_some() {
            reasons.push(format!(
                "duplicate {label} task ID: {}",
                diagnostic_id(&score.task_id)
            ));
        }
    }
    by_id
}

#[cfg(test)]
mod tests {
    use super::{gate, score_trajectory, split_tasks, validate_check};
    use crate::types::{
        CheckResult, CheckSpec, GateDecision, MinedTask, ModelRole, ReviewStatus, TaskScore,
        Trajectory, SCHEMA_VERSION,
    };
    use chrono::Utc;
    use std::collections::BTreeMap;

    fn task(id: &str, checks: Vec<CheckSpec>) -> MinedTask {
        let now = Utc::now();
        MinedTask {
            id: id.into(),
            title: id.into(),
            prompt: "prompt".into(),
            source_session_ids: vec!["session".into()],
            source_occurrences: BTreeMap::from([("session".into(), 1)]),
            frequency: 1,
            status: ReviewStatus::Approved,
            checks,
            rubric: None,
            review_note: None,
            reviewed_at: Some(now),
            first_seen_at: now,
            last_seen_at: now,
        }
    }

    fn trajectory(task_id: &str, final_text: Option<&str>) -> Trajectory {
        Trajectory {
            schema_version: SCHEMA_VERSION,
            id: format!("trajectory-{task_id}"),
            role: ModelRole::Replay,
            task_id: Some(task_id.into()),
            started_at: Utc::now(),
            duration_ms: 1,
            prompt_hash: "prompt-hash".into(),
            skill_hash: "skill-hash".into(),
            model: None,
            process_success: true,
            exit_code: Some(0),
            timed_out: false,
            response_nonempty: true,
            final_text: final_text.map(str::to_owned),
            events: Vec::new(),
            stderr: String::new(),
            error: None,
        }
    }

    fn result(check: CheckSpec, passed: bool) -> CheckResult {
        CheckResult {
            check,
            passed,
            detail: String::new(),
        }
    }

    fn task_score(
        task_id: &str,
        score: f64,
        invariants_passed: bool,
        checks: Vec<CheckResult>,
    ) -> TaskScore {
        TaskScore {
            task_id: task_id.into(),
            passed_checks: checks.iter().filter(|result| result.passed).count(),
            total_checks: checks.len(),
            score,
            invariants_passed,
            check_results: checks,
            reasons: Vec::new(),
        }
    }

    #[test]
    fn validates_nonempty_text_checks_and_compilable_regexes() {
        for check in [
            CheckSpec::Exact { value: "ok".into() },
            CheckSpec::Contains {
                value: "ok".into(),
                case_sensitive: true,
            },
            CheckSpec::NotContains {
                value: "bad".into(),
                case_sensitive: false,
            },
            CheckSpec::Regex {
                pattern: r"^ok$".into(),
            },
        ] {
            validate_check(&check).unwrap();
        }

        for check in [
            CheckSpec::Exact {
                value: " \n".into(),
            },
            CheckSpec::Contains {
                value: String::new(),
                case_sensitive: true,
            },
            CheckSpec::NotContains {
                value: "\t".into(),
                case_sensitive: false,
            },
            CheckSpec::Regex {
                pattern: String::new(),
            },
            CheckSpec::Regex {
                pattern: "[".into(),
            },
        ] {
            assert!(validate_check(&check).is_err(), "accepted {check:?}");
        }
    }

    #[test]
    fn exact_trims_only_response_edges() {
        let exact = CheckSpec::Exact {
            value: "expected".into(),
        };
        let scored = score_trajectory(
            &task("task", vec![exact]),
            &trajectory("task", Some(" \nexpected\t")),
        );
        assert_eq!(scored.score, 1.0);

        let spaced_expected = CheckSpec::Exact {
            value: " expected ".into(),
        };
        let scored = score_trajectory(
            &task("task", vec![spaced_expected]),
            &trajectory("task", Some(" expected ")),
        );
        assert_eq!(scored.score, 0.0, "the expected value must not be trimmed");
    }

    #[test]
    fn contains_and_not_contains_respect_case_sensitivity() {
        let checks = vec![
            CheckSpec::Contains {
                value: "Needle".into(),
                case_sensitive: true,
            },
            CheckSpec::Contains {
                value: "Needle".into(),
                case_sensitive: false,
            },
            CheckSpec::NotContains {
                value: "NEEDLE".into(),
                case_sensitive: true,
            },
            CheckSpec::NotContains {
                value: "NEEDLE".into(),
                case_sensitive: false,
            },
        ];
        let scored = score_trajectory(
            &task("task", checks),
            &trajectory("task", Some("a needle appears")),
        );
        assert_eq!(
            scored
                .check_results
                .iter()
                .map(|r| r.passed)
                .collect::<Vec<_>>(),
            vec![false, true, true, false]
        );
        assert_eq!(scored.score, 0.5);
    }

    #[test]
    fn regex_checks_use_rust_regex_semantics() {
        let checks = vec![
            CheckSpec::Regex {
                pattern: r"(?m)^done:\s+\d+$".into(),
            },
            CheckSpec::Regex {
                pattern: r"^missing$".into(),
            },
        ];
        let scored = score_trajectory(
            &task("task", checks),
            &trajectory("task", Some("status\ndone: 42")),
        );
        assert_eq!(scored.passed_checks, 1);
        assert_eq!(scored.score, 0.5);
    }

    #[test]
    fn invalid_checks_produce_bounded_reasons() {
        let pattern = format!("{}[", "x".repeat(1_000));
        let scored = score_trajectory(
            &task("task", vec![CheckSpec::Regex { pattern }]),
            &trajectory("task", Some("response")),
        );
        assert_eq!(scored.score, 0.0);
        assert!(!scored.check_results[0].passed);
        assert!(scored.reasons.iter().all(|reason| reason.len() <= 160));
    }

    #[test]
    fn caseless_checks_use_full_default_unicode_folding() {
        let checks = vec![
            CheckSpec::Contains {
                value: "STRASSE".into(),
                case_sensitive: false,
            },
            CheckSpec::Contains {
                value: "οσ".into(),
                case_sensitive: false,
            },
            CheckSpec::NotContains {
                value: "STRASSE".into(),
                case_sensitive: false,
            },
            CheckSpec::NotContains {
                value: "Ος".into(),
                case_sensitive: false,
            },
        ];
        let scored = score_trajectory(
            &task("task", checks),
            &trajectory("task", Some("Straße ΟΣ")),
        );
        assert_eq!(
            scored
                .check_results
                .iter()
                .map(|result| result.passed)
                .collect::<Vec<_>>(),
            vec![true, true, false, false]
        );
    }

    #[test]
    fn scoring_requires_matching_task_id_and_replay_role() {
        let check = CheckSpec::Exact {
            value: "done".into(),
        };
        let mut missing_id = trajectory("task", Some("done"));
        missing_id.task_id = None;
        let mut wrong_id = trajectory("task", Some("done"));
        wrong_id.task_id = Some("other-task".into());
        let mut judge = trajectory("task", Some("done"));
        judge.role = ModelRole::Judge;

        for invalid in [missing_id, wrong_id, judge] {
            let scored = score_trajectory(&task("task", vec![check.clone()]), &invalid);
            assert!(!scored.invariants_passed);
            assert_eq!(scored.score, 0.0);
            assert_eq!(scored.passed_checks, 1);
            assert!(scored.reasons.iter().all(|reason| reason.len() <= 160));
            assert!(scored
                .reasons
                .iter()
                .any(|reason| reason.contains("validation replay")));
            assert!(scored
                .reasons
                .iter()
                .all(|reason| !reason.contains("other-task")));
        }
    }

    #[test]
    fn invalid_check_invalidates_an_otherwise_passing_suite() {
        let checks = vec![
            CheckSpec::Exact {
                value: "done".into(),
            },
            CheckSpec::Regex {
                pattern: "[".into(),
            },
        ];
        let scored = score_trajectory(&task("task", checks), &trajectory("task", Some("done")));
        assert!(!scored.invariants_passed);
        assert_eq!(scored.passed_checks, 1);
        assert_eq!(scored.total_checks, 2);
        assert_eq!(scored.score, 0.0);
    }

    #[test]
    fn invariant_failures_force_zero_but_preserve_textual_results() {
        let check = CheckSpec::Contains {
            value: "done".into(),
            case_sensitive: true,
        };
        let mut variants = Vec::new();

        let mut failed_process = trajectory("task", Some("done"));
        failed_process.process_success = false;
        variants.push(failed_process);

        let mut timed_out = trajectory("task", Some("done"));
        timed_out.timed_out = true;
        variants.push(timed_out);

        let mut marked_empty = trajectory("task", Some("done"));
        marked_empty.response_nonempty = false;
        variants.push(marked_empty);

        variants.push(trajectory("task", None));
        variants.push(trajectory("task", Some(" \n\t")));

        let mut reported_error = trajectory("task", Some("done"));
        reported_error.error = Some("runner failed".into());
        variants.push(reported_error);

        for trajectory in variants {
            let scored = score_trajectory(&task("task", vec![check.clone()]), &trajectory);
            assert!(!scored.invariants_passed);
            assert_eq!(scored.score, 0.0);
            assert_eq!(
                scored.passed_checks,
                usize::from(trajectory.final_text.as_deref() == Some("done"))
            );
            assert!(!scored.reasons.is_empty());
            assert!(scored.check_results[0].detail.len() <= 160);
            assert!(!scored.check_results[0].detail.contains("runner failed"));
        }
    }

    #[test]
    fn empty_check_list_has_finite_zero_score() {
        let scored = score_trajectory(&task("task", Vec::new()), &trajectory("task", Some("done")));
        assert_eq!(scored.score, 0.0);
        assert!(scored.score.is_finite());
    }

    #[test]
    fn split_is_stable_under_input_reordering_and_returns_sorted_ids() {
        let tasks = (0..10)
            .map(|index| task(&format!("task-{index}"), vec![]))
            .collect::<Vec<_>>();
        let forward = tasks.iter().collect::<Vec<_>>();
        let reverse = tasks.iter().rev().collect::<Vec<_>>();

        let first = split_tasks(&forward, 0.3, 2).unwrap();
        let second = split_tasks(&reverse, 0.3, 2).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.validation_ids.len(), 3);
        assert_eq!(first.train_ids.len(), 7);
        assert!(first.train_ids.windows(2).all(|ids| ids[0] < ids[1]));
        assert!(first.validation_ids.windows(2).all(|ids| ids[0] < ids[1]));
    }

    #[test]
    fn split_enforces_population_ratio_minimum_and_unique_ids() {
        let four = (0..4)
            .map(|index| task(&format!("task-{index}"), vec![]))
            .collect::<Vec<_>>();
        assert!(split_tasks(&four.iter().collect::<Vec<_>>(), 0.4, 2).is_err());

        let five = (0..5)
            .map(|index| task(&format!("task-{index}"), vec![]))
            .collect::<Vec<_>>();
        let refs = five.iter().collect::<Vec<_>>();
        for ratio in [f64::NAN, f64::INFINITY, 0.0, 1.0, -0.1, 1.1] {
            assert!(
                split_tasks(&refs, ratio, 2).is_err(),
                "accepted ratio {ratio}"
            );
        }
        assert!(split_tasks(&refs, 0.4, 0).is_err());
        assert!(split_tasks(&refs, 0.4, 1).is_err());

        let duplicates = vec![&five[0], &five[1], &five[2], &five[3], &five[0]];
        assert!(split_tasks(&duplicates, 0.4, 2).is_err());

        let capped = split_tasks(&refs, 0.9, 4).unwrap();
        assert_eq!(capped.train_ids.len(), 3);
        assert_eq!(capped.validation_ids.len(), 2);
    }

    #[test]
    fn gate_accepts_mean_improvement_without_regressions_and_with_new_pass() {
        let a = CheckSpec::Contains {
            value: "a".into(),
            case_sensitive: true,
        };
        let b = CheckSpec::Contains {
            value: "b".into(),
            case_sensitive: true,
        };
        let baseline = vec![
            task_score(
                "one",
                0.5,
                true,
                vec![result(a.clone(), true), result(b.clone(), false)],
            ),
            task_score("two", 0.0, true, vec![result(a.clone(), false)]),
        ];
        let candidate = vec![
            task_score("two", 1.0, true, vec![result(a, true)]),
            task_score(
                "one",
                0.5,
                true,
                vec![
                    result(
                        CheckSpec::Contains {
                            value: "a".into(),
                            case_sensitive: true,
                        },
                        true,
                    ),
                    result(b, false),
                ],
            ),
        ];

        let decision = gate(&baseline, &candidate, 0.4);
        assert!(decision.accepted, "{decision:?}");
        assert_eq!(decision.baseline_mean, 0.25);
        assert_eq!(decision.candidate_mean, 0.75);
        assert_eq!(decision.delta, 0.5);
        assert_eq!(decision.improved_checks, 1);
        assert!(decision.regressions.is_empty());
    }

    #[test]
    fn gate_is_stable_under_input_reordering() {
        let check = CheckSpec::Exact { value: "ok".into() };
        let values = [
            0.4504710074760754,
            0.8485259732348974,
            0.8574756383970945,
            0.7079094028219656,
            0.8362122933816198,
            0.5635989362674384,
            0.38989360681055185,
            0.9768932615471252,
            0.6097709006412119,
            0.17849935167292474,
        ];
        let baseline = values
            .iter()
            .enumerate()
            .map(|(index, score)| {
                task_score(
                    &format!("task-{index}"),
                    *score,
                    true,
                    vec![result(check.clone(), false)],
                )
            })
            .collect::<Vec<_>>();
        let candidate = values
            .iter()
            .enumerate()
            .map(|(index, score)| {
                task_score(
                    &format!("task-{index}"),
                    (score + 0.01).min(1.0),
                    true,
                    vec![result(check.clone(), true)],
                )
            })
            .collect::<Vec<_>>();
        let forward = gate(&baseline, &candidate, 0.0);
        let reverse = gate(
            &baseline.iter().cloned().rev().collect::<Vec<_>>(),
            &candidate.iter().cloned().rev().collect::<Vec<_>>(),
            0.0,
        );
        assert_eq!(forward, reverse);
    }

    #[test]
    fn gate_rejects_mean_delta_miss_and_requires_a_new_check_pass() {
        let checks = (0..10)
            .map(|index| CheckSpec::Exact {
                value: index.to_string(),
            })
            .collect::<Vec<_>>();
        let baseline_results = checks
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, check)| result(check, index < 5))
            .collect::<Vec<_>>();
        let candidate_results = checks
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, check)| result(check, index < 6))
            .collect::<Vec<_>>();
        let miss = gate(
            &[task_score("task", 0.5, true, baseline_results)],
            &[task_score("task", 0.6, true, candidate_results)],
            0.2,
        );
        assert!(!miss.accepted);
        assert!(miss
            .reasons
            .iter()
            .any(|reason| reason.contains("below required")));

        let a = CheckSpec::Exact { value: "a".into() };
        let b = CheckSpec::Exact { value: "b".into() };
        let unchanged = vec![result(a, true), result(b, false)];
        let no_new_pass = gate(
            &[task_score("task", 0.5, true, unchanged.clone())],
            &[task_score("task", 0.5, true, unchanged)],
            0.0,
        );
        assert!(!no_new_pass.accepted);
        assert_eq!(no_new_pass.improved_checks, 0);
        assert!(no_new_pass
            .reasons
            .iter()
            .any(|reason| reason.contains("failed check")));
    }

    #[test]
    fn gate_rejects_task_invariant_and_check_regressions() {
        let a = CheckSpec::Exact { value: "a".into() };
        let b = CheckSpec::Exact { value: "b".into() };

        let score_checks = (0..10)
            .map(|index| CheckSpec::Exact {
                value: index.to_string(),
            })
            .collect::<Vec<_>>();
        let baseline_results = score_checks
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, check)| result(check, index < 8))
            .collect::<Vec<_>>();
        let candidate_results = score_checks
            .into_iter()
            .enumerate()
            .map(|(index, check)| result(check, index < 6 || index == 8))
            .collect::<Vec<_>>();
        let score_regression = gate(
            &[task_score("task", 0.8, true, baseline_results)],
            &[task_score("task", 0.7, true, candidate_results)],
            0.0,
        );
        assert!(!score_regression.accepted);
        assert!(score_regression
            .regressions
            .contains(&"task task regressed: score 0.800000 -> 0.700000".to_owned()));
        assert!(score_regression
            .reasons
            .contains(&"task task regressed: score 0.800000 -> 0.700000".to_owned()));

        let invariant_regression = gate(
            &[task_score(
                "task",
                0.0,
                true,
                vec![result(a.clone(), false)],
            )],
            &[task_score(
                "task",
                0.0,
                false,
                vec![result(a.clone(), true)],
            )],
            0.0,
        );
        assert!(!invariant_regression.accepted);
        assert!(invariant_regression
            .regressions
            .iter()
            .any(|reason| reason.contains("invariants regressed")));

        let check_regression = gate(
            &[task_score(
                "task",
                0.5,
                true,
                vec![result(a.clone(), true), result(b.clone(), false)],
            )],
            &[task_score(
                "task",
                0.5,
                true,
                vec![result(a, false), result(b, true)],
            )],
            0.0,
        );
        assert!(!check_regression.accepted);
        assert_eq!(check_regression.improved_checks, 1);
        assert!(check_regression
            .regressions
            .iter()
            .any(|reason| reason.contains("check 0 regressed")));
    }

    #[test]
    fn gate_requires_successful_baseline_and_candidate_replays() {
        let check = CheckSpec::Exact { value: "ok".into() };
        let baseline = vec![
            task_score("recovered", 0.0, false, vec![result(check.clone(), false)]),
            task_score(
                "still-failed",
                0.0,
                false,
                vec![result(check.clone(), false)],
            ),
            task_score("improved", 0.0, true, vec![result(check.clone(), false)]),
        ];
        let candidate = vec![
            task_score("recovered", 1.0, true, vec![result(check.clone(), true)]),
            task_score(
                "still-failed",
                0.0,
                false,
                vec![result(check.clone(), false)],
            ),
            task_score("improved", 1.0, true, vec![result(check, true)]),
        ];

        let decision = gate(&baseline, &candidate, 0.5);
        assert!(!decision.accepted, "{decision:?}");
        assert!(decision
            .reasons
            .contains(&"baseline task recovered invariants failed".to_owned()));
        assert!(decision
            .reasons
            .contains(&"baseline task still-failed invariants failed".to_owned()));
        assert!(decision
            .reasons
            .contains(&"candidate task still-failed invariants failed".to_owned()));
        assert!((decision.delta - 2.0 / 3.0).abs() <= f64::EPSILON);
        assert_eq!(decision.improved_checks, 2);
    }

    #[test]
    fn gate_derives_scores_and_rejects_inconsistent_serialized_scores() {
        let check = CheckSpec::Exact { value: "ok".into() };
        let baseline = vec![
            task_score("forged", 0.0, true, vec![result(check.clone(), false)]),
            task_score("genuine", 0.0, true, vec![result(check.clone(), false)]),
        ];
        let genuine = task_score("genuine", 1.0, true, vec![result(check.clone(), true)]);
        let mut variants = Vec::new();

        variants.push(task_score(
            "forged",
            1.0,
            true,
            vec![result(check.clone(), false)],
        ));

        let mut wrong_total = task_score("forged", 0.0, true, vec![result(check.clone(), false)]);
        wrong_total.total_checks = 2;
        variants.push(wrong_total);

        let mut wrong_passed = task_score("forged", 0.0, true, vec![result(check.clone(), false)]);
        wrong_passed.passed_checks = 1;
        variants.push(wrong_passed);

        variants.push(task_score(
            "forged",
            0.0,
            true,
            vec![result(
                CheckSpec::Exact {
                    value: String::new(),
                },
                false,
            )],
        ));

        variants.push(task_score("forged", 1.0, false, vec![result(check, true)]));

        for forged in variants {
            let decision = gate(&baseline, &[forged, genuine.clone()], 0.75);
            assert!(!decision.accepted, "{decision:?}");
            assert!(decision
                .reasons
                .iter()
                .any(|reason| reason.contains("candidate task forged is invalid")));
        }

        let inflated = task_score(
            "forged",
            1.0,
            true,
            vec![result(CheckSpec::Exact { value: "ok".into() }, false)],
        );
        let decision = gate(&baseline, &[inflated, genuine], 0.75);
        assert_eq!(decision.candidate_mean, 0.5);
        assert!(!decision.accepted);
    }

    #[test]
    fn gate_rejects_reordered_check_sequences() {
        let a = CheckSpec::Exact { value: "a".into() };
        let b = CheckSpec::Exact { value: "b".into() };
        let baseline = vec![task_score(
            "task",
            0.5,
            true,
            vec![result(a.clone(), true), result(b.clone(), false)],
        )];
        let candidate = vec![task_score(
            "task",
            1.0,
            true,
            vec![result(b, true), result(a, true)],
        )];
        let decision = gate(&baseline, &candidate, 0.0);
        assert!(!decision.accepted);
        assert!(decision
            .reasons
            .contains(&"task task check results are not aligned".to_owned()));
    }

    #[test]
    fn gate_rejects_empty_mismatched_duplicate_or_misaligned_inputs() {
        let check = CheckSpec::Exact { value: "ok".into() };
        let empty = gate(&[], &[], 0.0);
        assert!(!empty.accepted);
        assert!(empty.baseline_mean.is_finite());
        assert!(empty.candidate_mean.is_finite());

        let baseline = vec![task_score(
            "one",
            0.0,
            true,
            vec![result(check.clone(), false)],
        )];
        let mismatch = gate(
            &baseline,
            &[task_score(
                "two",
                1.0,
                true,
                vec![result(check.clone(), true)],
            )],
            0.0,
        );
        assert!(!mismatch.accepted);
        assert!(mismatch
            .reasons
            .iter()
            .any(|reason| reason.contains("ID sets differ")));

        let duplicate = gate(
            &[baseline[0].clone(), baseline[0].clone()],
            &[task_score(
                "one",
                1.0,
                true,
                vec![result(check.clone(), true)],
            )],
            0.0,
        );
        assert!(!duplicate.accepted);
        assert!(duplicate
            .reasons
            .iter()
            .any(|reason| reason.contains("duplicate baseline task ID")));

        let misaligned = gate(
            &baseline,
            &[task_score(
                "one",
                1.0,
                true,
                vec![result(
                    CheckSpec::Exact {
                        value: "different".into(),
                    },
                    true,
                )],
            )],
            0.0,
        );
        assert!(!misaligned.accepted);
        assert!(misaligned
            .reasons
            .iter()
            .any(|reason| reason.contains("not aligned")));
    }

    #[test]
    fn externally_supplied_task_ids_are_escaped_and_bounded_in_diagnostics() {
        let long_id = format!("unsafe\n{}", "x".repeat(1_000));
        let duplicate_tasks = vec![
            task(&long_id, vec![]),
            task("two", vec![]),
            task("three", vec![]),
            task("four", vec![]),
            task(&long_id, vec![]),
        ];
        let split_error = split_tasks(&duplicate_tasks.iter().collect::<Vec<_>>(), 0.4, 2)
            .unwrap_err()
            .to_string();
        assert!(split_error.len() <= 160, "{split_error:?}");
        assert!(!split_error.contains('\n'), "{split_error:?}");

        let check = CheckSpec::Exact { value: "ok".into() };
        let invalid_gate = gate(
            &[
                task_score(&long_id, 0.0, true, vec![result(check.clone(), false)]),
                task_score(&long_id, 0.0, true, vec![result(check.clone(), false)]),
            ],
            &[task_score(&long_id, 1.0, true, vec![result(check, true)])],
            0.0,
        );
        assert!(invalid_gate
            .reasons
            .iter()
            .all(|reason| reason.len() <= 160 && !reason.contains('\n')));
        assert!(invalid_gate
            .regressions
            .iter()
            .all(|reason| reason.len() <= 160 && !reason.contains('\n')));
    }

    #[test]
    fn gate_rejects_nonfinite_or_negative_thresholds_and_scores() {
        let check = CheckSpec::Exact { value: "ok".into() };
        let baseline = vec![task_score(
            "task",
            0.0,
            true,
            vec![result(check.clone(), false)],
        )];
        let candidate = vec![task_score("task", 1.0, true, vec![result(check, true)])];
        for min_delta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
            let decision = gate(&baseline, &candidate, min_delta);
            assert!(!decision.accepted, "accepted min_delta {min_delta}");
            assert!(decision.baseline_mean.is_finite());
            assert!(decision.candidate_mean.is_finite());
            assert!(decision.delta.is_finite());
        }

        let nonfinite = gate(
            &[task_score("task", f64::NAN, true, Vec::new())],
            &[task_score("task", 1.0, true, Vec::new())],
            0.0,
        );
        assert!(!nonfinite.accepted);
        assert!(nonfinite.baseline_mean.is_finite());
    }

    #[test]
    fn gate_decision_shape_remains_serializable_and_explicit() {
        let decision = GateDecision {
            accepted: false,
            baseline_mean: 0.0,
            candidate_mean: 0.0,
            delta: 0.0,
            improved_checks: 0,
            regressions: Vec::new(),
            reasons: Vec::new(),
        };
        assert!(!decision.accepted);
    }
}
