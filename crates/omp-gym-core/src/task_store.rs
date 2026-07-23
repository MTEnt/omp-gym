use crate::evaluation::validate_check;
use crate::paths::atomic_write_json;
use crate::types::{CheckSpec, MinedTask, ReviewStatus, TasksFile, SCHEMA_VERSION};
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

const FUZZY_MATCH_THRESHOLD: f64 = 0.70;

static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bhttps?://\S+").expect("valid URL regex"));
static PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?m)(^|[^\p{L}\p{N}_./\\~-])(?:~/|/|[A-Za-z]:[\\/])[\p{L}\p{N}_./\\~-]+"#)
        .expect("valid path regex")
});
static LONG_HEX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[0-9a-f]{7,}\b").expect("valid identifier regex"));
static WHITESPACE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("valid whitespace regex"));

pub(crate) fn normalize_prompt(prompt: &str) -> String {
    let lower = prompt.to_lowercase();
    let normalized = URL_RE.replace_all(&lower, "<url>");
    let normalized = PATH_RE.replace_all(&normalized, "${1}<path>");
    let normalized = LONG_HEX_RE.replace_all(&normalized, "<id>");
    WHITESPACE_RE
        .replace_all(&normalized, " ")
        .trim()
        .to_owned()
}

pub(crate) fn significant_tokens(normalized: &str) -> Vec<String> {
    const STOP_WORDS: [&str; 13] = [
        "the", "a", "an", "to", "of", "and", "in", "for", "on", "please", "kindly", "can", "you",
    ];

    normalized
        .split_whitespace()
        .map(|token| token.trim_matches(|character: char| !character.is_alphanumeric()))
        .filter(|token| token.len() > 2 && !STOP_WORDS.contains(token))
        .take(24)
        .map(str::to_owned)
        .collect()
}

fn prompt_word(token: &str) -> &str {
    token.trim_matches(|character: char| {
        !character.is_alphanumeric() && character != '\'' && character != '’'
    })
}

fn is_negator(token: &str) -> bool {
    let token = prompt_word(token);
    matches!(
        token,
        "not"
            | "no"
            | "never"
            | "without"
            | "cannot"
            | "dont"
            | "wont"
            | "cant"
            | "doesnt"
            | "didnt"
            | "isnt"
            | "arent"
            | "wasnt"
            | "werent"
            | "shouldnt"
            | "wouldnt"
            | "couldnt"
            | "mustnt"
            | "neednt"
            | "havent"
            | "hasnt"
            | "hadnt"
    ) || token.ends_with("n't")
        || token.ends_with("n’t")
}

fn is_auxiliary(token: &str) -> bool {
    matches!(
        prompt_word(token),
        "do" | "does"
            | "did"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "can"
            | "could"
            | "may"
            | "might"
            | "must"
            | "shall"
            | "should"
            | "will"
            | "would"
    )
}

#[derive(Debug, Clone)]
pub(crate) struct PromptSignature {
    pub(crate) normalized: String,
    pub(crate) tokens: HashSet<String>,
    pub(crate) action_anchor: Option<String>,
    pub(crate) negated: bool,
}

pub(crate) fn prompt_signature(prompt: &str) -> PromptSignature {
    let normalized = normalize_prompt(prompt);
    let sequence = significant_tokens(&normalized);
    let negated = normalized.split_whitespace().any(is_negator);
    let action_anchor = sequence
        .iter()
        .find(|token| !is_negator(token) && !is_auxiliary(token))
        .cloned();
    PromptSignature {
        normalized,
        tokens: sequence.iter().cloned().collect(),
        action_anchor,
        negated,
    }
}

pub(crate) fn jaccard_similarity(left: &HashSet<String>, right: &HashSet<String>) -> f64 {
    let intersection = left.intersection(right).count();
    let union = left.len() + right.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

pub fn stable_task_id(prompt: &str) -> String {
    let digest = Sha256::digest(normalize_prompt(prompt).as_bytes());
    let hex = format!("{digest:x}");
    format!("task-{}", &hex[..24])
}

#[derive(Debug, Deserialize)]
struct TasksFileV1 {
    generated_at: DateTime<Utc>,
    project: PathBuf,
    reviewed: bool,
    tasks: Vec<MinedTaskV1>,
}

#[derive(Debug, Deserialize)]
struct MinedTaskV1 {
    id: String,
    title: String,
    prompt: String,
    source_session_ids: Vec<String>,
    frequency: usize,
    reviewed: bool,
}

fn canonical_project(project: &Path) -> Result<PathBuf> {
    project
        .canonicalize()
        .with_context(|| format!("canonicalize project {}", project.display()))
}

fn ensure_matching_project(stored: &Path, expected: &Path, source: &Path) -> Result<()> {
    let stored = canonical_project(stored).with_context(|| {
        format!(
            "validate task-store project {} from {}",
            stored.display(),
            source.display()
        )
    })?;
    if stored != expected {
        bail!(
            "task store project mismatch in {}: expected {}, found {}",
            source.display(),
            expected.display(),
            stored.display()
        );
    }
    Ok(())
}

fn occurrence_sum(occurrences: &BTreeMap<String, usize>) -> usize {
    occurrences
        .values()
        .fold(0usize, |total, count| total.saturating_add(*count))
}

fn normalize_task_sources(task: &mut MinedTask) {
    task.source_session_ids.sort();
    task.source_session_ids.dedup();
    if !task.source_occurrences.is_empty() {
        task.source_session_ids = task.source_occurrences.keys().cloned().collect();
        task.frequency = occurrence_sum(&task.source_occurrences);
    }
}

fn validate_task_timestamps(task: &MinedTask) -> Result<()> {
    if task.first_seen_at > task.last_seen_at {
        bail!(
            "task {} has invalid timestamp order: first_seen_at is after last_seen_at",
            task.id
        );
    }
    Ok(())
}

pub fn load_tasks(path: &Path, project: &Path) -> Result<TasksFile> {
    let project = canonical_project(project)?;
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(TasksFile {
                schema_version: SCHEMA_VERSION,
                generated_at: Utc::now(),
                project,
                tasks: vec![],
            });
        }
        Err(error) => {
            return Err(error).with_context(|| format!("read tasks file {}", path.display()));
        }
    };
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse tasks JSON {}", path.display()))?;

    if let Some(version) = value.get("schema_version") {
        let version = version.as_u64().with_context(|| {
            format!(
                "invalid tasks schema version in {}: expected an integer",
                path.display()
            )
        })?;
        if version != u64::from(SCHEMA_VERSION) {
            bail!(
                "unsupported tasks schema version {version} in {}; expected {}",
                path.display(),
                SCHEMA_VERSION
            );
        }
        let mut file: TasksFile = serde_json::from_value(value)
            .with_context(|| format!("decode v{SCHEMA_VERSION} tasks file {}", path.display()))?;
        if file.schema_version != SCHEMA_VERSION {
            bail!(
                "unsupported tasks schema version {} in {}; expected {}",
                file.schema_version,
                path.display(),
                SCHEMA_VERSION
            );
        }
        ensure_matching_project(&file.project, &project, path)?;
        file.project = project;
        for task in &mut file.tasks {
            normalize_task_sources(task);
            validate_task_timestamps(task)
                .with_context(|| format!("invalid task timestamp in {}", path.display()))?;
        }
        return Ok(file);
    }

    let legacy: TasksFileV1 = serde_json::from_value(value)
        .with_context(|| format!("decode legacy v1 tasks file {}", path.display()))?;
    ensure_matching_project(&legacy.project, &project, path)?;
    let _legacy_reviewed = legacy.reviewed;
    let tasks = legacy
        .tasks
        .into_iter()
        .map(|legacy_task| -> Result<MinedTask> {
            let _legacy_reviewed = legacy_task.reviewed;
            let mut task = MinedTask {
                id: legacy_task.id,
                title: legacy_task.title,
                prompt: legacy_task.prompt,
                source_session_ids: legacy_task.source_session_ids,
                source_occurrences: BTreeMap::new(),
                frequency: legacy_task.frequency,
                status: ReviewStatus::Pending,
                checks: vec![],
                rubric: None,
                review_note: None,
                reviewed_at: None,
                first_seen_at: legacy.generated_at,
                last_seen_at: legacy.generated_at,
            };
            normalize_task_sources(&mut task);
            validate_task_timestamps(&task)?;
            Ok(task)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TasksFile {
        schema_version: SCHEMA_VERSION,
        generated_at: legacy.generated_at,
        project,
        tasks,
    })
}

fn representative_should_change(existing: &str, incoming: &str) -> bool {
    let existing_length = existing.chars().count();
    let incoming_length = incoming.chars().count();
    incoming_length > existing_length || (incoming_length == existing_length && incoming < existing)
}

fn merge_task_sources(existing: &mut MinedTask, incoming: &MinedTask) {
    let all_sources: Vec<String> = existing
        .source_session_ids
        .iter()
        .chain(&incoming.source_session_ids)
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let existing_complete = !existing.source_occurrences.is_empty();
    let incoming_complete = !incoming.source_occurrences.is_empty();

    match (existing_complete, incoming_complete) {
        (true, true) => {
            for (source, incoming_count) in &incoming.source_occurrences {
                let existing_count = existing
                    .source_occurrences
                    .entry(source.clone())
                    .or_insert(0);
                *existing_count = (*existing_count).max(*incoming_count);
            }
        }
        (false, true)
            if existing
                .source_session_ids
                .iter()
                .all(|source| incoming.source_occurrences.contains_key(source))
                && incoming.frequency >= existing.frequency =>
        {
            existing.source_occurrences = incoming.source_occurrences.clone();
        }
        (true, false)
            if incoming
                .source_session_ids
                .iter()
                .all(|source| existing.source_occurrences.contains_key(source))
                && existing.frequency >= incoming.frequency => {}
        _ => {
            existing.source_occurrences.clear();
            existing.frequency = existing.frequency.max(incoming.frequency);
        }
    }

    if existing.source_occurrences.is_empty() {
        existing.source_session_ids = all_sources;
    } else {
        existing.source_session_ids = existing.source_occurrences.keys().cloned().collect();
        existing.frequency = occurrence_sum(&existing.source_occurrences);
    }
}

fn merge_into(existing: &mut MinedTask, incoming: MinedTask, now: DateTime<Utc>) {
    if existing.status == ReviewStatus::Pending
        && representative_should_change(&existing.prompt, &incoming.prompt)
    {
        existing.prompt = incoming.prompt.clone();
        existing.title = incoming.title.clone();
    }
    merge_task_sources(existing, &incoming);
    existing.first_seen_at = existing.first_seen_at.min(incoming.first_seen_at);
    existing.last_seen_at = existing.last_seen_at.max(incoming.last_seen_at).max(now);
}

pub(crate) fn signatures_are_compatible(left: &PromptSignature, right: &PromptSignature) -> bool {
    left.action_anchor == right.action_anchor && left.negated == right.negated
}

fn unique_fuzzy_match(existing: &[PromptSignature], incoming: &PromptSignature) -> Option<usize> {
    let mut best_index = None;
    let mut best_score = 0.0;
    let mut tied = false;

    for (index, signature) in existing.iter().enumerate() {
        if !signatures_are_compatible(signature, incoming) {
            continue;
        }
        let score = jaccard_similarity(&signature.tokens, &incoming.tokens);
        if score > best_score {
            best_index = Some(index);
            best_score = score;
            tied = false;
        } else if score == best_score && score >= FUZZY_MATCH_THRESHOLD {
            tied = true;
        }
    }

    if best_score >= FUZZY_MATCH_THRESHOLD && !tied {
        best_index
    } else {
        None
    }
}

pub fn merge_tasks(
    mut existing: Vec<MinedTask>,
    mined: Vec<MinedTask>,
    now: DateTime<Utc>,
) -> Result<Vec<MinedTask>> {
    for task in &mut existing {
        normalize_task_sources(task);
        validate_task_timestamps(task)?;
    }
    let original_existing_count = existing.len();
    let original_ids: HashSet<String> = existing.iter().map(|task| task.id.clone()).collect();

    let mut mined_with_signatures: Vec<(PromptSignature, MinedTask)> = mined
        .into_iter()
        .map(|mut task| -> Result<_> {
            normalize_task_sources(&mut task);
            validate_task_timestamps(&task)?;
            let signature = prompt_signature(&task.prompt);
            Ok((signature, task))
        })
        .collect::<Result<Vec<_>>>()?;
    mined_with_signatures.sort_by(|left, right| {
        left.0
            .normalized
            .cmp(&right.0.normalized)
            .then_with(|| left.1.id.cmp(&right.1.id))
    });
    let (exact, remaining): (Vec<_>, Vec<_>) = mined_with_signatures
        .into_iter()
        .partition(|(_, task)| original_ids.contains(&task.id));

    for (_, incoming) in exact {
        let index = existing
            .iter()
            .position(|task| task.id == incoming.id)
            .expect("exact task ID came from the existing store");
        merge_into(&mut existing[index], incoming, now);
    }
    let mut existing_signatures: Vec<PromptSignature> = existing[..original_existing_count]
        .iter()
        .map(|task| prompt_signature(&task.prompt))
        .collect();

    for (incoming_signature, mut incoming) in remaining {
        if let Some(index) = existing.iter().position(|task| task.id == incoming.id) {
            merge_into(&mut existing[index], incoming, now);
            if index < original_existing_count {
                existing_signatures[index] = prompt_signature(&existing[index].prompt);
            }
            continue;
        }
        if let Some(index) = unique_fuzzy_match(&existing_signatures, &incoming_signature) {
            merge_into(&mut existing[index], incoming, now);
            existing_signatures[index] = prompt_signature(&existing[index].prompt);
            continue;
        }

        incoming.status = ReviewStatus::Pending;
        incoming.checks.clear();
        incoming.rubric = None;
        incoming.review_note = None;
        incoming.reviewed_at = None;
        incoming.last_seen_at = incoming.last_seen_at.max(now);
        validate_task_timestamps(&incoming)?;
        existing.push(incoming);
    }

    existing.sort_by(|left, right| {
        right
            .frequency
            .cmp(&left.frequency)
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(existing)
}

pub fn save_tasks(path: &Path, file: &TasksFile) -> Result<()> {
    if file.schema_version != SCHEMA_VERSION {
        bail!(
            "cannot save tasks schema version {}; expected {}",
            file.schema_version,
            SCHEMA_VERSION
        );
    }
    let mut normalized = file.clone();
    for task in &mut normalized.tasks {
        normalize_task_sources(task);
        validate_task_timestamps(task).with_context(|| {
            format!(
                "refuse to save invalid task timestamp to {}",
                path.display()
            )
        })?;
    }
    atomic_write_json(path, &normalized)
        .with_context(|| format!("save tasks file {}", path.display()))
}

fn validate_checks(checks: &[CheckSpec]) -> Result<()> {
    if checks.is_empty() {
        bail!("approved task requires at least one check");
    }
    for check in checks {
        validate_check(check)?;
    }
    Ok(())
}

fn find_task_mut<'a>(file: &'a mut TasksFile, id: &str) -> Result<&'a mut MinedTask> {
    file.tasks
        .iter_mut()
        .find(|task| task.id == id)
        .with_context(|| format!("unknown task ID {id}"))
}

pub fn approve_task(
    file: &mut TasksFile,
    id: &str,
    checks: Vec<CheckSpec>,
    rubric: Option<String>,
    note: Option<String>,
) -> Result<()> {
    let task = find_task_mut(file, id)?;
    validate_checks(&checks).with_context(|| format!("cannot approve task {id}"))?;
    task.status = ReviewStatus::Approved;
    task.checks = checks;
    task.rubric = rubric;
    task.review_note = note;
    task.reviewed_at = Some(Utc::now());
    Ok(())
}

pub fn reject_task(file: &mut TasksFile, id: &str, note: Option<String>) -> Result<()> {
    let task = find_task_mut(file, id)?;
    task.status = ReviewStatus::Rejected;
    task.checks.clear();
    task.rubric = None;
    task.review_note = note;
    task.reviewed_at = Some(Utc::now());
    Ok(())
}

pub fn reopen_task(file: &mut TasksFile, id: &str) -> Result<()> {
    let task = find_task_mut(file, id)?;
    task.status = ReviewStatus::Pending;
    task.checks.clear();
    task.rubric = None;
    task.review_note = None;
    task.reviewed_at = None;
    Ok(())
}

pub fn validate_reviewed_tasks(file: &TasksFile) -> Result<Vec<&MinedTask>> {
    let mut reviewed = Vec::new();
    for task in &file.tasks {
        if task.status == ReviewStatus::Approved {
            validate_checks(&task.checks)
                .with_context(|| format!("approved task {} has invalid checks", task.id))?;
            reviewed.push(task);
        }
    }
    Ok(reviewed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CheckSpec, MinedTask, ReviewStatus, TasksFile, SCHEMA_VERSION};
    use chrono::{DateTime, TimeZone, Utc};
    use std::collections::BTreeSet;
    use std::path::Path;
    use tempfile::tempdir;

    fn at(hour: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 23, hour, 0, 0)
            .single()
            .expect("valid timestamp")
    }

    fn pending(id: &str, prompt: &str, sources: &[&str], frequency: usize) -> MinedTask {
        let unique_sources: Vec<String> = sources
            .iter()
            .map(|source| (*source).to_owned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let base = if unique_sources.is_empty() {
            0
        } else {
            frequency / unique_sources.len()
        };
        let remainder = if unique_sources.is_empty() {
            0
        } else {
            frequency % unique_sources.len()
        };
        let source_occurrences = unique_sources
            .iter()
            .enumerate()
            .map(|(index, source)| (source.clone(), base + usize::from(index < remainder)))
            .collect();
        MinedTask {
            id: id.into(),
            title: format!("Title for {prompt}"),
            prompt: prompt.into(),
            source_session_ids: unique_sources,
            source_occurrences,
            frequency,
            status: ReviewStatus::Pending,
            checks: vec![],
            rubric: None,
            review_note: None,
            reviewed_at: None,
            first_seen_at: at(10),
            last_seen_at: at(10),
        }
    }

    fn file(project: &Path, tasks: Vec<MinedTask>) -> TasksFile {
        TasksFile {
            schema_version: SCHEMA_VERSION,
            generated_at: at(10),
            project: project.into(),
            tasks,
        }
    }

    fn contains(value: &str) -> CheckSpec {
        CheckSpec::Contains {
            value: value.into(),
            case_sensitive: false,
        }
    }

    #[test]
    fn normalized_prompts_have_identical_stable_ids() {
        let first = "Fix LOGIN at https://example.com/a for `/Users/me/app` id ABCDEF123456";
        let second = "  fix login at https://other.test/b for `/tmp/app` id deadbeef9999  ";

        assert_eq!(stable_task_id(first), stable_task_id(second));
        let id = stable_task_id(first);
        assert!(id.starts_with("task-"));
        assert_eq!(id.len(), 29);
        assert!(id[5..]
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));
    }

    #[test]
    fn fuzzy_merge_preserves_the_entire_approved_contract() {
        let mut existing = pending(
            "task-existing",
            "Fix login failures in auth module",
            &["session-b", "session-a"],
            2,
        );
        existing.status = ReviewStatus::Approved;
        existing.checks = vec![contains("resolved")];
        existing.rubric = Some("Explain the authentication fix".into());
        existing.review_note = Some("owner reviewed".into());
        existing.reviewed_at = Some(at(11));
        existing.first_seen_at = at(9);

        let incoming_prompt = "Please fix recurring login failures in auth module today";
        let incoming = pending(
            &stable_task_id(incoming_prompt),
            incoming_prompt,
            &["session-c", "session-a"],
            2,
        );
        let merged = merge_tasks(vec![existing], vec![incoming], at(12)).expect("merge tasks");

        assert_eq!(merged.len(), 1);
        let task = &merged[0];
        assert_eq!(task.id, "task-existing");
        assert_eq!(task.status, ReviewStatus::Approved);
        assert_eq!(task.checks, vec![contains("resolved")]);
        assert_eq!(
            task.rubric.as_deref(),
            Some("Explain the authentication fix")
        );
        assert_eq!(task.review_note.as_deref(), Some("owner reviewed"));
        assert_eq!(task.reviewed_at, Some(at(11)));
        assert_eq!(task.first_seen_at, at(9));
        assert_eq!(task.last_seen_at, at(12));
        assert_eq!(task.prompt, "Fix login failures in auth module");
        assert_eq!(task.title, "Title for Fix login failures in auth module");
        assert_eq!(
            task.source_session_ids,
            ["session-a", "session-b", "session-c"]
        );
        assert_eq!(task.frequency, 3);
    }

    #[test]
    fn exact_merge_deduplicates_sources_without_inflating_frequency() {
        let prompt = "Fix the recurring login authentication failure";
        let id = stable_task_id(prompt);
        let existing = pending(&id, prompt, &["session-b", "session-a"], 2);
        let incoming = pending(&id, prompt, &["session-a", "session-b"], 2);

        let merged = merge_tasks(vec![existing], vec![incoming], at(12)).expect("merge tasks");

        assert_eq!(merged[0].source_session_ids, ["session-a", "session-b"]);
        assert_eq!(merged[0].frequency, 2);
    }

    #[test]
    fn partial_overlap_uses_per_source_max_and_repeat_harvest_is_idempotent() {
        let prompt = "Fix recurring login authentication failures";
        let id = stable_task_id(prompt);
        let mut existing = pending(&id, prompt, &["session-a", "session-b"], 5);
        existing.source_occurrences =
            BTreeMap::from([("session-a".into(), 2), ("session-b".into(), 3)]);
        let mut incoming = pending(&id, prompt, &["session-b", "session-c"], 7);
        incoming.source_occurrences =
            BTreeMap::from([("session-b".into(), 5), ("session-c".into(), 2)]);
        incoming.last_seen_at = at(13);

        let merged =
            merge_tasks(vec![existing], vec![incoming.clone()], at(12)).expect("first merge");

        assert_eq!(
            merged[0].source_occurrences,
            BTreeMap::from([
                ("session-a".into(), 2),
                ("session-b".into(), 5),
                ("session-c".into(), 2),
            ])
        );
        assert_eq!(merged[0].frequency, 9);
        assert_eq!(merged[0].last_seen_at, at(13));

        let repeated = merge_tasks(merged.clone(), vec![incoming], at(12)).expect("repeat merge");
        assert_eq!(repeated, merged);
    }

    #[test]
    fn incomplete_legacy_counts_reconcile_only_from_a_covering_snapshot() {
        let prompt = "Fix recurring login authentication failures";
        let id = stable_task_id(prompt);
        let mut legacy = pending(&id, prompt, &["session-a", "session-b"], 3);
        legacy.source_occurrences.clear();
        let covering = pending(&id, prompt, &["session-a", "session-b"], 3);

        let reconciled =
            merge_tasks(vec![legacy], vec![covering.clone()], at(12)).expect("reconcile counts");

        assert_eq!(
            reconciled[0].source_occurrences,
            covering.source_occurrences
        );
        assert_eq!(reconciled[0].frequency, 3);
        let repeated =
            merge_tasks(reconciled.clone(), vec![covering], at(12)).expect("repeat harvest");
        assert_eq!(repeated, reconciled);
    }

    #[test]
    fn partial_snapshot_does_not_inflate_incomplete_legacy_counts() {
        let prompt = "Fix recurring login authentication failures";
        let id = stable_task_id(prompt);
        let mut legacy = pending(&id, prompt, &["session-a", "session-b"], 3);
        legacy.source_occurrences.clear();
        let partial = pending(&id, prompt, &["session-b", "session-c"], 2);

        let merged = merge_tasks(vec![legacy], vec![partial], at(12)).expect("merge partial");

        assert!(merged[0].source_occurrences.is_empty());
        assert_eq!(merged[0].frequency, 3);
        assert_eq!(
            merged[0].source_session_ids,
            ["session-a", "session-b", "session-c"]
        );
    }

    #[test]
    fn rejected_tasks_stay_rejected_and_new_tasks_stay_pending() {
        let prompt = "Repair authentication failures in the login service";
        let id = stable_task_id(prompt);
        let mut rejected = pending(&id, prompt, &["old"], 1);
        rejected.status = ReviewStatus::Rejected;
        rejected.review_note = Some("out of scope".into());
        rejected.reviewed_at = Some(at(11));
        let reharvested_prompt =
            "Repair recurring authentication failures in the login service today";
        let reharvested = pending(
            &stable_task_id(reharvested_prompt),
            reharvested_prompt,
            &["new"],
            1,
        );
        let new_prompt = "Document the deployment rollback procedure";
        let new_task = pending(&stable_task_id(new_prompt), new_prompt, &["new"], 1);

        let merged =
            merge_tasks(vec![rejected], vec![reharvested, new_task], at(12)).expect("merge tasks");

        let rejected = merged
            .iter()
            .find(|task| task.id == id)
            .expect("preserved rejected task");
        assert_eq!(rejected.status, ReviewStatus::Rejected);
        assert_eq!(rejected.review_note.as_deref(), Some("out of scope"));
        assert_eq!(rejected.prompt, prompt);
        assert_eq!(rejected.title, format!("Title for {prompt}"));
        let new_task = merged
            .iter()
            .find(|task| task.prompt == new_prompt)
            .expect("new task");
        assert_eq!(new_task.status, ReviewStatus::Pending);
        assert!(new_task.checks.is_empty());
    }

    #[test]
    fn fuzzy_merge_requires_a_unique_highest_match() {
        let best = pending(
            "task-best",
            "Fix login authentication error in module",
            &["old-a"],
            1,
        );
        let lower = pending(
            "task-lower",
            "Fix login billing failure in module",
            &["old-b"],
            1,
        );
        let incoming_prompt = "Fix login authentication error in module today";
        let incoming = pending(
            &stable_task_id(incoming_prompt),
            incoming_prompt,
            &["new"],
            1,
        );

        let merged = merge_tasks(vec![best, lower], vec![incoming], at(12)).expect("merge tasks");

        assert_eq!(merged.len(), 2);
        assert!(merged
            .iter()
            .any(|task| task.id == "task-best" && task.source_session_ids.contains(&"new".into())));
    }

    #[test]
    fn tied_best_fuzzy_matches_leave_the_incoming_task_new() {
        let left = pending("task-left", "Fix login authentication error", &["old-a"], 1);
        let right = pending(
            "task-right",
            "Fix login authentication failure",
            &["old-b"],
            1,
        );
        let incoming_prompt = "Fix login authentication";
        let incoming_id = stable_task_id(incoming_prompt);
        let incoming = pending(&incoming_id, incoming_prompt, &["new"], 1);

        let merged = merge_tasks(vec![left, right], vec![incoming], at(12)).expect("merge tasks");

        assert_eq!(merged.len(), 3);
        let new_task = merged
            .iter()
            .find(|task| task.id == incoming_id)
            .expect("ambiguous incoming task stays new");
        assert_eq!(new_task.status, ReviewStatus::Pending);
    }

    #[test]
    fn incompatible_actions_and_negation_do_not_fuzzy_merge() {
        let enabled = "Enable automatic nightly cleanup for temporary build artifact files safely";
        let deleted = "Delete automatic nightly cleanup for temporary build artifact files safely";
        let negated =
            "Enable no automatic nightly cleanup for temporary build artifact files safely";

        let action_split = merge_tasks(
            vec![pending(&stable_task_id(enabled), enabled, &["old"], 1)],
            vec![pending(&stable_task_id(deleted), deleted, &["new"], 1)],
            at(12),
        )
        .expect("merge incompatible actions");
        assert_eq!(action_split.len(), 2);
        assert!(action_split
            .iter()
            .all(|task| task.status == ReviewStatus::Pending));

        let negation_split = merge_tasks(
            vec![pending(&stable_task_id(enabled), enabled, &["old"], 1)],
            vec![pending(&stable_task_id(negated), negated, &["new"], 1)],
            at(12),
        )
        .expect("merge incompatible negation");
        assert_eq!(negation_split.len(), 2);
        assert!(negation_split
            .iter()
            .all(|task| task.status == ReviewStatus::Pending));
    }

    #[test]
    fn contractions_mark_negation_and_do_not_merge_with_positive_tasks() {
        for contraction in [
            "doesn't",
            "won't",
            "didn't",
            "isn't",
            "shouldn't",
            "couldn't",
        ] {
            let signature = prompt_signature(&format!(
                "Configure cleanup so it {contraction} delete temporary files"
            ));
            assert!(signature.negated, "{contraction} must mark negation");
        }

        let positive = "Configure cleanup to delete temporary files";
        let negative = "Configure cleanup so it doesn't delete temporary files";
        let merged = merge_tasks(
            vec![pending(&stable_task_id(positive), positive, &["old"], 1)],
            vec![pending(&stable_task_id(negative), negative, &["new"], 1)],
            at(12),
        )
        .expect("merge opposite polarity");
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn equivalent_negative_phrasings_share_the_action_anchor() {
        let first =
            "Do not enable automatic nightly cleanup for temporary build artifact files safely";
        let second =
            "Never enable automatic nightly cleanup for temporary build artifact files safely";
        assert_eq!(
            prompt_signature(first).action_anchor.as_deref(),
            Some("enable")
        );
        assert_eq!(
            prompt_signature(second).action_anchor.as_deref(),
            Some("enable")
        );

        let merged = merge_tasks(
            vec![pending(&stable_task_id(first), first, &["old"], 1)],
            vec![pending(&stable_task_id(second), second, &["new"], 1)],
            at(12),
        )
        .expect("merge equivalent negative phrasings");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].frequency, 2);
    }

    #[test]
    fn polite_request_prefix_does_not_replace_the_action_anchor() {
        let plain = "Enable automatic nightly cleanup for temporary build artifact files safely";
        let polite =
            "Kindly enable automatic nightly cleanup for temporary build artifact files safely";
        assert_eq!(
            prompt_signature(polite).action_anchor.as_deref(),
            Some("enable")
        );

        let merged = merge_tasks(
            vec![pending(&stable_task_id(plain), plain, &["old"], 1)],
            vec![pending(&stable_task_id(polite), polite, &["new"], 1)],
            at(12),
        )
        .expect("merge polite equivalent");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].frequency, 2);
    }

    #[test]
    fn merge_is_deterministic_when_mined_order_changes() {
        let existing = pending("task-existing", "alpha beta gamma delta", &["old"], 1);
        let extra_prompt = "alpha beta gamma delta extra";
        let other_prompt = "alpha beta gamma delta other";
        let extra = pending(&stable_task_id(extra_prompt), extra_prompt, &["extra"], 1);
        let other = pending(&stable_task_id(other_prompt), other_prompt, &["other"], 1);

        let first = merge_tasks(
            vec![existing.clone()],
            vec![other.clone(), extra.clone()],
            at(12),
        )
        .expect("merge tasks");
        let second = merge_tasks(vec![existing], vec![extra, other], at(12)).expect("merge tasks");

        assert_eq!(first, second);
    }

    #[test]
    fn missing_file_loads_an_empty_v2_store_for_the_canonical_project() {
        let root = tempdir().expect("temporary directory");
        let project = root.path().join("project");
        std::fs::create_dir(&project).expect("create project");
        let canonical = project.canonicalize().expect("canonical project");

        let loaded = load_tasks(&root.path().join("missing.json"), &project).expect("empty store");

        assert_eq!(loaded.schema_version, SCHEMA_VERSION);
        assert_eq!(loaded.project, canonical);
        assert!(loaded.tasks.is_empty());
    }

    #[test]
    fn save_rejects_reversed_task_timestamps() {
        let root = tempdir().expect("temporary directory");
        let project = root.path().join("project");
        std::fs::create_dir(&project).expect("create project");
        let path = root.path().join("tasks.json");
        let mut task = pending("task-a", "Fix recurring authentication failures", &["s"], 1);
        task.first_seen_at = at(12);
        task.last_seen_at = at(10);

        let error = save_tasks(
            &path,
            &file(
                &project.canonicalize().expect("canonical project"),
                vec![task],
            ),
        )
        .expect_err("invalid timestamps must not save");

        assert!(error.to_string().contains("timestamp"));
        assert!(!path.exists());
    }

    #[test]
    fn load_rejects_reversed_task_timestamps() {
        let root = tempdir().expect("temporary directory");
        let project = root.path().join("project");
        std::fs::create_dir(&project).expect("create project");
        let path = root.path().join("tasks.json");
        let mut task = pending("task-a", "Fix recurring authentication failures", &["s"], 1);
        task.first_seen_at = at(12);
        task.last_seen_at = at(10);
        atomic_write_json(
            &path,
            &file(
                &project.canonicalize().expect("canonical project"),
                vec![task],
            ),
        )
        .expect("write invalid fixture");

        let error = load_tasks(&path, &project).expect_err("invalid timestamps must not load");

        assert!(error.to_string().contains("timestamp"));
    }

    #[test]
    fn explicit_v1_migration_resets_legacy_review_state() {
        let root = tempdir().expect("temporary directory");
        let project = root.path().join("project");
        std::fs::create_dir(&project).expect("create project");
        let canonical = project.canonicalize().expect("canonical project");
        let path = root.path().join("tasks.json");
        let legacy = serde_json::json!({
            "generated_at": "2026-07-23T10:00:00Z",
            "project": canonical,
            "reviewed": true,
            "tasks": [{
                "id": "legacy-id",
                "title": "Legacy",
                "prompt": "Fix the legacy login authentication failure",
                "source_session_ids": ["session-b", "session-a", "session-a"],
                "frequency": 3,
                "reviewed": true
            }]
        });
        std::fs::write(&path, serde_json::to_vec(&legacy).expect("legacy json"))
            .expect("write legacy file");

        let loaded = load_tasks(&path, &project).expect("migrate v1");

        assert_eq!(loaded.schema_version, SCHEMA_VERSION);
        assert_eq!(loaded.tasks[0].id, "legacy-id");
        assert_eq!(loaded.tasks[0].status, ReviewStatus::Pending);
        assert!(loaded.tasks[0].checks.is_empty());
        assert!(loaded.tasks[0].rubric.is_none());
        assert!(loaded.tasks[0].review_note.is_none());
        assert!(loaded.tasks[0].reviewed_at.is_none());
        assert_eq!(
            loaded.tasks[0].source_session_ids,
            ["session-a", "session-b"]
        );
        assert!(loaded.tasks[0].source_occurrences.is_empty());
        assert_eq!(loaded.tasks[0].frequency, 3);
    }

    #[test]
    fn load_rejects_project_mismatch_unsupported_version_and_corrupt_json() {
        let root = tempdir().expect("temporary directory");
        let project = root.path().join("project");
        let other = root.path().join("other");
        std::fs::create_dir(&project).expect("create project");
        std::fs::create_dir(&other).expect("create other project");
        let path = root.path().join("tasks.json");

        save_tasks(
            &path,
            &file(&other.canonicalize().expect("canonical other"), vec![]),
        )
        .expect("write mismatched store");
        assert!(load_tasks(&path, &project)
            .expect_err("project mismatch")
            .to_string()
            .contains("project"));

        std::fs::write(
            &path,
            br#"{"schema_version":99,"generated_at":"2026-07-23T10:00:00Z","project":"/tmp","tasks":[]}"#,
        )
        .expect("write unsupported version");
        assert!(load_tasks(&path, &project)
            .expect_err("unsupported version")
            .to_string()
            .contains("version"));

        std::fs::write(&path, b"{not json").expect("write corrupt json");
        let error = load_tasks(&path, &project).expect_err("corrupt JSON");
        assert!(error.to_string().contains("tasks.json"));
    }

    #[test]
    fn review_mutations_validate_contracts_and_clear_obsolete_state() {
        let root = tempdir().expect("temporary directory");
        let project = root.path().join("project");
        std::fs::create_dir(&project).expect("create project");
        let mut store = file(
            &project.canonicalize().expect("canonical project"),
            vec![pending(
                "task-a",
                "Return a concise final answer",
                &["s"],
                1,
            )],
        );

        assert!(approve_task(&mut store, "missing", vec![contains("done")], None, None).is_err());
        assert!(approve_task(&mut store, "task-a", vec![], None, None).is_err());
        for invalid_check in [
            CheckSpec::Exact { value: " ".into() },
            CheckSpec::Contains {
                value: String::new(),
                case_sensitive: false,
            },
            CheckSpec::NotContains {
                value: "\t".into(),
                case_sensitive: true,
            },
            CheckSpec::Regex {
                pattern: String::new(),
            },
            CheckSpec::Regex {
                pattern: "(".into(),
            },
        ] {
            assert!(approve_task(&mut store, "task-a", vec![invalid_check], None, None).is_err());
        }

        approve_task(
            &mut store,
            "task-a",
            vec![contains("done")],
            Some("Be precise".into()),
            Some("reviewed".into()),
        )
        .expect("approve valid task");
        assert_eq!(
            validate_reviewed_tasks(&store).expect("valid suite").len(),
            1
        );
        assert_eq!(store.tasks[0].status, ReviewStatus::Approved);
        assert!(store.tasks[0].reviewed_at.is_some());

        reject_task(&mut store, "task-a", Some("not useful".into())).expect("reject task");
        assert_eq!(store.tasks[0].status, ReviewStatus::Rejected);
        assert!(store.tasks[0].checks.is_empty());
        assert!(store.tasks[0].rubric.is_none());
        assert_eq!(store.tasks[0].review_note.as_deref(), Some("not useful"));
        assert!(store.tasks[0].reviewed_at.is_some());

        reopen_task(&mut store, "task-a").expect("reopen task");
        assert_eq!(store.tasks[0].status, ReviewStatus::Pending);
        assert!(store.tasks[0].checks.is_empty());
        assert!(store.tasks[0].rubric.is_none());
        assert!(store.tasks[0].review_note.is_none());
        assert!(store.tasks[0].reviewed_at.is_none());
        assert!(reopen_task(&mut store, "missing").is_err());
    }

    #[test]
    fn approved_tasks_with_missing_or_invalid_checks_fail_suite_validation() {
        let root = tempdir().expect("temporary directory");
        let project = root.path().join("project");
        std::fs::create_dir(&project).expect("create project");
        let canonical = project.canonicalize().expect("canonical project");
        let mut approved = pending(
            "task-approved",
            "Return the requested final answer",
            &["s"],
            1,
        );
        approved.status = ReviewStatus::Approved;
        approved.reviewed_at = Some(at(11));
        let mut store = file(&canonical, vec![approved]);

        assert!(validate_reviewed_tasks(&store).is_err());
        store.tasks[0].checks = vec![CheckSpec::Exact {
            value: "   ".into(),
        }];
        assert!(validate_reviewed_tasks(&store).is_err());
        store.tasks[0].checks = vec![CheckSpec::Regex {
            pattern: "[".into(),
        }];
        assert!(validate_reviewed_tasks(&store).is_err());
    }
}
