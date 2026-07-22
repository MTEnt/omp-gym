use crate::paths::atomic_write_json;
use crate::types::{CheckSpec, MinedTask, ReviewStatus, TasksFile, SCHEMA_VERSION};
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
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

pub(crate) fn significant_tokens(normalized: &str) -> HashSet<String> {
    const STOP_WORDS: [&str; 12] = [
        "the", "a", "an", "to", "of", "and", "in", "for", "on", "please", "can", "you",
    ];

    normalized
        .split_whitespace()
        .map(|token| token.trim_matches(|character: char| !character.is_alphanumeric()))
        .filter(|token| token.len() > 2 && !STOP_WORDS.contains(token))
        .take(24)
        .map(str::to_owned)
        .collect()
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

fn sort_and_deduplicate_sources(task: &mut MinedTask) {
    task.source_session_ids.sort();
    task.source_session_ids.dedup();
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
            sort_and_deduplicate_sources(task);
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
        .map(|legacy_task| {
            let _legacy_reviewed = legacy_task.reviewed;
            let mut task = MinedTask {
                id: legacy_task.id,
                title: legacy_task.title,
                prompt: legacy_task.prompt,
                source_session_ids: legacy_task.source_session_ids,
                frequency: legacy_task.frequency,
                status: ReviewStatus::Pending,
                checks: vec![],
                rubric: None,
                review_note: None,
                reviewed_at: None,
                first_seen_at: legacy.generated_at,
                last_seen_at: legacy.generated_at,
            };
            sort_and_deduplicate_sources(&mut task);
            task
        })
        .collect();
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

fn merge_frequency(existing: &MinedTask, incoming: &MinedTask) -> usize {
    let existing_sources: HashSet<&str> = existing
        .source_session_ids
        .iter()
        .map(String::as_str)
        .collect();
    let incoming_sources: HashSet<&str> = incoming
        .source_session_ids
        .iter()
        .map(String::as_str)
        .collect();
    if incoming_sources.is_empty() || existing_sources.is_empty() {
        return existing.frequency.max(incoming.frequency);
    }
    let overlap = existing_sources.intersection(&incoming_sources).count();
    if overlap == 0 {
        existing.frequency.saturating_add(incoming.frequency)
    } else {
        let new_sources = incoming_sources.difference(&existing_sources).count();
        existing
            .frequency
            .saturating_add(new_sources)
            .max(incoming.frequency)
    }
}

fn merge_into(existing: &mut MinedTask, incoming: MinedTask, now: DateTime<Utc>) {
    let merged_frequency = merge_frequency(existing, &incoming);
    if representative_should_change(&existing.prompt, &incoming.prompt) {
        existing.prompt = incoming.prompt.clone();
        existing.title = incoming.title.clone();
    }
    existing
        .source_session_ids
        .extend(incoming.source_session_ids);
    sort_and_deduplicate_sources(existing);
    existing.frequency = merged_frequency;
    existing.first_seen_at = existing.first_seen_at.min(incoming.first_seen_at);
    existing.last_seen_at = now;
}

fn unique_fuzzy_match(existing: &[MinedTask], incoming: &MinedTask) -> Option<usize> {
    let incoming_tokens = significant_tokens(&normalize_prompt(&incoming.prompt));
    let mut best_index = None;
    let mut best_score = 0.0;
    let mut tied = false;

    for (index, task) in existing.iter().enumerate() {
        let score = jaccard_similarity(
            &significant_tokens(&normalize_prompt(&task.prompt)),
            &incoming_tokens,
        );
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
        sort_and_deduplicate_sources(task);
    }
    let original_existing_count = existing.len();
    let original_ids: HashSet<String> = existing.iter().map(|task| task.id.clone()).collect();

    let mut mined_with_keys: Vec<(String, MinedTask)> = mined
        .into_iter()
        .map(|mut task| {
            sort_and_deduplicate_sources(&mut task);
            (normalize_prompt(&task.prompt), task)
        })
        .collect();
    mined_with_keys.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.id.cmp(&right.1.id))
    });
    let (exact, remaining): (Vec<_>, Vec<_>) = mined_with_keys
        .into_iter()
        .map(|(_, task)| task)
        .partition(|task| original_ids.contains(&task.id));

    for incoming in exact {
        let index = existing
            .iter()
            .position(|task| task.id == incoming.id)
            .expect("exact task ID came from the existing store");
        merge_into(&mut existing[index], incoming, now);
    }

    for mut incoming in remaining {
        if let Some(index) = existing.iter().position(|task| task.id == incoming.id) {
            merge_into(&mut existing[index], incoming, now);
            continue;
        }
        if let Some(index) = unique_fuzzy_match(&existing[..original_existing_count], &incoming) {
            merge_into(&mut existing[index], incoming, now);
            continue;
        }

        incoming.status = ReviewStatus::Pending;
        incoming.checks.clear();
        incoming.rubric = None;
        incoming.review_note = None;
        incoming.reviewed_at = None;
        incoming.last_seen_at = now;
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
    atomic_write_json(path, file).with_context(|| format!("save tasks file {}", path.display()))
}

fn validate_checks(checks: &[CheckSpec]) -> Result<()> {
    if checks.is_empty() {
        bail!("approved task requires at least one check");
    }
    for check in checks {
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
                Regex::new(pattern)
                    .with_context(|| format!("invalid task regex check {pattern:?}"))?;
            }
        }
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
    use std::path::Path;
    use tempfile::tempdir;

    fn at(hour: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 23, hour, 0, 0)
            .single()
            .expect("valid timestamp")
    }

    fn pending(id: &str, prompt: &str, sources: &[&str], frequency: usize) -> MinedTask {
        MinedTask {
            id: id.into(),
            title: format!("Title for {prompt}"),
            prompt: prompt.into(),
            source_session_ids: sources.iter().map(|source| (*source).into()).collect(),
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
        assert_eq!(task.prompt, incoming_prompt);
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
    fn rejected_tasks_stay_rejected_and_new_tasks_stay_pending() {
        let prompt = "Repair authentication failures in the login service";
        let id = stable_task_id(prompt);
        let mut rejected = pending(&id, prompt, &["old"], 1);
        rejected.status = ReviewStatus::Rejected;
        rejected.review_note = Some("out of scope".into());
        rejected.reviewed_at = Some(at(11));
        let reharvested = pending(&id, prompt, &["new"], 1);
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
