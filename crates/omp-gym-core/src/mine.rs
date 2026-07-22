use crate::task_store::{jaccard_similarity, normalize_prompt, significant_tokens, stable_task_id};
use crate::types::{MinedTask, ReviewStatus, SessionSummary};
use chrono::Utc;
use std::collections::{BTreeSet, HashSet};

const CLUSTER_THRESHOLD: f64 = 0.70;

/// Mine recurring / representative tasks from harvested sessions.
///
/// Candidates and output are sorted so filesystem/session enumeration cannot
/// change clustering, task identity, or ranking.
pub fn mine_tasks(sessions: &[SessionSummary], max_tasks: usize) -> Vec<MinedTask> {
    let mut candidates = Vec::new();
    for session in sessions {
        for excerpt in &session.user_excerpts {
            let normalized = normalize_prompt(excerpt);
            if normalized.len() < 12 {
                continue;
            }
            let tokens = significant_tokens(&normalized);
            if tokens.is_empty() {
                continue;
            }
            candidates.push(Candidate {
                normalized,
                prompt: excerpt.clone(),
                session_id: session.id.clone(),
                tokens,
            });
        }
    }
    candidates.sort_by(|left, right| {
        left.normalized
            .cmp(&right.normalized)
            .then_with(|| left.session_id.cmp(&right.session_id))
            .then_with(|| left.prompt.cmp(&right.prompt))
    });

    let mut clusters: Vec<Cluster> = Vec::new();
    for candidate in candidates {
        let matching_cluster = clusters.iter().position(|cluster| {
            jaccard_similarity(&cluster.tokens, &candidate.tokens) >= CLUSTER_THRESHOLD
        });
        if let Some(index) = matching_cluster {
            let cluster = &mut clusters[index];
            cluster.count += 1;
            cluster.sessions.insert(candidate.session_id);
            if representative_should_change(&cluster.prompt, &candidate.prompt) {
                cluster.title = titleize(&candidate.normalized);
                cluster.prompt = candidate.prompt;
                cluster.tokens = candidate.tokens;
            }
        } else {
            clusters.push(Cluster {
                title: titleize(&candidate.normalized),
                prompt: candidate.prompt,
                sessions: BTreeSet::from([candidate.session_id]),
                count: 1,
                tokens: candidate.tokens,
            });
        }
    }

    let now = Utc::now();
    let mut tasks: Vec<MinedTask> = clusters
        .into_iter()
        .map(|cluster| MinedTask {
            id: stable_task_id(&cluster.prompt),
            title: cluster.title,
            prompt: cluster.prompt,
            source_session_ids: cluster.sessions.into_iter().collect(),
            frequency: cluster.count,
            status: ReviewStatus::Pending,
            checks: vec![],
            rubric: None,
            review_note: None,
            reviewed_at: None,
            first_seen_at: now,
            last_seen_at: now,
        })
        .collect();
    tasks.sort_by(|left, right| {
        right
            .frequency
            .cmp(&left.frequency)
            .then_with(|| left.id.cmp(&right.id))
    });
    tasks.truncate(max_tasks);
    tasks
}

struct Candidate {
    normalized: String,
    prompt: String,
    session_id: String,
    tokens: HashSet<String>,
}

struct Cluster {
    title: String,
    prompt: String,
    sessions: BTreeSet<String>,
    count: usize,
    tokens: HashSet<String>,
}

fn representative_should_change(existing: &str, incoming: &str) -> bool {
    let existing_length = existing.chars().count();
    let incoming_length = incoming.chars().count();
    incoming_length > existing_length || (incoming_length == existing_length && incoming < existing)
}

fn titleize(norm: &str) -> String {
    let mut t: String = norm.chars().take(80).collect();
    if norm.chars().count() > 80 {
        t.push('…');
    }
    if let Some(first) = t.chars().next() {
        let upper = first.to_uppercase().to_string();
        t = upper + &t.chars().skip(1).collect::<String>();
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn clusters_similar_prompts() {
        let sessions = vec![
            SessionSummary {
                id: "1".into(),
                path: PathBuf::from("a"),
                title: None,
                cwd: None,
                started_at: None,
                user_turns: 1,
                assistant_turns: 1,
                tool_calls: 0,
                user_excerpts: vec!["please fix the login bug in auth".into()],
            },
            SessionSummary {
                id: "2".into(),
                path: PathBuf::from("b"),
                title: None,
                cwd: None,
                started_at: None,
                user_turns: 1,
                assistant_turns: 1,
                tool_calls: 0,
                user_excerpts: vec!["fix the login bug in auth module".into()],
            },
        ];
        let tasks = mine_tasks(&sessions, 5);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].frequency, 2);
        assert_eq!(tasks[0].source_session_ids, ["1", "2"]);
    }

    fn session(id: &str, excerpts: &[&str]) -> SessionSummary {
        SessionSummary {
            id: id.into(),
            path: PathBuf::from(id),
            title: None,
            cwd: None,
            started_at: None,
            user_turns: excerpts.len(),
            assistant_turns: 1,
            tool_calls: 0,
            user_excerpts: excerpts.iter().map(|excerpt| (*excerpt).into()).collect(),
        }
    }

    #[test]
    fn reordered_sessions_and_excerpts_produce_identical_task_ids_and_order() {
        let first = vec![
            session(
                "session-b",
                &[
                    "Document the release rollback procedure carefully",
                    "Fix login authentication failures in auth module",
                ],
            ),
            session(
                "session-a",
                &[
                    "Fix recurring login authentication failures in auth module",
                    "Document the release rollback procedure carefully",
                ],
            ),
        ];
        let second = vec![
            session(
                "session-a",
                &[
                    "Document the release rollback procedure carefully",
                    "Fix recurring login authentication failures in auth module",
                ],
            ),
            session(
                "session-b",
                &[
                    "Fix login authentication failures in auth module",
                    "Document the release rollback procedure carefully",
                ],
            ),
        ];

        let summarize = |tasks: Vec<MinedTask>| {
            tasks
                .into_iter()
                .map(|task| {
                    (
                        task.id,
                        task.prompt,
                        task.frequency,
                        task.source_session_ids,
                    )
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(
            summarize(mine_tasks(&first, 10)),
            summarize(mine_tasks(&second, 10))
        );
    }

    #[test]
    fn normalized_equivalent_representatives_receive_the_same_id() {
        let first = session(
            "session-a",
            &["Fix LOGIN at https://example.com for /Users/me/app id ABCDEF123456"],
        );
        let second = session(
            "session-b",
            &[" fix login at https://other.test for /tmp/app id deadbeef9999 "],
        );

        assert_eq!(
            mine_tasks(&[first], 1)[0].id,
            mine_tasks(&[second], 1)[0].id
        );
    }
}
