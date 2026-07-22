use crate::types::{MinedTask, SessionSummary};
use regex::Regex;
use std::collections::HashSet;
use uuid::Uuid;

/// Mine recurring / representative tasks from harvested sessions.
/// v0.1: normalize user excerpts, cluster by token similarity, rank by frequency.
pub fn mine_tasks(sessions: &[SessionSummary], max_tasks: usize) -> Vec<MinedTask> {
    let mut clusters: Vec<Cluster> = Vec::new();

    for session in sessions {
        for excerpt in &session.user_excerpts {
            let norm = normalize(excerpt);
            if norm.len() < 12 {
                continue;
            }
            let tokens = significant_tokens(&norm);
            if tokens.is_empty() {
                continue;
            }
            let matching_cluster = clusters
                .iter()
                .position(|cluster| token_similarity(&cluster.tokens, &tokens) >= 0.7);
            if matching_cluster.is_none() {
                clusters.push(Cluster {
                    title: titleize(&norm),
                    prompt: excerpt.clone(),
                    sessions: Vec::new(),
                    count: 0,
                    tokens,
                });
            }
            let cluster_index = matching_cluster.unwrap_or(clusters.len() - 1);
            let entry = &mut clusters[cluster_index];
            entry.count += 1;
            if !entry.sessions.iter().any(|s| s == &session.id) {
                entry.sessions.push(session.id.clone());
            }
            // keep longest prompt as canonical
            if excerpt.len() > entry.prompt.len() {
                entry.prompt = excerpt.clone();
                entry.title = titleize(&norm);
            }
        }
    }

    let mut items = clusters;
    items.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then(b.prompt.len().cmp(&a.prompt.len()))
    });

    items
        .into_iter()
        .take(max_tasks)
        .map(|c| MinedTask {
            id: Uuid::new_v4().to_string(),
            title: c.title,
            prompt: c.prompt,
            source_session_ids: c.sessions,
            frequency: c.count,
            reviewed: false,
        })
        .collect()
}

struct Cluster {
    title: String,
    prompt: String,
    sessions: Vec<String>,
    count: usize,
    tokens: HashSet<String>,
}

fn normalize(s: &str) -> String {
    let s = s.to_lowercase();
    let s = Regex::new(r"\s+").unwrap().replace_all(&s, " ");
    let s = Regex::new(r"https?://\S+")
        .unwrap()
        .replace_all(&s, "<url>");
    let s = Regex::new(r"/[\w./-]+").unwrap().replace_all(&s, "<path>");
    let s = Regex::new(r"\b[0-9a-f]{7,}\b")
        .unwrap()
        .replace_all(&s, "<id>");
    s.trim().to_string()
}

fn significant_tokens(norm: &str) -> HashSet<String> {
    const STOP_WORDS: [&str; 12] = [
        "the", "a", "an", "to", "of", "and", "in", "for", "on", "please", "can", "you",
    ];
    norm.split_whitespace()
        .map(|token| token.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|token| token.len() > 2 && !STOP_WORDS.contains(token))
        .take(24)
        .map(str::to_owned)
        .collect()
}

fn token_similarity(left: &HashSet<String>, right: &HashSet<String>) -> f64 {
    let intersection = left.intersection(right).count();
    let union = left.len() + right.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
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
}
