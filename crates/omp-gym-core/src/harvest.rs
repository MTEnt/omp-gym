use crate::types::SessionSummary;
use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use regex::Regex;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Harvest OMP session JSONL files under sessions_root.
pub fn harvest_sessions(
    sessions_root: &Path,
    lookback_hours: u64,
    max_sessions: usize,
) -> Result<Vec<SessionSummary>> {
    if !sessions_root.exists() {
        return Ok(vec![]);
    }

    let cutoff = if lookback_hours == 0 {
        None
    } else {
        Some(Utc::now() - Duration::hours(lookback_hours as i64))
    };

    let mut paths: Vec<PathBuf> = WalkDir::new(sessions_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| x == "jsonl")
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    // newest first by mtime
    paths.sort_by_key(|p| {
        std::cmp::Reverse(
            p.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
        )
    });

    let mut out = Vec::new();
    for path in paths {
        if out.len() >= max_sessions {
            break;
        }
        match parse_session(&path) {
            Ok(s) => {
                if let (Some(cut), Some(started)) = (cutoff, s.started_at) {
                    if started < cut {
                        continue;
                    }
                }
                out.push(s);
            }
            Err(_) => continue,
        }
    }
    Ok(out)
}

fn parse_session(path: &Path) -> Result<SessionSummary> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let mut title = None;
    let mut cwd = None;
    let mut started_at = None;
    let mut user_turns = 0usize;
    let mut assistant_turns = 0usize;
    let mut tool_calls = 0usize;
    let mut user_excerpts = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match ty {
            "session" => {
                if let Some(sid) = v.get("id").and_then(|x| x.as_str()) {
                    id = sid.to_string();
                }
                cwd = v
                    .get("cwd")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                title = v
                    .get("title")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                started_at = parse_ts(v.get("timestamp"));
            }
            "title" | "title_change" => {
                if let Some(t) = v.get("title").and_then(|x| x.as_str()) {
                    title = Some(t.to_string());
                }
            }
            "message" => {
                let role = v
                    .pointer("/message/role")
                    .and_then(|x| x.as_str())
                    .unwrap_or("");
                match role {
                    "user" => {
                        user_turns += 1;
                        if let Some(text) = extract_text(v.pointer("/message/content")) {
                            let cleaned = redact(&text);
                            if is_useful_user_text(&cleaned) && user_excerpts.len() < 12 {
                                user_excerpts.push(truncate(&cleaned, 500));
                            }
                        }
                    }
                    "assistant" => {
                        assistant_turns += 1;
                        if let Some(content) = v.pointer("/message/content").and_then(|c| c.as_array())
                        {
                            for part in content {
                                if part.get("type").and_then(|t| t.as_str()) == Some("toolCall")
                                    || part.get("type").and_then(|t| t.as_str()) == Some("tool_use")
                                {
                                    tool_calls += 1;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // fallback started_at from filename timestamp prefix if present
    if started_at.is_none() {
        started_at = filename_timestamp(path);
    }

    Ok(SessionSummary {
        id,
        path: path.to_path_buf(),
        title,
        cwd,
        started_at,
        user_turns,
        assistant_turns,
        tool_calls,
        user_excerpts,
    })
}

fn extract_text(content: Option<&Value>) -> Option<String> {
    let content = content?;
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }
    let arr = content.as_array()?;
    let mut parts = Vec::new();
    for item in arr {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(t) = item.get("text").and_then(|x| x.as_str()) {
                    parts.push(t.to_string());
                }
            }
            None => {
                if let Some(t) = item.get("text").and_then(|x| x.as_str()) {
                    parts.push(t.to_string());
                }
            }
            _ => {}
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn parse_ts(v: Option<&Value>) -> Option<DateTime<Utc>> {
    let v = v?;
    if let Some(s) = v.as_str() {
        return DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|d| d.with_timezone(&Utc));
    }
    if let Some(n) = v.as_i64() {
        // ms or s
        let secs = if n > 10_000_000_000 { n / 1000 } else { n };
        return DateTime::from_timestamp(secs, 0);
    }
    None
}

fn filename_timestamp(path: &Path) -> Option<DateTime<Utc>> {
    let name = path.file_name()?.to_str()?;
    // 2026-07-18T16-02-33-255Z_...
    let re = Regex::new(r"^(\d{4}-\d{2}-\d{2}T\d{2})-(\d{2})-(\d{2})").ok()?;
    let caps = re.captures(name)?;
    let s = format!(
        "{}:{}:{}Z",
        &caps[1].replace('T', "T"),
        &caps[2],
        &caps[3]
    );
    // 2026-07-18T16:02:33Z
    let normalized = s.replacen(' ', "T", 1);
    DateTime::parse_from_rfc3339(&normalized)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

fn is_useful_user_text(s: &str) -> bool {
    let t = s.trim();
    if t.len() < 8 {
        return false;
    }
    if t.starts_with('/') && t.len() < 40 {
        // bare slash commands
        return false;
    }
    if t.starts_with("<system") || t.starts_with("System:") {
        return false;
    }
    true
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

fn redact(s: &str) -> String {
    // best-effort secret-shaped redaction
    let patterns = [
        (r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+", "$1=[REDACTED]"),
        (r"sk-[A-Za-z0-9]{10,}", "sk-[REDACTED]"),
        (r"ghp_[A-Za-z0-9]{20,}", "ghp_[REDACTED]"),
        (
            r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
            "[EMAIL]",
        ),
    ];
    let mut out = s.to_string();
    for (pat, rep) in patterns {
        if let Ok(re) = Regex::new(pat) {
            out = re.replace_all(&out, rep).to_string();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_minimal_session() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("omp-gym-test-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sess.jsonl");
        let mut f = File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"type":"session","id":"abc","cwd":"/tmp","title":"Hello","timestamp":"2026-07-22T12:00:00Z"}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"message","message":{{"role":"user","content":[{{"type":"text","text":"please fix the login bug in auth.rs"}}]}}}}"#
        )
        .unwrap();
        let s = parse_session(&path).unwrap();
        assert_eq!(s.id, "abc");
        assert_eq!(s.user_turns, 1);
        assert!(s.user_excerpts[0].contains("login bug"));
        std::fs::remove_dir_all(dir).ok();
    }
}
