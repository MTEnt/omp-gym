use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

static REDACTIONS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (
            r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+",
            "$1=[REDACTED]",
        ),
        (r"sk-[A-Za-z0-9]{10,}", "sk-[REDACTED]"),
        (r"ghp_[A-Za-z0-9]{20,}", "ghp_[REDACTED]"),
        (
            r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
            "[EMAIL]",
        ),
        (
            r"(?i)(authorization\s*:\s*)?bearer\s+\S+",
            "${1}Bearer [REDACTED]",
        ),
        (
            r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            "[REDACTED PRIVATE KEY]",
        ),
    ]
    .into_iter()
    .map(|(pattern, replacement)| {
        (
            Regex::new(pattern).expect("privacy redaction regex is valid"),
            replacement,
        )
    })
    .collect()
});

pub fn redact_text(text: &str) -> String {
    let mut redacted = text.to_owned();
    for (pattern, replacement) in REDACTIONS.iter() {
        redacted = pattern.replace_all(&redacted, *replacement).into_owned();
    }
    redacted
}

pub fn redact_json_strings(value: &mut Value) {
    match value {
        Value::String(text) => *text = redact_text(text),
        Value::Array(items) => {
            for item in items {
                redact_json_strings(item);
            }
        }
        Value::Object(fields) => {
            for field in fields.values_mut() {
                redact_json_strings(field);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

pub fn bound_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut bounded: String = text.chars().take(max_chars).collect();
    bounded.push('…');
    bounded
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redact_text_covers_existing_and_runner_secret_shapes() {
        let input = "api_key=abc token: xyz sk-1234567890ABC ghp_12345678901234567890 owner@example.com Authorization: Bearer abc.def.ghi -----BEGIN PRIVATE KEY-----\nTOP-SECRET-KEY-MATERIAL\n-----END PRIVATE KEY-----";
        let redacted = redact_text(input);
        assert!(!redacted.contains("abc.def.ghi"));
        assert!(!redacted.contains("owner@example.com"));
        assert!(!redacted.contains("BEGIN PRIVATE KEY"));
        assert!(!redacted.contains("TOP-SECRET-KEY-MATERIAL"));
        assert!(redacted.contains("[REDACTED]"));
        assert!(redacted.contains("[EMAIL]"));
    }

    #[test]
    fn redact_json_recurses_through_nested_strings() {
        let mut value = json!({
            "event": {"content": ["safe", {"text": "Bearer highly-secret-token"}]},
            "number": 7
        });
        redact_json_strings(&mut value);
        assert_eq!(value["event"]["content"][0], "safe");
        assert_eq!(value["event"]["content"][1]["text"], "Bearer [REDACTED]");
        assert_eq!(value["number"], 7);
    }

    #[test]
    fn bound_chars_is_unicode_safe() {
        assert_eq!(bound_chars("aé🙂z", 3), "aé🙂…");
        assert_eq!(bound_chars("aé🙂", 3), "aé🙂");
    }
}
