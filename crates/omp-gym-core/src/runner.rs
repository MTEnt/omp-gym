use crate::config::GymConfig;
use crate::privacy::redact_text;
use crate::types::{ModelRole, Trajectory, SCHEMA_VERSION};
use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{self, Read};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use wait_timeout::ChildExt;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

// wait-timeout installs process-global SIGCHLD bookkeeping on Unix. Serializing
// its wait section avoids cross-child wakeup races while capture remains concurrent.
static PROCESS_WAIT: Mutex<()> = Mutex::new(());


pub struct ModelRequest<'a> {
    pub role: ModelRole,
    pub prompt: &'a str,
    pub skill: &'a str,
}

pub trait ModelRunner {
    fn run(&self, request: &ModelRequest<'_>) -> Result<Trajectory>;
}

pub struct OmpRunner {
    config: GymConfig,
}

impl OmpRunner {
    pub fn new(config: GymConfig) -> Self {
        Self { config }
    }
}

impl ModelRunner for OmpRunner {
    fn run(&self, request: &ModelRequest<'_>) -> Result<Trajectory> {
        let timeout_secs = self.config.timeout_secs_for(&request.role);
        if timeout_secs == 0 {
            bail!("model timeout seconds must be greater than zero");
        }
        if self.config.max_output_bytes == 0 {
            bail!("maximum output bytes must be greater than zero");
        }

        let workspace = tempfile::tempdir().context("create isolated OMP replay directory")?;
        let skill_path = workspace.path().join("skill.md");
        let overlay_path = workspace.path().join("omp-overlay.yaml");
        std::fs::write(&skill_path, request.skill).context("write isolated replay skill")?;
        std::fs::write(
            &overlay_path,
            "advisor:\n  enabled: false\nprewalk:\n  enabled: false\nplan:\n  defaultOnStartup: false\n",
        )
        .context("write isolated OMP configuration overlay")?;

        let mut command = Command::new(&self.config.omp_bin);
        command
            .arg("-p")
            .args(["--mode", "json"])
            .arg("--no-session")
            .arg("--no-tools")
            .arg("--no-skills")
            .arg("--no-extensions")
            .arg("--no-rules")
            .arg("--no-prewalk")
            .arg("--no-title")
            .arg("--cwd")
            .arg(workspace.path())
            .arg("--config")
            .arg(&overlay_path)
            .arg("--append-system-prompt")
            .arg(&skill_path)
            .arg("--max-time")
            .arg(timeout_secs.to_string());
        if let Some(model) = self.config.model_for(&request.role) {
            command.args(["--model", model]);
        }
        command
            .arg("--")
            .arg(request.prompt)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(unix)]
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let started_at = Utc::now();
        let started = Instant::now();
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(_) => {
                return Ok(Trajectory {
                    schema_version: SCHEMA_VERSION,
                    id: format!("trajectory-{}", uuid::Uuid::new_v4()),
                    role: request.role.clone(),
                    task_id: None,
                    started_at,
                    duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                    prompt_hash: hash_text(request.prompt),
                    skill_hash: hash_text(request.skill),
                    model: None,
                    process_success: false,
                    exit_code: None,
                    timed_out: false,
                    response_nonempty: false,
                    final_text: None,
                    events: Vec::new(),
                    stderr: String::new(),
                    error: Some("failed to start OMP process".into()),
                });
            }
        };
        let child_pid = child.id();
        let stdout = child.stdout.take().context("capture OMP stdout")?;
        let stderr = child.stderr.take().context("capture OMP stderr")?;
        let limit = self.config.max_output_bytes;
        let stdout_thread = thread::spawn(move || read_bounded(stdout, limit));
        let stderr_thread = thread::spawn(move || read_bounded(stderr, limit));

        let timeout = Duration::from_secs(timeout_secs);
        let mut timed_out = false;
        let wait_guard = PROCESS_WAIT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (status, wait_error) = match child.wait_timeout(timeout) {
            Ok(Some(status)) => (Some(status), None),
            Ok(None) => {
                timed_out = true;
                terminate_process_group(&mut child, child_pid);
                match child.wait() {
                    Ok(status) => (Some(status), None),
                    Err(error) => (
                        None,
                        Some(format!("reap timed-out OMP process: {error}")),
                    ),
                }
            }
            Err(error) => {
                terminate_process_group(&mut child, child_pid);
                let status = child.wait().ok();
                (
                    status,
                    Some(format!("wait for OMP process failed: {error}")),
                )
            }
        };
        drop(wait_guard);
        // The leader may exit while descendants still hold capture pipes.
        terminate_process_group(&mut child, child_pid);

        let (stdout_capture, stdout_error) = join_capture(stdout_thread, "stdout");
        let (stderr_capture, stderr_error) = join_capture(stderr_thread, "stderr");
        let stdout = String::from_utf8_lossy(&stdout_capture.bytes);
        let stderr = sanitize_text_evidence(
            &String::from_utf8_lossy(&stderr_capture.bytes),
            request.prompt,
            request.skill,
        );
        let stderr = bounded_utf8(&stderr, limit);
        let (events, final_text, model, event_error) =
            parse_events(&stdout, limit, request.prompt, request.skill);

        let mut errors = Vec::new();
        errors.extend(wait_error);
        errors.extend(stdout_error);
        errors.extend(stderr_error);
        if stdout_capture.truncated {
            errors.push(format!("stdout capture truncated at {limit} bytes"));
        }
        if stderr_capture.truncated {
            errors.push(format!("stderr capture truncated at {limit} bytes"));
        }
        if timed_out {
            errors.push(format!("OMP process timed out after {timeout_secs} seconds"));
        }
        if status.as_ref().is_some_and(|status| !status.success()) && !timed_out {
            errors.push(format!(
                "OMP process exited with status {}",
                status
                    .as_ref()
                    .and_then(|status| status.code())
                    .map(|code| code.to_string())
                    .unwrap_or_else(|| "terminated by signal".into())
            ));
        }
        if let Some(error) = event_error {
            errors.push(error);
        }
        if final_text
            .as_deref()
            .is_none_or(|text| text.trim().is_empty())
        {
            errors.push("OMP output contained no terminal assistant event".into());
        }

        let response_nonempty = final_text
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty());
        let process_success = status.as_ref().is_some_and(|status| status.success())
            && !timed_out
            && errors.is_empty();
        Ok(Trajectory {
            schema_version: SCHEMA_VERSION,
            id: format!("trajectory-{}", uuid::Uuid::new_v4()),
            role: request.role.clone(),
            task_id: None,
            started_at,
            duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            prompt_hash: hash_text(request.prompt),
            skill_hash: hash_text(request.skill),
            model,
            process_success,
            exit_code: status.as_ref().and_then(|status| status.code()),
            timed_out,
            response_nonempty,
            final_text,
            events,
            stderr,
            error: (!errors.is_empty()).then(|| errors.join("; ")),
        })
    }
}

fn terminate_process_group(child: &mut std::process::Child, child_pid: u32) {
    #[cfg(unix)]
    {
        let _ = child;
        unsafe {
            libc::kill(-(child_pid as i32), libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child_pid;
        let _ = child.kill();
    }
}

struct Capture {
    bytes: Vec<u8>,
    truncated: bool,
}

fn read_bounded(mut reader: impl Read, limit: usize) -> io::Result<Capture> {
    let mut retained = Vec::with_capacity(limit.min(8192));
    let mut truncated = false;
    let mut buffer = [0_u8; 8192];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(Capture {
                bytes: retained,
                truncated,
            });
        }
        let remaining = limit.saturating_sub(retained.len());
        let retained_now = read.min(remaining);
        retained.extend_from_slice(&buffer[..retained_now]);
        truncated |= retained_now < read;
    }
}

fn join_capture(
    handle: thread::JoinHandle<io::Result<Capture>>,
    stream: &str,
) -> (Capture, Option<String>) {
    match handle.join() {
        Ok(Ok(capture)) => (capture, None),
        Ok(Err(error)) => (
            Capture {
                bytes: Vec::new(),
                truncated: false,
            },
            Some(format!("read OMP {stream}: {error}")),
        ),
        Err(_) => (
            Capture {
                bytes: Vec::new(),
                truncated: false,
            },
            Some(format!("OMP {stream} capture thread panicked")),
        ),
    }
}

fn bounded_utf8(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_owned();
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

fn parse_events(
    stdout: &str,
    limit: usize,
    prompt: &str,
    skill: &str,
) -> (Vec<Value>, Option<String>, Option<String>, Option<String>) {
    let mut events = Vec::new();
    let mut events_bytes = 2_usize;
    let mut events_truncated = false;
    let mut assistant_message_observed = false;
    let mut message_end = None;
    let mut message_error = None;
    let mut agent_end = None;
    let mut agent_error = None;
    let mut model = None;

    for (index, line) in stdout.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut event: Value = match serde_json::from_str(line) {
            Ok(event) => event,
            Err(error) => {
                return (
                    events,
                    None,
                    model,
                    Some(format!("parse NDJSON line {}: {error}", index + 1)),
                );
            }
        };
        model = extract_model(&event).or(model);
        match event.get("type").and_then(Value::as_str) {
            Some("message_end") => {
                if let Some(message) = event.get("message").filter(|message| {
                    message.get("role").and_then(Value::as_str) == Some("assistant")
                }) {
                    assistant_message_observed = true;
                    message_end = message
                        .get("content")
                        .and_then(extract_text)
                        .map(|text| redact_text(&text));
                    message_error = terminal_failure(message);
                }
            }
            Some("agent_end") => {
                agent_end = extract_agent_end(&event)
                    .map(|text| redact_text(&text))
                    .or(agent_end);
                agent_error = terminal_failure(&event).or_else(|| {
                    event
                        .get("messages")
                        .and_then(Value::as_array)
                        .and_then(|messages| {
                            messages
                                .iter()
                                .rev()
                                .find(|message| {
                                    message.get("role").and_then(Value::as_str)
                                        == Some("assistant")
                                })
                                .and_then(terminal_failure)
                        })
                });
            }
            _ => {}
        }

        sanitize_event_evidence(&mut event, prompt, skill);
        let event_bytes = serde_json::to_vec(&event)
            .expect("JSON value evidence must serialize")
            .len();
        let separator = usize::from(!events.is_empty());
        if events_bytes
            .saturating_add(separator)
            .saturating_add(event_bytes)
            <= limit
        {
            events_bytes += separator + event_bytes;
            events.push(event);
        } else {
            events_truncated = true;
        }
    }

    let (final_text, terminal_error) = if assistant_message_observed {
        (message_end, message_error)
    } else {
        (agent_end, agent_error)
    };
    let terminal_error =
        terminal_error.map(|error| sanitize_text_evidence(&error, prompt, skill));
    let event_error = if events_truncated {
        Some(match terminal_error {
            Some(error) => format!("{error}; retained events truncated at {limit} bytes"),
            None => format!("retained events truncated at {limit} bytes"),
        })
    } else {
        terminal_error
    };
    (events, final_text, model, event_error)
}

fn sanitize_event_evidence(value: &mut Value, prompt: &str, skill: &str) {
    match value {
        Value::String(text) => *text = sanitize_text_evidence(text, prompt, skill),
        Value::Array(items) => {
            for item in items {
                sanitize_event_evidence(item, prompt, skill);
            }
        }
        Value::Object(fields) => {
            let sensitive_role = fields
                .get("role")
                .and_then(Value::as_str)
                .is_some_and(|role| matches!(role, "user" | "system" | "developer"));
            for (key, field) in fields {
                let normalized_key: String = key
                    .chars()
                    .filter(|character| character.is_ascii_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect();
                if normalized_key.contains("prompt") {
                    *field = Value::String("[REDACTED PROMPT FIELD]".into());
                } else if sensitive_role && normalized_key == "content" {
                    *field = Value::String("[REDACTED MESSAGE CONTENT]".into());
                } else {
                    sanitize_event_evidence(field, prompt, skill);
                }
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn sanitize_text_evidence(text: &str, prompt: &str, skill: &str) -> String {
    let mut sanitized = text.to_owned();
    if !prompt.is_empty() {
        sanitized = sanitized.replace(prompt, "[REDACTED REQUEST PROMPT]");
    }
    if !skill.is_empty() {
        sanitized = sanitized.replace(skill, "[REDACTED REQUEST SKILL]");
    }
    redact_text(&sanitized)
}

fn extract_agent_end(event: &Value) -> Option<String> {
    event
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| {
            messages.iter().rev().find_map(|message| {
                (message.get("role").and_then(Value::as_str) == Some("assistant"))
                    .then(|| message.get("content").and_then(extract_text))
                    .flatten()
            })
        })
        .or_else(|| event.pointer("/message/content").and_then(extract_text))
        .or_else(|| event.get("content").and_then(extract_text))
        .or_else(|| event.get("text").and_then(Value::as_str).map(str::to_owned))
}

fn terminal_failure(value: &Value) -> Option<String> {
    let reason = value
        .get("stopReason")
        .or_else(|| value.get("stop_reason"))
        .and_then(Value::as_str);
    let message = value
        .get("errorMessage")
        .or_else(|| value.get("error_message"))
        .and_then(Value::as_str);
    match (reason, message) {
        (Some("error" | "aborted"), Some(message)) => Some(message.to_owned()),
        (Some(reason @ ("error" | "aborted")), None) => {
            Some(format!("terminal assistant stop reason was {reason}"))
        }
        (_, Some(message)) => Some(message.to_owned()),
        _ => None,
    }
}

fn extract_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let text: String = parts
                .iter()
                .filter(|part| {
                    part.get("type")
                        .and_then(Value::as_str)
                        .is_none_or(|kind| kind == "text")
                })
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect();
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn extract_model(event: &Value) -> Option<String> {
    ["/message/model", "/model", "/message/modelId", "/model/id"]
        .into_iter()
        .find_map(|pointer| event.pointer(pointer).and_then(Value::as_str))
        .map(str::to_owned)
}

fn hash_text(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::config::GymConfig;
    use crate::types::ModelRole;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn write_executable(path: &Path, body: &str) {
        std::fs::write(path, body).unwrap();
        let mut permissions = std::fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(path, permissions).unwrap();
    }

    fn config_with_bin(project: &Path, bin: PathBuf) -> GymConfig {
        let mut config = GymConfig::load(project).unwrap();
        config.omp_bin = bin;
        config.replay_timeout_secs = 2;
        config.max_output_bytes = 4096;
        config
    }

    fn replay_request<'a>() -> ModelRequest<'a> {
        ModelRequest {
            role: ModelRole::Replay,
            prompt: "Return the required result",
            skill: "Always answer with GYM_OK.",
        }
    }

    #[test]
    fn extracts_latest_assistant_message_end_and_redacts_events() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            r#"#!/bin/sh
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"old"}]}}'
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"GYM_OK result"}],"provider":"local","model":"fixture-model"},"nested":{"secret":"Bearer abc.def.ghi"}}'
"#,
        );
        let runner = OmpRunner::new(config_with_bin(root.path(), fake));

        let trajectory = runner.run(&replay_request()).unwrap();

        assert!(trajectory.process_success, "{:?}", trajectory.error);
        assert_eq!(trajectory.final_text.as_deref(), Some("GYM_OK result"));
        assert_eq!(trajectory.model.as_deref(), Some("fixture-model"));
        assert!(!serde_json::to_string(&trajectory.events).unwrap().contains("abc.def.ghi"));
        assert_eq!(trajectory.task_id, None);
        assert!(!trajectory.prompt_hash.is_empty());
        assert!(!trajectory.skill_hash.is_empty());
    }

    #[test]
    fn uses_agent_end_as_terminal_fallback() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            r#"#!/bin/sh
printf '%s\n' '{"type":"agent_end","messages":[{"role":"user","content":"question"},{"role":"assistant","content":[{"type":"text","text":"fallback result"}]}]}'
"#,
        );

        let trajectory = OmpRunner::new(config_with_bin(root.path(), fake))
            .run(&replay_request())
            .unwrap();

        assert!(trajectory.process_success, "{:?}", trajectory.error);
        assert_eq!(trajectory.final_text.as_deref(), Some("fallback result"));
    }

    #[test]
    fn observed_message_end_never_falls_back_to_agent_end() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            r#"#!/bin/sh
printf '%s\n' '{"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"stale fallback"}]}]}'
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[],"stopReason":"stop"}}'
"#,
        );

        let trajectory = OmpRunner::new(config_with_bin(root.path(), fake))
            .run(&replay_request())
            .unwrap();

        assert!(!trajectory.process_success);
        assert_eq!(trajectory.final_text, None);
        assert!(trajectory.error.unwrap().contains("terminal assistant"));
    }

    #[test]
    fn spawn_failure_returns_a_generic_failed_trajectory() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("missing-secret-binary");
        let non_executable = root.path().join("non-executable-secret-binary");
        std::fs::write(&non_executable, "#!/bin/sh\nexit 0\n").unwrap();
        let request = replay_request();

        for binary in [missing, non_executable] {
            let trajectory = OmpRunner::new(config_with_bin(root.path(), binary))
                .run(&request)
                .unwrap();

            assert!(!trajectory.process_success);
            assert_eq!(trajectory.exit_code, None);
            assert!(!trajectory.timed_out);
            assert_eq!(trajectory.role, ModelRole::Replay);
            assert_eq!(trajectory.prompt_hash, hash_text(request.prompt));
            assert_eq!(trajectory.skill_hash, hash_text(request.skill));
            let error = trajectory.error.unwrap();
            assert_eq!(error, "failed to start OMP process");
            assert!(!error.contains("secret-binary"));
        }
    }

    #[test]
    fn strips_input_events_and_uses_latest_nonterminal_model_metadata() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            r#"#!/bin/sh
printf '%s\n' '{"type":"model_start","model":"old-model"}'
printf '%s\n' '{"type":"model_change","model":"latest-model","nested":{"message":{"role":"user","content":"Return the required result"},"systemPrompt":"Always answer with GYM_OK.","developer":{"role":"developer","content":"Return the required result"}}}'
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"answer secret=do-not-keep"}],"stopReason":"stop"}}'
printf '%s' 'Return the required result Always answer with GYM_OK. token=stderr-secret' >&2
"#,
        );

        let trajectory = OmpRunner::new(config_with_bin(root.path(), fake))
            .run(&replay_request())
            .unwrap();

        let events = serde_json::to_string(&trajectory.events).unwrap();
        assert!(!events.contains("Return the required result"));
        assert!(!events.contains("Always answer with GYM_OK."));
        assert!(!trajectory.stderr.contains("Return the required result"));
        assert!(!trajectory.stderr.contains("Always answer with GYM_OK."));
        assert!(!trajectory.stderr.contains("stderr-secret"));
        let error = trajectory.error.as_deref().unwrap_or_default();
        assert!(!error.contains("Return the required result"));
        assert!(!error.contains("Always answer with GYM_OK."));
        assert_eq!(trajectory.model.as_deref(), Some("latest-model"));
        assert_eq!(
            trajectory.final_text.as_deref(),
            Some("answer secret=[REDACTED]")
        );
    }

    #[test]
    fn passes_exact_isolation_contract_and_cleans_temporary_files() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            r#"#!/bin/sh
printf '%s\n' "$@" > "$0.args"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cwd) printf '%s' "$2" > "$0.cwd"; shift 2 ;;
    --config) cp "$2" "$0.overlay"; shift 2 ;;
    --append-system-prompt) cp "$2" "$0.skill"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"ok"}],"stopReason":"stop"}}'
"#,
        );
        let mut config = config_with_bin(root.path(), fake.clone());
        config.optimizer_timeout_secs = 9;
        config.optimizer_model = Some("fixture-optimizer".into());
        let request = ModelRequest {
            role: ModelRole::Optimizer,
            prompt: "--help",
            skill: "exact optimizer skill",
        };

        let trajectory = OmpRunner::new(config).run(&request).unwrap();

        assert!(trajectory.process_success, "{:?}", trajectory.error);
        let args: Vec<_> = std::fs::read_to_string(fake.with_extension("args"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect();
        assert_eq!(
            &args[..8],
            ["-p", "--mode", "json", "--no-session", "--no-tools", "--no-skills", "--no-extensions", "--no-rules"]
        );
        assert_eq!(&args[8..11], ["--no-prewalk", "--no-title", "--cwd"]);
        let cwd = PathBuf::from(&args[11]);
        assert_eq!(args[12], "--config");
        let overlay = PathBuf::from(&args[13]);
        assert_eq!(args[14], "--append-system-prompt");
        let skill = PathBuf::from(&args[15]);
        assert_eq!(
            &args[16..],
            ["--max-time", "9", "--model", "fixture-optimizer", "--", "--help"]
        );
        assert_eq!(
            std::fs::read_to_string(fake.with_extension("overlay")).unwrap(),
            "advisor:\n  enabled: false\nprewalk:\n  enabled: false\nplan:\n  defaultOnStartup: false\n"
        );
        assert_eq!(
            std::fs::read_to_string(fake.with_extension("skill")).unwrap(),
            "exact optimizer skill"
        );
        assert!(!cwd.exists());
        assert!(!overlay.exists());
        assert!(!skill.exists());
    }

    #[test]
    fn malformed_ndjson_is_an_explicit_trajectory_failure() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(&fake, "#!/bin/sh\nprintf '%s\\n' 'not-json'\n");

        let trajectory = OmpRunner::new(config_with_bin(root.path(), fake))
            .run(&replay_request())
            .unwrap();

        assert!(!trajectory.process_success);
        assert!(trajectory.error.unwrap().contains("parse NDJSON"));
    }

    #[test]
    fn missing_terminal_assistant_event_is_an_explicit_failure() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(&fake, "#!/bin/sh\nprintf '%s\\n' '{\"type\":\"start\"}'\n");

        let trajectory = OmpRunner::new(config_with_bin(root.path(), fake))
            .run(&replay_request())
            .unwrap();

        assert!(!trajectory.process_success);
        assert!(trajectory.error.unwrap().contains("terminal assistant"));
    }

    #[test]
    fn error_stop_reason_is_an_explicit_trajectory_failure() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            r#"#!/bin/sh
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"partial"}],"stopReason":"error","errorMessage":"provider failed"}}'
"#,
        );

        let trajectory = OmpRunner::new(config_with_bin(root.path(), fake))
            .run(&replay_request())
            .unwrap();

        assert!(!trajectory.process_success);
        assert!(trajectory.error.unwrap().contains("provider failed"));
    }

    #[test]
    fn successful_leader_does_not_leave_inherited_capture_pipes_open() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            r#"#!/bin/sh
(sleep 10) &
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"complete"}],"stopReason":"stop"}}'
"#,
        );
        let started = std::time::Instant::now();

        let trajectory = OmpRunner::new(config_with_bin(root.path(), fake))
            .run(&replay_request())
            .unwrap();

        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(trajectory.process_success, "{:?}", trajectory.error);
    }

    #[test]
    fn bounds_captured_output_without_splitting_unicode() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            r#"#!/bin/sh
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]}}'
printf 'éééééééééééééééééééé' >&2
"#,
        );
        let mut config = config_with_bin(root.path(), fake);
        config.max_output_bytes = 17;

        let trajectory = OmpRunner::new(config).run(&replay_request()).unwrap();

        assert!(trajectory.stderr.len() <= 17);
        assert!(std::str::from_utf8(trajectory.stderr.as_bytes()).is_ok());
        assert!(serde_json::to_vec(&trajectory.events).unwrap().len() <= 17);
        assert!(trajectory.error.unwrap().contains("truncated"));
    }

    #[test]
    fn nonzero_exit_returns_failed_trajectory_with_evidence() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(
            &fake,
            "#!/bin/sh\nprintf '%s\\n' '{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"partial\"}]}}'\nprintf 'fixture failure' >&2\nexit 7\n",
        );

        let trajectory = OmpRunner::new(config_with_bin(root.path(), fake))
            .run(&replay_request())
            .unwrap();

        assert!(!trajectory.process_success);
        assert_eq!(trajectory.exit_code, Some(7));
        assert!(trajectory.stderr.contains("fixture failure"));
        assert!(trajectory.error.unwrap().contains("status"));
    }

    #[test]
    fn timeout_kills_process_group_and_returns_failed_trajectory() {
        let root = tempfile::tempdir().unwrap();
        let fake = root.path().join("omp");
        write_executable(&fake, "#!/bin/sh\nsleep 10\n");
        let mut config = config_with_bin(root.path(), fake);
        config.replay_timeout_secs = 1;
        let started = std::time::Instant::now();

        let trajectory = OmpRunner::new(config).run(&replay_request()).unwrap();

        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(trajectory.timed_out);
        assert!(!trajectory.process_success);
        assert!(trajectory.error.unwrap().contains("timed out"));
    }
}
