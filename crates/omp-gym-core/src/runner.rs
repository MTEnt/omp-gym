use crate::config::GymConfig;
use crate::privacy::{redact_json_strings, redact_text};
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

// wait-timeout's Unix waiter is process-global; serialize waits so simultaneous
// replay tests (and callers) cannot race SIGCHLD bookkeeping.
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

        let wait_guard = PROCESS_WAIT
            .lock()
            .map_err(|_| anyhow::anyhow!("OMP process wait lock is poisoned"))?;
        let started_at = Utc::now();
        let started = Instant::now();
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn OMP binary {}", self.config.omp_bin.display()))?;
        let child_pid = child.id();
        let stdout = child.stdout.take().context("capture OMP stdout")?;
        let stderr = child.stderr.take().context("capture OMP stderr")?;
        let limit = self.config.max_output_bytes;
        let stdout_thread = thread::spawn(move || read_bounded(stdout, limit));
        let stderr_thread = thread::spawn(move || read_bounded(stderr, limit));

        let timeout = Duration::from_secs(timeout_secs);
        let mut timed_out = false;
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
        // The leader may exit while descendants still hold capture pipes.
        terminate_process_group(&mut child, child_pid);
        drop(wait_guard);

        let (stdout_bytes, stdout_error) = join_capture(stdout_thread, "stdout");
        let (stderr_bytes, stderr_error) = join_capture(stderr_thread, "stderr");
        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let stderr = redact_text(&String::from_utf8_lossy(&stderr_bytes));
        let stderr = bounded_utf8(&stderr, limit);
        let (events, final_text, model, event_error) = parse_events(&stdout);

        let mut errors = Vec::new();
        errors.extend(wait_error);
        errors.extend(stdout_error);
        errors.extend(stderr_error);
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

fn read_bounded(mut reader: impl Read, limit: usize) -> io::Result<Vec<u8>> {
    let mut retained = Vec::with_capacity(limit.min(8192));
    let mut buffer = [0_u8; 8192];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(retained);
        }
        let remaining = limit.saturating_sub(retained.len());
        retained.extend_from_slice(&buffer[..read.min(remaining)]);
    }
}

fn join_capture(
    handle: thread::JoinHandle<io::Result<Vec<u8>>>,
    stream: &str,
) -> (Vec<u8>, Option<String>) {
    match handle.join() {
        Ok(Ok(bytes)) => (bytes, None),
        Ok(Err(error)) => (Vec::new(), Some(format!("read OMP {stream}: {error}"))),
        Err(_) => (
            Vec::new(),
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

fn parse_events(stdout: &str) -> (Vec<Value>, Option<String>, Option<String>, Option<String>) {
    let mut events = Vec::new();
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
        match event.get("type").and_then(Value::as_str) {
            Some("message_end") => {
                if let Some(message) = event.get("message").filter(|message| {
                    message.get("role").and_then(Value::as_str) == Some("assistant")
                }) {
                    message_end = message.get("content").and_then(extract_text);
                    message_error = terminal_failure(message);
                    model = extract_model(&event).or(model);
                }
            }
            Some("agent_end") => {
                agent_end = extract_agent_end(&event).or(agent_end);
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
                model = extract_model(&event).or(model);
            }
            _ => {}
        }
        redact_json_strings(&mut event);
        events.push(event);
    }

    if message_end.is_some() {
        (events, message_end, model, message_error)
    } else {
        (events, agent_end, model, agent_error)
    }
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

        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(trajectory.timed_out);
        assert!(!trajectory.process_success);
        assert!(trajectory.error.unwrap().contains("timed out"));
    }
}
