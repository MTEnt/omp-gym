use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use omp_gym_core::config::GymConfig;
use omp_gym_core::pipeline::{adopt, dry_run, run_night, status};
use omp_gym_core::state::{load_state, save_state, ScheduleState};
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(
    name = "omp-gym",
    version,
    about = "Project-scoped OMP session harvester and task-mining prototype"
)]
struct Cli {
    /// Project root (default: cwd)
    #[arg(long, global = true, default_value = ".")]
    project: PathBuf,

    /// Reserved target skill path; v0.1 never modifies it
    #[arg(long, global = true)]
    target_skill: Option<PathBuf>,

    /// Backend (v0.1 run supports mock only)
    #[arg(long, global = true, default_value = "mock")]
    backend: String,

    /// Max sessions to harvest
    #[arg(long, global = true, default_value_t = 20)]
    max_sessions: usize,

    /// Max mined tasks
    #[arg(long, global = true, default_value_t = 10)]
    max_tasks: usize,

    /// Lookback hours (0 = all)
    #[arg(long, global = true, default_value_t = 72)]
    lookback_hours: u64,

    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Show gym state and latest proposal
    Status,
    /// Harvest + mine only; write tasks.json; stage nothing
    DryRun,
    /// Harvest, mine, and stage mock proposal metadata
    Run {
        /// Harvest and mine without staging mock metadata
        #[arg(long)]
        no_stage: bool,
    },
    /// Reserved for non-mock proposals; v0.1 always refuses
    Adopt,
    /// Schedule daily mock harvest snapshots with macOS launchd
    Schedule {
        #[arg(long, default_value_t = 2)]
        hour: u32,
        #[arg(long, default_value_t = 15)]
        minute: u32,
        /// Remove schedule instead
        #[arg(long)]
        off: bool,
    },
    /// Print paths and environment diagnostics
    Doctor,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut cfg = GymConfig::for_project(&cli.project)?;
    cfg.backend = cli.backend;
    cfg.max_sessions = cli.max_sessions;
    cfg.max_tasks = cli.max_tasks;
    cfg.lookback_hours = cli.lookback_hours;
    if let Some(t) = cli.target_skill {
        cfg.target_skill = Some(t);
    }

    match cli.cmd {
        Commands::Status => {
            println!("{}", status(&cfg)?);
        }
        Commands::DryRun => {
            let report = dry_run(&cfg)?;
            print_report("dry-run", &report);
        }
        Commands::Run { no_stage } => {
            let report = run_night(&cfg, !no_stage)?;
            print_report("run", &report);
        }
        Commands::Adopt => {
            println!("{}", adopt(&cfg)?);
        }
        Commands::Schedule { hour, minute, off } => {
            schedule(&cfg, hour, minute, off)?;
        }
        Commands::Doctor => {
            doctor(&cfg)?;
        }
    }
    Ok(())
}

fn print_report(kind: &str, report: &omp_gym_core::GymReport) {
    println!("omp-gym {kind}");
    println!("  sessions: {}", report.sessions);
    println!("  tasks:    {}", report.tasks);
    println!("  backend:  {}", report.backend);
    println!("  staged:   {}", report.staged);
    if let Some(id) = &report.proposal_id {
        println!("  proposal: {id}");
    }
    println!("  gym dir:  {}", report.gym_dir.display());
    for n in &report.notes {
        println!("  - {n}");
    }
}

fn doctor(cfg: &GymConfig) -> Result<()> {
    println!("omp-gym doctor");
    println!("  project:        {}", cfg.project.display());
    println!("  gym dir:        {}", cfg.gym_dir().display());
    println!(
        "  sessions root:  {} {}",
        cfg.sessions_root.display(),
        if cfg.sessions_root.exists() {
            "OK"
        } else {
            "MISSING"
        }
    );
    println!(
        "  omp on PATH:    {}",
        which("omp").unwrap_or_else(|| "(not found)".into())
    );
    println!(
        "  omp-gym bin:    {}",
        std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "?".into())
    );
    if let Some(skill) = &cfg.target_skill {
        println!(
            "  target skill:   {} {}",
            skill.display(),
            if skill.exists() { "OK" } else { "MISSING" }
        );
    }
    Ok(())
}

fn which(bin: &str) -> Option<String> {
    Command::new("which")
        .arg(bin)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn schedule(cfg: &GymConfig, hour: u32, minute: u32, off: bool) -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (cfg, hour, minute, off);
        anyhow::bail!(
            "schedule is implemented for macOS launchd in v0.1; use cron manually elsewhere"
        );
    }

    #[cfg(target_os = "macos")]
    {
        let home = directories::UserDirs::new()
            .context("home dir")?
            .home_dir()
            .to_path_buf();
        let label = "com.mtent.omp-gym";
        let plist = home
            .join("Library/LaunchAgents")
            .join(format!("{label}.plist"));

        if off {
            let _ = Command::new("launchctl")
                .args(["unload", plist.to_str().unwrap()])
                .status();
            if plist.exists() {
                fs_err_remove(&plist)?;
            }
            let mut state = load_state(&cfg.state_path())?;
            state.schedule = Some(ScheduleState {
                enabled: false,
                hour_local: hour,
                minute_local: minute,
                label: label.into(),
            });
            save_state(&cfg.state_path(), &state)?;
            println!("schedule removed ({})", plist.display());
            return Ok(());
        }

        let exe = std::env::current_exe().context("current_exe")?;
        let project = cfg
            .project
            .canonicalize()
            .unwrap_or_else(|_| cfg.project.clone());
        omp_gym_core::paths::ensure_private_dir(&cfg.gym_dir())?;
        let log_dir = cfg.gym_dir().join("logs");
        omp_gym_core::paths::ensure_dir(&log_dir)?;
        let stdout = log_dir.join("launchd.out.log");
        let stderr = log_dir.join("launchd.err.log");

        let mut args = vec![
            exe.display().to_string(),
            "run".into(),
            "--project".into(),
            project.display().to_string(),
            "--backend".into(),
            cfg.backend.clone(),
            "--max-sessions".into(),
            cfg.max_sessions.to_string(),
            "--max-tasks".into(),
            cfg.max_tasks.to_string(),
        ];
        if let Some(skill) = &cfg.target_skill {
            args.push("--target-skill".into());
            args.push(skill.display().to_string());
        }

        let args_xml = args
            .iter()
            .map(|a| format!("    <string>{}</string>", xml_escape(a)))
            .collect::<Vec<_>>()
            .join("\n");

        let body = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
{args_xml}
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>{hour}</integer>
    <key>Minute</key>
    <integer>{minute}</integer>
  </dict>
  <key>WorkingDirectory</key>
  <string>{cwd}</string>
  <key>StandardOutPath</key>
  <string>{stdout}</string>
  <key>StandardErrorPath</key>
  <string>{stderr}</string>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
"#,
            cwd = xml_escape(&project.display().to_string()),
            stdout = xml_escape(&stdout.display().to_string()),
            stderr = xml_escape(&stderr.display().to_string()),
        );

        if let Some(parent) = plist.parent() {
            omp_gym_core::paths::ensure_dir(parent)?;
        }
        std::fs::write(&plist, body).with_context(|| format!("write {}", plist.display()))?;
        let _ = Command::new("launchctl")
            .args(["unload", plist.to_str().unwrap()])
            .status();
        let status = Command::new("launchctl")
            .args(["load", plist.to_str().unwrap()])
            .status()
            .context("launchctl load")?;
        if !status.success() {
            anyhow::bail!("launchctl load failed");
        }

        let mut state = load_state(&cfg.state_path())?;
        state.schedule = Some(ScheduleState {
            enabled: true,
            hour_local: hour,
            minute_local: minute,
            label: label.into(),
        });
        save_state(&cfg.state_path(), &state)?;
        println!(
            "scheduled {label} daily at {hour:02}:{minute:02} local\n  plist: {}\n  logs:  {}",
            plist.display(),
            log_dir.display()
        );
        Ok(())
    }
}

fn fs_err_remove(path: &std::path::Path) -> Result<()> {
    std::fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
    Ok(())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
