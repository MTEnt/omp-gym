use anyhow::{Context, Result};
use directories::UserDirs;
use std::path::{Path, PathBuf};

/// Default OMP agent home: ~/.omp/agent
pub fn omp_agent_home() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("OMP_AGENT_DIR") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("OMP_HOME") {
        return Ok(PathBuf::from(p).join("agent"));
    }
    let home = UserDirs::new()
        .context("could not resolve home directory")?
        .home_dir()
        .to_path_buf();
    Ok(home.join(".omp").join("agent"))
}

pub fn omp_sessions_root() -> Result<PathBuf> {
    Ok(omp_agent_home()?.join("sessions"))
}

pub fn omp_skills_dirs() -> Result<Vec<PathBuf>> {
    let agent = omp_agent_home()?;
    let home = UserDirs::new()
        .context("could not resolve home directory")?
        .home_dir()
        .to_path_buf();
    Ok(vec![
        agent.join("skills"),
        home.join(".agents").join("skills"),
    ])
}

/// Per-project gym state lives under <project>/.omp/gym/
pub fn project_gym_dir(project: &Path) -> PathBuf {
    project.join(".omp").join("gym")
}

pub fn ensure_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)
        .with_context(|| format!("create directory {}", path.display()))?;
    Ok(())
}
