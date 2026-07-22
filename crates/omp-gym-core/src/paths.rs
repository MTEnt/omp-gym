use anyhow::{Context, Result};
use directories::UserDirs;
use std::path::{Path, PathBuf};

/// Default OMP agent home: ~/.omp/agent
pub fn omp_agent_home() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("PI_CODING_AGENT_DIR") {
        return Ok(PathBuf::from(p));
    }
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
pub fn ensure_private_dir(path: &Path) -> Result<()> {
    ensure_dir(path)?;
    let ignore = path.join(".gitignore");
    if !ignore.exists() {
        std::fs::write(&ignore, "*\n!.gitignore\n")
            .with_context(|| format!("write {}", ignore.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn agent_home_honors_omp_native_environment_variable() {
        let _guard = ENV_LOCK.lock().expect("lock environment");
        let names = ["PI_CODING_AGENT_DIR", "OMP_AGENT_DIR", "OMP_HOME"];
        let previous: Vec<_> = names
            .iter()
            .map(|name| ((*name).to_owned(), std::env::var_os(name)))
            .collect();
        for name in names {
            std::env::remove_var(name);
        }
        std::env::set_var("PI_CODING_AGENT_DIR", "/tmp/omp-native-agent");

        let resolved = omp_agent_home().expect("resolve agent home");

        for (name, value) in previous {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        assert_eq!(resolved, PathBuf::from("/tmp/omp-native-agent"));
    }
}
