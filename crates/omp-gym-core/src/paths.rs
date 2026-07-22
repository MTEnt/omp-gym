use anyhow::{Context, Result};
use directories::UserDirs;
use serde::{de::DeserializeOwned, Serialize};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
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
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    ensure_dir(parent)?;

    let file_name = path
        .file_name()
        .with_context(|| format!("atomic write path has no file name: {}", path.display()))?;
    let mut temp_name = OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let temp_path = parent.join(temp_name);

    let permissions = match std::fs::metadata(path) {
        Ok(metadata) => Some(metadata.permissions()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read destination metadata {}", path.display()));
        }
    };

    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .with_context(|| format!("create temporary file {}", temp_path.display()))?;
        if let Some(permissions) = permissions {
            file.set_permissions(permissions).with_context(|| {
                format!("copy destination permissions to {}", temp_path.display())
            })?;
        }
        file.write_all(bytes)
            .with_context(|| format!("write temporary file {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("sync temporary file {}", temp_path.display()))?;
        drop(file);

        std::fs::rename(&temp_path, path).with_context(|| {
            format!(
                "rename temporary file {} over {}",
                temp_path.display(),
                path.display()
            )
        })?;

        #[cfg(unix)]
        {
            let directory = File::open(parent)
                .with_context(|| format!("open parent directory {}", parent.display()))?;
            directory
                .sync_all()
                .with_context(|| format!("sync parent directory {}", parent.display()))?;
        }

        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

pub fn atomic_write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize JSON for {}", path.display()))?;
    atomic_write(path, &bytes)
}

pub fn load_json<T: DeserializeOwned>(path: &Path, description: &str) -> Result<T> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read {description} JSON {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {description} JSON {}", path.display()))
}

pub fn ensure_private_dir(path: &Path) -> Result<()> {
    ensure_dir(path)?;
    let ignore = path.join(".gitignore");
    if !ignore.exists() {
        atomic_write(&ignore, b"*\n!.gitignore\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    fn entry_names(path: &Path) -> Vec<String> {
        let mut names: Vec<_> = std::fs::read_dir(path)
            .expect("read test directory")
            .map(|entry| {
                entry
                    .expect("read directory entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    #[test]
    fn atomic_json_replaces_complete_document() {
        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("state.json");
        atomic_write_json(
            &path,
            &serde_json::json!({"value": 1, "obsolete": "long trailing data"}),
        )
        .expect("write first document");
        atomic_write_json(&path, &serde_json::json!({"value": 2})).expect("replace document");

        let value: serde_json::Value = load_json(&path, "state").expect("load state");
        assert_eq!(value, serde_json::json!({"value": 2}));
        assert_eq!(
            std::fs::read_to_string(&path).expect("read document"),
            serde_json::to_string_pretty(&serde_json::json!({"value": 2}))
                .expect("serialize expected document")
        );
        assert_eq!(entry_names(root.path()), vec!["state.json"]);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_preserves_existing_permissions() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("private.json");
        std::fs::write(&path, b"old").expect("write destination");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640))
            .expect("set destination permissions");

        atomic_write(&path, b"new").expect("replace destination");

        assert_eq!(
            std::fs::metadata(&path)
                .expect("destination metadata")
                .mode()
                & 0o777,
            0o640
        );
    }

    #[test]
    fn atomic_write_cleans_temporary_file_after_rename_error() {
        let root = tempdir().expect("create temporary directory");
        let destination = root.path().join("destination");
        std::fs::create_dir(&destination).expect("create conflicting destination directory");

        atomic_write(&destination, b"cannot replace a directory")
            .expect_err("rename over a directory must fail");

        assert_eq!(entry_names(root.path()), vec!["destination"]);
    }

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
