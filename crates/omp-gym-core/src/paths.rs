use anyhow::{Context, Result};
use directories::UserDirs;
use serde::{de::DeserializeOwned, Serialize};
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
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
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
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
    #[cfg(unix)]
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("set private directory permissions {}", path.display()))?;
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
    fn ensure_private_dir_sets_new_directory_mode_to_0700() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempdir().expect("create temporary directory");
        let path = root.path().join(".omp").join("gym");

        ensure_private_dir(&path).expect("create private directory");

        assert_eq!(
            std::fs::metadata(&path)
                .expect("private directory metadata")
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_creates_new_artifact_mode_0600() {
        use std::os::unix::fs::PermissionsExt;

        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("artifact.json");

        atomic_write_json(&path, &serde_json::json!({"private": true}))
            .expect("write private artifact");

        assert_eq!(
            std::fs::metadata(&path)
                .expect("artifact metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }

    #[test]
    fn atomic_write_never_exposes_partial_replacements() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let root = tempdir().expect("create temporary directory");
        let path = root.path().join("artifact.bin");
        let old: Arc<[u8]> = vec![b'A'; 256 * 1024].into();
        let new: Arc<[u8]> = vec![b'B'; 384 * 1024].into();
        atomic_write(&path, old.as_ref()).expect("write initial artifact");

        let writer_path = path.clone();
        let writer_old = Arc::clone(&old);
        let writer_new = Arc::clone(&new);
        let start = Arc::new(Barrier::new(2));
        let writer_start = Arc::clone(&start);
        let writer = thread::spawn(move || {
            writer_start.wait();
            for iteration in 0..64 {
                let bytes = if iteration % 2 == 0 {
                    writer_new.as_ref()
                } else {
                    writer_old.as_ref()
                };
                atomic_write(&writer_path, bytes).expect("replace artifact atomically");
            }
        });

        start.wait();
        for _ in 0..256 {
            let observed = std::fs::read(&path).expect("read visible artifact");
            assert!(
                observed.as_slice() == old.as_ref() || observed.as_slice() == new.as_ref(),
                "reader observed a partial or combined artifact of {} bytes",
                observed.len()
            );
        }
        writer.join().expect("writer thread");
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
