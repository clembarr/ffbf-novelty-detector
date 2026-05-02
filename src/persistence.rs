use std::fs;
use std::path::Path;

use crate::filter::FFBF;

/// Serialize `ffbf` to a JSON string.
///
/// Returns:
///   `Ok(String)` with the JSON representation.
/// Errors:
///   `Err(String)` if serialization fails.
pub fn to_json(ffbf: &FFBF) -> Result<String, String> {
    serde_json::to_string(ffbf).map_err(|e| e.to_string())
}

/// Deserialize an `FFBF` from a JSON string.
///
/// Parameters:
///   s: JSON string previously produced by `to_json` or `save`.
/// Returns:
///   `Ok(FFBF)` on success.
/// Errors:
///   `Err(String)` if the input is malformed or fields are out of range.
pub fn from_json(s: &str) -> Result<FFBF, String> {
    serde_json::from_str(s).map_err(|e| e.to_string())
}

/// Write `ffbf` as JSON to `path`, creating or overwriting the file.
///
/// Parameters:
///   ffbf: Filter to persist.
///   path: Destination file path.
/// Errors:
///   `Err(String)` on serialization or I/O failure.
pub fn save(ffbf: &FFBF, path: impl AsRef<Path>) -> Result<(), String> {
    let json = to_json(ffbf)?;
    fs::write(path, json).map_err(|e| e.to_string())
}

/// Load an `FFBF` from a JSON file at `path`.
///
/// Parameters:
///   path: Path to a file written by `save`.
/// Returns:
///   `Ok(FFBF)` on success.
/// Errors:
///   `Err(String)` on I/O or deserialization failure.
pub fn load(path: impl AsRef<Path>) -> Result<FFBF, String> {
    let json = fs::read_to_string(path).map_err(|e| e.to_string())?;
    from_json(&json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FFBFConfig;

    fn make_ffbf() -> FFBF {
        let mut cfg = FFBFConfig::default_for(32, 20);
        cfg.seed = Some(7);
        FFBF::new(cfg).unwrap()
    }

    #[test]
    fn roundtrip_json() {
        let mut ffbf = make_ffbf();
        ffbf.add(&vec![0.5f32; ffbf.weights().len().min(32)]);
        let json = to_json(&ffbf).unwrap();
        let restored = from_json(&json).unwrap();
        assert_eq!(ffbf.weights(), restored.weights());
        assert_eq!(ffbf.window_len(), restored.window_len());
    }

    #[test]
    fn restored_filter_behaves_identically() {
        let mut ffbf = make_ffbf();
        let input = vec![1.0f32; 32];
        for _ in 0..5 {
            ffbf.add(&input);
        }
        let json = to_json(&ffbf).unwrap();
        let mut restored = from_json(&json).unwrap();
        //both should produce the same novelty and evolve identically
        assert!((ffbf.novelty(&input) - restored.novelty(&input)).abs() < 1e-6);
        ffbf.add(&input);
        restored.add(&input);
        assert_eq!(ffbf.weights(), restored.weights());
    }

    #[test]
    fn from_json_rejects_garbage() {
        assert!(from_json("not json").is_err());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let ffbf = make_ffbf();
        let path = std::env::temp_dir().join("ffbf_test.json");
        save(&ffbf, &path).unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(ffbf.weights(), loaded.weights());
        let _ = std::fs::remove_file(path);
    }
}
