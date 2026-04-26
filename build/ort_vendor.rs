//! ONNX Runtime vendor fetcher.
//!
//! Activated when the user enables one of the Microsoft-distributed GPU
//! features that pyke doesn't ship a prebuilt for. Currently:
//!
//! - `gpu-directml` — Microsoft's `onnxruntime-win-x64-gpu-*.zip` (DirectML
//!   provider DLLs are bundled with the GPU build).
//! - `gpu-qnn` — Microsoft's Windows ARM64x prebuilt (subject to MS shipping
//!   it in the target ORT version; verified per release).
//!
//! Vendors that are NOT bundled (no source we can fetch from):
//!
//! - `gpu-rocm` — AMD's `onnxruntime-rocm` package, distributed separately;
//!   user must use `provider-onnx-dynamic` and supply the dylib.
//! - `gpu-openvino` — Intel's `openvino-onnxruntime` package, same situation.
//!
//! For the un-bundleable vendors we emit a clear `cargo:warning=` and require
//! `provider-onnx-dynamic` to be active too — the build still produces a
//! load-dynamic binary, the user just has to manage the runtime tarball.
//!
//! Pyke-handled vendors (`gpu-cuda`, `gpu-tensorrt`, `gpu-coreml`, `gpu-wgpu`)
//! flow through ort-sys's `download-binaries` pipeline + `ort/copy-dylibs`
//! and don't need anything from us — this module no-ops for them.

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const ORT_VERSION: &str = "1.25.0";

/// Vendor distribution descriptor — one entry per (vendor × target).
struct VendorSpec {
    /// Cargo feature env var that activates this vendor (e.g. `CARGO_FEATURE_GPU_DIRECTML`).
    feature_env: &'static str,
    /// Target triple substring this entry applies to.
    target_match: TargetMatch,
    /// Microsoft release artifact filename (under
    /// `https://github.com/microsoft/onnxruntime/releases/download/v<ORT_VERSION>/`).
    artifact: &'static str,
    /// Pinned SHA256 of the artifact, lowercase hex. Empty `""` skips
    /// verification (used during initial development; production specs MUST
    /// have a pinned hash).
    sha256: &'static str,
    /// Format of the downloaded artifact.
    format: ArchiveFormat,
    /// Subdirectory inside the extracted tarball that contains the dylibs.
    /// For MS Linux releases this is `lib`; for Windows it varies.
    libs_subdir: &'static str,
}

#[derive(Clone, Copy)]
enum TargetMatch {
    /// Match if `(target_os, target_arch) == (os, arch)`.
    OsArch(&'static str, &'static str),
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // TarGz reserved for future Linux-targeted MS-fetched specs.
enum ArchiveFormat {
    TarGz,
    Zip,
}

const VENDOR_SPECS: &[VendorSpec] = &[
    // DirectML — bundled inside MS's Windows x64 GPU tarball.
    VendorSpec {
        feature_env: "CARGO_FEATURE_GPU_DIRECTML",
        target_match: TargetMatch::OsArch("windows", "x86_64"),
        artifact: "onnxruntime-win-x64-gpu-1.25.0.zip",
        sha256: "", // TODO(0.6.0): pin once verified locally
        format: ArchiveFormat::Zip,
        libs_subdir: "lib",
    },
    // QNN — Windows ARM64x. Ships in the arm64x bundle.
    VendorSpec {
        feature_env: "CARGO_FEATURE_GPU_QNN",
        target_match: TargetMatch::OsArch("windows", "aarch64"),
        artifact: "onnxruntime-win-arm64x-1.25.0.zip",
        sha256: "", // TODO(0.6.0): pin once verified locally
        format: ArchiveFormat::Zip,
        libs_subdir: "lib",
    },
];

/// Vendors with no Microsoft-distributed prebuilt. These are valid feature
/// activations but require `provider-onnx-dynamic` (user supplies the
/// vendor's runtime). build.rs emits a `cargo:warning=` if activated without
/// `provider-onnx-dynamic` so the user knows the deployment expectation.
const VENDORS_NO_BINARY_DIST: &[(&str, &str, &str)] = &[
    (
        "CARGO_FEATURE_GPU_ROCM",
        "gpu-rocm",
        "AMD's onnxruntime-rocm package (https://repo.radeon.com/rocm/)",
    ),
    (
        "CARGO_FEATURE_GPU_OPENVINO",
        "gpu-openvino",
        "Intel's openvino-onnxruntime package (https://docs.openvino.ai/)",
    ),
];

/// Entry point — called from `build.rs::main`.
///
/// Walks `VENDOR_SPECS`, finds the active spec (at most one — mutual
/// exclusion is already enforced in `build.rs::feature_guards`), and runs
/// the fetch + stage pipeline. Returns `Ok(())` when no MS-fetched vendor
/// is active (the common case for `provider-onnx` / `gpu-cuda` /
/// `gpu-tensorrt` / `gpu-coreml` / `gpu-wgpu` / `provider-onnx-dynamic`).
pub fn fetch_and_stage() -> Result<(), String> {
    // Warn for un-bundleable vendors. These features compile in the right
    // ort EP type but the user must supply the ONNX Runtime shared library
    // at deploy time via ORT_DYLIB_PATH (no central distribution we can
    // fetch from).
    for (env_var, feature, source_doc) in VENDORS_NO_BINARY_DIST {
        if env::var(env_var).is_ok() {
            println!(
                "cargo:warning=`{feature}`: no prebuilt ONNX Runtime distribution exists for this vendor."
            );
            println!("cargo:warning=  Source: {source_doc}");
            println!(
                "cargo:warning=  Deploy: set ORT_DYLIB_PATH to the vendor's libonnxruntime.so."
            );
        }
    }

    // CUDA-specific runtime advisory. Pyke's CUDA EP sidecars depend on
    // cuDNN at runtime (`libcudnn.so.9` on Linux), which is part of the
    // CUDA Toolkit / cuDNN package — not something we can or should bundle.
    // Users typically have it via their NVIDIA driver install; if not,
    // they need to add it to LD_LIBRARY_PATH or symlink it next to the
    // binary.
    if env::var("CARGO_FEATURE_GPU_CUDA").is_ok() {
        println!(
            "cargo:warning=gpu-cuda: NVIDIA driver + cuDNN must be on the host's loader path."
        );
        println!("cargo:warning=  cuDNN is typically at /usr/local/cuda-*/targets/<arch>/lib/.");
        println!("cargo:warning=  If `libcudnn.so.9: cannot open shared object file` at runtime,");
        println!(
            "cargo:warning=  add the cuDNN path to LD_LIBRARY_PATH or install the cuDNN package."
        );
    }

    // Pick the active spec for the current target.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let active_spec = VENDOR_SPECS.iter().find(|spec| {
        if env::var(spec.feature_env).is_err() {
            return false;
        }
        match spec.target_match {
            TargetMatch::OsArch(os, arch) => target_os == os && target_arch == arch,
        }
    });

    let Some(spec) = active_spec else {
        // No MS-fetched vendor active for this target — done.
        return Ok(());
    };

    // Bypass: pre-populated cache directory.
    if let Ok(dir) = env::var("UNI_XERVO_ORT_DIST_DIR") {
        let extracted = PathBuf::from(dir);
        if !extracted.exists() {
            return Err(format!(
                "UNI_XERVO_ORT_DIST_DIR points at non-existent path: {}",
                extracted.display()
            ));
        }
        emit_link_directives(&extracted.join(spec.libs_subdir))?;
        return Ok(());
    }

    // Fetch + verify + extract pipeline.
    let cache_dir = uni_xervo_cache_dir();
    fs::create_dir_all(&cache_dir).map_err(|e| format!("create cache dir: {e}"))?;

    let archive_path = cache_dir.join(spec.artifact);
    if !archive_path.exists() {
        download_artifact(spec, &archive_path)?;
    }

    if !spec.sha256.is_empty() {
        verify_sha256(&archive_path, spec.sha256)?;
    }

    let extracted_dir = cache_dir.join(format!("{}-{}", archive_stem(spec.artifact), ORT_VERSION));
    if !extracted_dir.exists() {
        extract_archive(spec, &archive_path, &extracted_dir)?;
    }

    let lib_dir = locate_lib_dir(&extracted_dir, spec.libs_subdir)?;
    emit_link_directives(&lib_dir)?;
    Ok(())
}

/// Cache directory for downloaded ONNX Runtime artifacts. Shared across all
/// crates in the workspace so multiple builds don't re-download.
fn uni_xervo_cache_dir() -> PathBuf {
    if let Ok(dir) = env::var("UNI_XERVO_ORT_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = dirs_home_dir() {
        return home
            .join(".cache")
            .join("uni-xervo")
            .join("ort")
            .join(ORT_VERSION);
    }
    // Fallback: target dir.
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    PathBuf::from(out_dir)
        .join("uni-xervo-ort-cache")
        .join(ORT_VERSION)
}

/// Minimal home-dir lookup that doesn't require a `dirs` crate dep.
fn dirs_home_dir() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        env::var("USERPROFILE").ok().map(PathBuf::from)
    } else {
        env::var("HOME").ok().map(PathBuf::from)
    }
}

/// Strip the trailing `.tgz` / `.zip` from an artifact filename.
fn archive_stem(name: &str) -> String {
    name.trim_end_matches(".tgz")
        .trim_end_matches(".tar.gz")
        .trim_end_matches(".zip")
        .to_string()
}

fn download_artifact(spec: &VendorSpec, dest: &Path) -> Result<(), String> {
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{}/{}",
        ORT_VERSION, spec.artifact
    );
    println!("cargo:warning=uni-xervo: downloading {url}");

    let response = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(900)) // 15 min for big GPU tarballs
        .call()
        .map_err(|e| format!("HTTP fetch of {url}: {e}"))?;

    let tmp = dest.with_extension("tmp");
    let mut writer = fs::File::create(&tmp).map_err(|e| format!("create tmp file: {e}"))?;
    let mut reader = response.into_reader();
    io::copy(&mut reader, &mut writer).map_err(|e| format!("write tmp file: {e}"))?;
    writer.flush().ok();
    fs::rename(&tmp, dest).map_err(|e| format!("rename tmp -> dest: {e}"))?;
    Ok(())
}

fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = hex::encode(hasher.finalize());
    if actual.to_lowercase() != expected_hex.to_lowercase() {
        return Err(format!(
            "SHA256 mismatch for {path:?}: expected {expected_hex}, got {actual}"
        ));
    }
    Ok(())
}

fn extract_archive(spec: &VendorSpec, archive: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|e| format!("create extract dir: {e}"))?;
    match spec.format {
        ArchiveFormat::TarGz => extract_tar_gz(archive, dest),
        ArchiveFormat::Zip => extract_zip(archive, dest),
    }
}

fn extract_tar_gz(archive: &Path, dest: &Path) -> Result<(), String> {
    let file = fs::File::open(archive).map_err(|e| format!("open archive: {e}"))?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(gz);
    tar.unpack(dest).map_err(|e| format!("untar: {e}"))?;
    Ok(())
}

fn extract_zip(archive: &Path, dest: &Path) -> Result<(), String> {
    let file = fs::File::open(archive).map_err(|e| format!("open archive: {e}"))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("open zip: {e}"))?;
    zip.extract(dest).map_err(|e| format!("unzip: {e}"))?;
    Ok(())
}

/// MS tarballs nest everything inside a single top-level dir matching the
/// archive stem. Handle that automatically.
fn locate_lib_dir(extracted: &Path, libs_subdir: &str) -> Result<PathBuf, String> {
    // Try direct `<extracted>/lib`.
    let direct = extracted.join(libs_subdir);
    if direct.exists() {
        return Ok(direct);
    }
    // Try `<extracted>/<single-subdir>/lib`.
    if let Ok(entries) = fs::read_dir(extracted) {
        for entry in entries.flatten() {
            let candidate = entry.path().join(libs_subdir);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }
    Err(format!(
        "could not find lib subdirectory `{libs_subdir}` under {extracted:?}"
    ))
}

/// Emit cargo: directives so the binary links against and finds the dylibs
/// at runtime.
fn emit_link_directives(lib_dir: &Path) -> Result<(), String> {
    if !lib_dir.exists() {
        return Err(format!("lib dir does not exist: {lib_dir:?}"));
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Embed an absolute rpath into the binary so it finds the dylibs at
    // runtime without any env-var setup. Works for `cargo run/test/install`
    // on the same host. For cross-host deployment, users run the
    // `uni-xervo-bundle` helper to produce a $ORIGIN-based portable tree.
    let rpath_arg = format!("-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rustc-link-arg={rpath_arg}");

    // Surface the path to library code at runtime via env-var.
    println!(
        "cargo:rustc-env=UNI_XERVO_ORT_RUNTIME_DIR={}",
        lib_dir.display()
    );
    Ok(())
}
