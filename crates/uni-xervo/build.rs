//! Build script. Single responsibility: feature-flag validation.
//!
//! Enforces:
//! - Mutual exclusion between the two ONNX linking modes (`provider-onnx`
//!   vs `provider-onnx-dynamic`). ort itself would eventually error with a
//!   confusing low-level message; we catch it here with a clear one.
//! - Target-OS guards on platform-specific GPU features.

fn main() {
    feature_guards();
    println!("cargo:rerun-if-changed=build.rs");
}

fn feature_guards() {
    let bundled = std::env::var("CARGO_FEATURE_PROVIDER_ONNX").is_ok();
    let dynamic = std::env::var("CARGO_FEATURE_PROVIDER_ONNX_DYNAMIC").is_ok();

    if bundled && dynamic {
        panic!(
            "Both `provider-onnx` (bundled, statically linked) and \
             `provider-onnx-dynamic` (load-dynamic, BYO tarball) are \
             enabled simultaneously. These are mutually exclusive at the \
             ort crate level — pick one based on your deployment model:\n  \
             - `provider-onnx`         : self-contained CPU binary (default).\n  \
             - `gpu-cuda`              : self-contained NVIDIA binary.\n  \
             - `provider-onnx-dynamic` : load-dynamic, supply ORT_DYLIB_PATH at runtime."
        );
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if std::env::var("CARGO_FEATURE_GPU_METAL").is_ok()
        && !["macos", "ios"].contains(&target_os.as_str())
    {
        panic!(
            "The `gpu-metal` feature is only supported on macOS / iOS targets.\n\
             Reason: candle-metal-kernels and the Objective-C bindings only \
             build on Apple targets, and the CoreML execution provider links \
             Apple-only frameworks.\n\
             Remove `gpu-metal` from your feature list when building for `{target_os}`."
        );
    }
}
