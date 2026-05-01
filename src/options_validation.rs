//! Compile-time validation of provider-specific options JSON.
//!
//! Called during [`ModelRuntimeBuilder::build`](crate::runtime::ModelRuntimeBuilder::build) and
//! [`ModelRuntime::register`](crate::runtime::ModelRuntime::register) to reject
//! unknown or malformed options before any model loading occurs.

use crate::api::ModelTask;
use crate::error::{Result, RuntimeError};
use serde_json::Value;

/// Validate provider-specific options for the given `provider_id` and `task`.
///
/// Returns `Ok(())` if the options are valid or the provider is unknown (unknown
/// providers are silently accepted to allow third-party extensions).
pub fn validate_provider_options(
    provider_id: &str,
    task: ModelTask,
    options: &Value,
) -> Result<()> {
    match provider_id {
        "remote/openai" => validate_openai_options(provider_id, task, options),
        "remote/mistral" | "remote/voyageai" => {
            validate_with_embedding_dimensions(provider_id, task, options, &["api_key_env"])
        }
        "remote/gemini" => validate_with_embedding_dimensions(
            provider_id,
            task,
            options,
            &["api_key_env", "api_version"],
        ),
        "remote/anthropic" => {
            validate_string_keys_only(provider_id, options, &["api_key_env", "anthropic_version"])
        }
        "remote/cohere" => validate_with_embedding_dimensions(
            provider_id,
            task,
            options,
            &["api_key_env", "input_type"],
        ),
        "remote/azure-openai" => validate_azure_openai_options(provider_id, task, options),
        "remote/vertexai" => validate_vertexai_options(provider_id, task, options),
        "local/candle" => validate_string_keys_only(provider_id, options, &["cache_dir"]),
        "local/onnx" => validate_local_onnx_options(provider_id, task, options),
        "local/mistralrs" => validate_mistralrs_options(provider_id, task, options),
        _ => Ok(()),
    }
}

/// Parse `options` as a JSON object map, returning `None` for null and an
/// error for non-object types.
fn as_object<'a>(
    provider_id: &str,
    options: &'a Value,
) -> Result<Option<&'a serde_json::Map<String, Value>>> {
    match options {
        Value::Null => Ok(None),
        Value::Object(map) => Ok(Some(map)),
        _ => Err(RuntimeError::Config(format!(
            "Options for provider '{}' must be a JSON object or null",
            provider_id
        ))),
    }
}

/// Return an error if `map` contains any key not in `allowed`.
fn reject_unknown_keys(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    allowed: &[&str],
) -> Result<()> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(RuntimeError::Config(format!(
                "Unknown option '{}' for provider '{}'",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// Require that all specified keys, if present, are strings.
fn require_string_keys(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Result<()> {
    for key in keys {
        if let Some(value) = map.get(*key)
            && !value.is_string()
        {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a string",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// Require that the named key, if present, is a positive (> 0) integer.
fn require_positive_u64(
    provider_id: &str,
    map: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<()> {
    if let Some(value) = map.get(key) {
        let Some(v) = value.as_u64() else {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be a positive integer",
                key, provider_id
            )));
        };
        if v == 0 {
            return Err(RuntimeError::Config(format!(
                "Option '{}' for provider '{}' must be greater than 0",
                key, provider_id
            )));
        }
    }
    Ok(())
}

/// Validate that the embedding_dimensions option is a positive integer and only
/// used for embed tasks.
fn require_embedding_dimensions(
    provider_id: &str,
    task: ModelTask,
    map: &serde_json::Map<String, Value>,
) -> Result<()> {
    if map.contains_key("embedding_dimensions") {
        require_positive_u64(provider_id, map, "embedding_dimensions")?;
        if task != ModelTask::Embed {
            return Err(RuntimeError::Config(
                "Option 'embedding_dimensions' is only valid for embed tasks".to_string(),
            ));
        }
    }
    Ok(())
}

/// Validate providers whose options are all optional string keys.
fn validate_string_keys_only(
    provider_id: &str,
    options: &Value,
    allowed_keys: &[&str],
) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    reject_unknown_keys(provider_id, map, allowed_keys)?;
    require_string_keys(provider_id, map, allowed_keys)
}

/// Validate providers that accept string keys plus an optional
/// `embedding_dimensions` integer.
fn validate_with_embedding_dimensions(
    provider_id: &str,
    task: ModelTask,
    options: &Value,
    string_keys: &[&str],
) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    let mut all_keys: Vec<&str> = string_keys.to_vec();
    all_keys.push("embedding_dimensions");
    reject_unknown_keys(provider_id, map, &all_keys)?;
    require_string_keys(provider_id, map, string_keys)?;
    require_embedding_dimensions(provider_id, task, map)
}

/// Validate OpenAI options: `api_key_env`, optional `base_url`, and optional
/// `embedding_dimensions`. The `base_url` option lets users target OpenAI-
/// compatible servers (OpenRouter, vLLM, LM Studio, Ollama, internal proxies).
fn validate_openai_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    reject_unknown_keys(
        provider_id,
        map,
        &["api_key_env", "base_url", "embedding_dimensions"],
    )?;
    require_string_keys(provider_id, map, &["api_key_env", "base_url"])?;
    require_embedding_dimensions(provider_id, task, map)
}

/// Validate Azure OpenAI options: string keys, `api_version`, and optional
/// `embedding_dimensions`.
fn validate_azure_openai_options(
    provider_id: &str,
    task: ModelTask,
    options: &Value,
) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    reject_unknown_keys(
        provider_id,
        map,
        &[
            "api_key_env",
            "resource_name",
            "api_version",
            "embedding_dimensions",
        ],
    )?;
    require_string_keys(
        provider_id,
        map,
        &["api_key_env", "resource_name", "api_version"],
    )?;
    require_embedding_dimensions(provider_id, task, map)
}

/// Validate Vertex AI-specific options: string keys plus optional
/// `embedding_dimensions`.
fn validate_vertexai_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };
    reject_unknown_keys(
        provider_id,
        map,
        &[
            "api_token_env",
            "project_id",
            "location",
            "publisher",
            "embedding_dimensions",
        ],
    )?;
    require_string_keys(
        provider_id,
        map,
        &["api_token_env", "project_id", "location", "publisher"],
    )?;
    require_embedding_dimensions(provider_id, task, map)
}

/// Validate mistral.rs-specific options: ISQ type, boolean flags, GGUF files,
/// pipeline selection, and embedding dimensions.
fn validate_mistralrs_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };

    // All known keys across all pipelines
    reject_unknown_keys(
        provider_id,
        map,
        &[
            "isq",
            "force_cpu",
            "paged_attention",
            "max_num_seqs",
            "chat_template",
            "tokenizer_json",
            "embedding_dimensions",
            "gguf_files",
            "dtype",
            "pipeline",
            "diffusion_loader_type",
            "speech_loader_type",
            "max_seq_len",
            "max_batch_size",
            "max_image_shape",
            "max_num_images",
            "uqff_files",
        ],
    )?;

    // Validate pipeline value if present
    let pipeline = if let Some(value) = map.get("pipeline") {
        let s = value.as_str().ok_or_else(|| {
            RuntimeError::Config(format!(
                "Option 'pipeline' for provider '{}' must be a string",
                provider_id
            ))
        })?;
        let valid = ["text", "vision", "diffusion", "speech"];
        if !valid.contains(&s) {
            return Err(RuntimeError::Config(format!(
                "Option 'pipeline' for provider '{}' must be one of: text, vision, diffusion, speech",
                provider_id
            )));
        }
        s
    } else {
        "text"
    };

    // Shared validation across all pipelines
    require_string_keys(provider_id, map, &["dtype"])?;

    if let Some(value) = map.get("dtype") {
        if let Some(s) = value.as_str() {
            let valid = ["auto", "f16", "bf16", "f32"];
            if !valid.contains(&s.to_lowercase().as_str()) {
                return Err(RuntimeError::Config(format!(
                    "Option 'dtype' for provider '{}' must be one of: auto, f16, bf16, f32",
                    provider_id
                )));
            }
        }
    }

    if let Some(value) = map.get("force_cpu")
        && !value.is_boolean()
    {
        return Err(RuntimeError::Config(format!(
            "Option 'force_cpu' for provider '{}' must be a boolean",
            provider_id
        )));
    }

    // Auto-device-mapper overrides (positive integers, applicable to text,
    // vision, and embedding pipelines; rejected for diffusion/speech below).
    require_positive_u64(provider_id, map, "max_seq_len")?;
    require_positive_u64(provider_id, map, "max_batch_size")?;
    require_positive_u64(provider_id, map, "max_num_images")?;

    if let Some(value) = map.get("max_image_shape") {
        let items = value.as_array().ok_or_else(|| {
            RuntimeError::Config(format!(
                "Option 'max_image_shape' for provider '{}' must be a 2-element array of positive integers",
                provider_id
            ))
        })?;
        if items.len() != 2
            || items
                .iter()
                .any(|v| v.as_u64().is_none_or(|n| n == 0 || n > usize::MAX as u64))
        {
            return Err(RuntimeError::Config(format!(
                "Option 'max_image_shape' for provider '{}' must be a 2-element array of positive integers",
                provider_id
            )));
        }
    }

    if let Some(value) = map.get("uqff_files") {
        let Some(items) = value.as_array() else {
            return Err(RuntimeError::Config(format!(
                "Option 'uqff_files' for provider '{}' must be a non-empty array of strings",
                provider_id
            )));
        };
        if items.is_empty()
            || items
                .iter()
                .any(|item| !item.is_string() || item.as_str().unwrap().is_empty())
        {
            return Err(RuntimeError::Config(format!(
                "Option 'uqff_files' for provider '{}' must be a non-empty array of strings",
                provider_id
            )));
        }
        if map.contains_key("gguf_files") {
            return Err(RuntimeError::Config(
                "Options 'uqff_files' and 'gguf_files' are mutually exclusive (both load pre-quantized weights)"
                    .to_string(),
            ));
        }
        if map.contains_key("isq") {
            return Err(RuntimeError::Config(
                "Option 'isq' is incompatible with 'uqff_files' (UQFF files embed their own quantization)"
                    .to_string(),
            ));
        }
    }

    // Pipeline-specific validation
    match pipeline {
        "vision" => {
            if map.contains_key("gguf_files") {
                return Err(RuntimeError::Config(
                    "Option 'gguf_files' is not supported for the vision pipeline".to_string(),
                ));
            }
            if map.contains_key("embedding_dimensions") {
                return Err(RuntimeError::Config(
                    "Option 'embedding_dimensions' is not supported for the vision pipeline"
                        .to_string(),
                ));
            }
            require_string_keys(
                provider_id,
                map,
                &["isq", "chat_template", "tokenizer_json"],
            )?;
            if let Some(value) = map.get("paged_attention")
                && !value.is_boolean()
            {
                return Err(RuntimeError::Config(format!(
                    "Option 'paged_attention' for provider '{}' must be a boolean",
                    provider_id
                )));
            }
            require_positive_u64(provider_id, map, "max_num_seqs")?;
        }
        "diffusion" => {
            // Validate diffusion_loader_type
            if let Some(value) = map.get("diffusion_loader_type") {
                let s = value.as_str().ok_or_else(|| {
                    RuntimeError::Config(format!(
                        "Option 'diffusion_loader_type' for provider '{}' must be a string",
                        provider_id
                    ))
                })?;
                let valid = ["flux", "flux_offloaded"];
                if !valid.contains(&s) {
                    return Err(RuntimeError::Config(format!(
                        "Option 'diffusion_loader_type' for provider '{}' must be one of: flux, flux_offloaded",
                        provider_id
                    )));
                }
            }
            // Reject options not relevant to diffusion
            for key in [
                "isq",
                "paged_attention",
                "max_num_seqs",
                "chat_template",
                "tokenizer_json",
                "embedding_dimensions",
                "gguf_files",
                "uqff_files",
                "speech_loader_type",
                "max_seq_len",
                "max_batch_size",
                "max_image_shape",
                "max_num_images",
            ] {
                if map.contains_key(key) {
                    return Err(RuntimeError::Config(format!(
                        "Option '{}' is not supported for the diffusion pipeline",
                        key
                    )));
                }
            }
        }
        "speech" => {
            // Validate speech_loader_type
            if let Some(value) = map.get("speech_loader_type") {
                let s = value.as_str().ok_or_else(|| {
                    RuntimeError::Config(format!(
                        "Option 'speech_loader_type' for provider '{}' must be a string",
                        provider_id
                    ))
                })?;
                let valid = ["dia"];
                if !valid.contains(&s) {
                    return Err(RuntimeError::Config(format!(
                        "Option 'speech_loader_type' for provider '{}' must be one of: dia",
                        provider_id
                    )));
                }
            }
            // Reject options not relevant to speech
            for key in [
                "isq",
                "paged_attention",
                "max_num_seqs",
                "chat_template",
                "tokenizer_json",
                "embedding_dimensions",
                "gguf_files",
                "uqff_files",
                "diffusion_loader_type",
                "max_seq_len",
                "max_batch_size",
                "max_image_shape",
                "max_num_images",
            ] {
                if map.contains_key(key) {
                    return Err(RuntimeError::Config(format!(
                        "Option '{}' is not supported for the speech pipeline",
                        key
                    )));
                }
            }
        }
        _ => {
            // text pipeline (default) — full validation
            require_string_keys(
                provider_id,
                map,
                &["isq", "chat_template", "tokenizer_json"],
            )?;

            for key in ["max_image_shape", "max_num_images"] {
                if map.contains_key(key) {
                    return Err(RuntimeError::Config(format!(
                        "Option '{}' is only supported for the vision pipeline",
                        key
                    )));
                }
            }

            if let Some(value) = map.get("paged_attention")
                && !value.is_boolean()
            {
                return Err(RuntimeError::Config(format!(
                    "Option 'paged_attention' for provider '{}' must be a boolean",
                    provider_id
                )));
            }

            require_positive_u64(provider_id, map, "max_num_seqs")?;
            require_embedding_dimensions(provider_id, task, map)?;

            if let Some(value) = map.get("gguf_files") {
                let Some(items) = value.as_array() else {
                    return Err(RuntimeError::Config(format!(
                        "Option 'gguf_files' for provider '{}' must be an array of strings",
                        provider_id
                    )));
                };
                if items.iter().any(|item| !item.is_string()) {
                    return Err(RuntimeError::Config(format!(
                        "Option 'gguf_files' for provider '{}' must be an array of strings",
                        provider_id
                    )));
                }
            }
        }
    }

    Ok(())
}

fn validate_local_onnx_options(provider_id: &str, task: ModelTask, options: &Value) -> Result<()> {
    let Some(map) = as_object(provider_id, options)? else {
        return Ok(());
    };

    // Keys shared across every task. Per-task validators extend this list.
    let common_keys: &[&str] = &[
        "artifact",
        "max_batch_size",
        "execution_providers",
        "graph_optimization_level",
        "inter_op_num_threads",
        "intra_op_num_threads",
        "cache_dir",
    ];

    let allowed: Vec<&str> = match task {
        ModelTask::Embed => {
            let mut keys = common_keys.to_vec();
            keys.extend_from_slice(&[
                "tokenizer_path",
                "pooling",
                "normalize",
                "dimensions",
                "max_seq_len",
                "token_type_ids",
                "output_name",
            ]);
            keys
        }
        ModelTask::Rerank => {
            let mut keys = common_keys.to_vec();
            keys.push("max_seq_len");
            keys
        }
        _ => common_keys.to_vec(),
    };

    reject_unknown_keys(provider_id, map, &allowed)?;

    require_positive_u64(provider_id, map, "max_batch_size")?;
    require_positive_u64(provider_id, map, "inter_op_num_threads")?;
    require_positive_u64(provider_id, map, "intra_op_num_threads")?;
    require_string_keys(provider_id, map, &["artifact", "cache_dir"])?;

    if matches!(task, ModelTask::Embed) {
        require_string_keys(
            provider_id,
            map,
            &["tokenizer_path", "pooling", "output_name"],
        )?;
        require_positive_u64(provider_id, map, "dimensions")?;
        require_positive_u64(provider_id, map, "max_seq_len")?;

        if let Some(pooling) = map.get("pooling") {
            let Some(pooling) = pooling.as_str() else {
                return Err(RuntimeError::Config(format!(
                    "Option 'pooling' for provider '{}' must be a string",
                    provider_id
                )));
            };
            match pooling {
                "cls" | "mean" | "max" => {}
                _ => {
                    return Err(RuntimeError::Config(format!(
                        "Option 'pooling' for provider '{}' must be one of cls, mean, max",
                        provider_id
                    )));
                }
            }
        }

        if let Some(v) = map.get("normalize")
            && !v.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option 'normalize' for provider '{}' must be a boolean",
                provider_id
            )));
        }
        if let Some(v) = map.get("token_type_ids")
            && !v.is_boolean()
        {
            return Err(RuntimeError::Config(format!(
                "Option 'token_type_ids' for provider '{}' must be a boolean",
                provider_id
            )));
        }
    }

    if let Some(level) = map.get("graph_optimization_level") {
        let Some(level) = level.as_str() else {
            return Err(RuntimeError::Config(format!(
                "Option 'graph_optimization_level' for provider '{}' must be a string",
                provider_id
            )));
        };
        match level {
            "disable" | "basic" | "extended" | "all" => {}
            _ => {
                return Err(RuntimeError::Config(format!(
                    "Option 'graph_optimization_level' for provider '{}' must be one of disable, basic, extended, all",
                    provider_id
                )));
            }
        }
    }

    if let Some(execution_providers) = map.get("execution_providers") {
        validate_execution_provider_array(
            provider_id,
            execution_providers,
            &["cpu", "cuda", "coreml", "directml"],
        )?;
    }

    Ok(())
}

fn validate_execution_provider_array(
    provider_id: &str,
    value: &Value,
    allowed: &[&str],
) -> Result<()> {
    let Some(values) = value.as_array() else {
        return Err(RuntimeError::Config(format!(
            "Option 'execution_providers' for provider '{}' must be an array of strings",
            provider_id
        )));
    };
    for item in values {
        match item.as_str() {
            Some(name) if allowed.contains(&name) => {}
            Some(other) => {
                return Err(RuntimeError::Config(format!(
                    "Execution provider '{}' is not supported for provider '{}'",
                    other, provider_id
                )));
            }
            None => {
                return Err(RuntimeError::Config(format!(
                    "Option 'execution_providers' for provider '{}' must be an array of strings",
                    provider_id
                )));
            }
        }
    }
    Ok(())
}
