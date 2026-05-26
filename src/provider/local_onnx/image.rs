// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! Shared image preprocessing helpers for `local/onnx` tasks that consume
//! [`ImageInput`].
//!
//! Used by `image_embed.rs` (SigLIP-2-style ViT embedders), `ocr.rs` (PR-2b),
//! and `document_extract.rs` (PR-5). Decode → resize → normalize → emit a
//! `[batch, 3, H, W]` (NCHW) `Array4<f32>` tensor.
//!
//! The provider impls do not perform file I/O — `ImageInput::Url` is rejected
//! here with a clear error, matching the policy used elsewhere (callers fetch
//! URLs upstream and pass `Bytes`).

use crate::error::{Result, RuntimeError};
use crate::traits::ImageInput;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;

/// Per-channel normalization constants. The two common conventions:
///
/// - **SigLIP / SigLIP-2**: `mean = std = (0.5, 0.5, 0.5)` (i.e. centre at 0,
///   scale to `[-1, 1]`).
/// - **ImageNet**: `mean = (0.485, 0.456, 0.406)`, `std = (0.224, 0.225, 0.225)`.
///
/// Callers pass whichever set their model was trained on.
#[derive(Debug, Clone, Copy)]
pub struct Normalization {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Normalization {
    /// SigLIP / SigLIP-2 default — `(0.5, 0.5, 0.5)` for both mean and std.
    pub const SIGLIP: Self = Self {
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
    };

    /// Classic ImageNet stats — used by most ViT/CLIP-style models that
    /// weren't trained with SigLIP's normalization.
    pub const IMAGENET: Self = Self {
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };
}

/// Preprocess one batch of [`ImageInput`]s into a `[batch, 3, H, W]` float
/// tensor with the given target size and normalization.
///
/// Steps per image:
/// 1. Decode bytes (PNG/JPEG/WebP per the `image` crate's defaults).
/// 2. Convert to RGB8 — alpha is dropped, paletted images are expanded.
/// 3. Resize to `target_size × target_size` with Lanczos3 (high quality).
/// 4. Scale pixels to `[0, 1]` then `(pixel - mean) / std` per channel.
///
/// # Errors
/// Returns an error if the input is `ImageInput::Url` (not supported — see
/// module doc), or if decoding fails.
pub fn preprocess_batch(
    images: &[ImageInput],
    target_size: u32,
    norm: Normalization,
) -> Result<Array4<f32>> {
    let mut tensor =
        Array4::<f32>::zeros((images.len(), 3, target_size as usize, target_size as usize));

    for (i, input) in images.iter().enumerate() {
        let bytes = match input {
            ImageInput::Bytes { data, .. } => data,
            ImageInput::Url(url) => {
                return Err(RuntimeError::Config(format!(
                    "local/onnx image preprocessing does not fetch URLs; \
                     fetch '{url}' upstream and pass the bytes via \
                     ImageInput::Bytes"
                )));
            }
        };

        let img =
            image::load_from_memory(bytes).map_err(|e| RuntimeError::OnnxInvocationFailure {
                alias: "local/onnx".to_string(),
                cause: format!("Image decode failed at index {i}: {e}"),
            })?;
        let resized = resize_to_square(&img, target_size);
        let rgb = resized.to_rgb8();

        for (x, y, pixel) in rgb.enumerate_pixels() {
            let xi = x as usize;
            let yi = y as usize;
            // image::Rgb<u8> uses [r, g, b].
            for c in 0..3 {
                let raw = pixel.0[c] as f32 / 255.0;
                let normalized = (raw - norm.mean[c]) / norm.std[c];
                tensor[[i, c, yi, xi]] = normalized;
            }
        }
    }

    Ok(tensor)
}

/// Resize an image to a square of `target` pixels using Lanczos3. Non-square
/// inputs are stretched (no letterboxing) — matches the convention used by
/// SigLIP / CLIP exports.
fn resize_to_square(img: &DynamicImage, target: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w == target && h == target {
        return img.clone();
    }
    img.resize_exact(target, target, FilterType::Lanczos3)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiny_png() -> Vec<u8> {
        // 2x2 RGB image, alternating red/green/blue/white. Encode to PNG.
        let mut buf: image::RgbImage = image::ImageBuffer::new(2, 2);
        buf.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        buf.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        buf.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        buf.put_pixel(1, 1, image::Rgb([255, 255, 255]));
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode test PNG");
        out
    }

    #[test]
    fn preprocess_emits_correct_shape() {
        let inputs = vec![ImageInput::Bytes {
            data: make_tiny_png(),
            media_type: "image/png".to_string(),
        }];
        let tensor = preprocess_batch(&inputs, 8, Normalization::SIGLIP).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 8, 8]);
    }

    #[test]
    fn preprocess_normalizes_per_channel() {
        let inputs = vec![ImageInput::Bytes {
            data: make_tiny_png(),
            media_type: "image/png".to_string(),
        }];
        let tensor = preprocess_batch(&inputs, 2, Normalization::SIGLIP).unwrap();
        // Resize a 2x2 to 2x2 = identity. Pixel (0,0) is red = (255,0,0).
        // After /255 and (- 0.5) / 0.5: r=1.0, g=-1.0, b=-1.0.
        assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((tensor[[0, 1, 0, 0]] + 1.0).abs() < 1e-5);
        assert!((tensor[[0, 2, 0, 0]] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn preprocess_rejects_url_inputs() {
        let inputs = vec![ImageInput::Url("https://example.com/x.png".to_string())];
        let res = preprocess_batch(&inputs, 8, Normalization::SIGLIP);
        assert!(matches!(res, Err(RuntimeError::Config(_))));
    }

    #[test]
    fn preprocess_handles_empty_batch() {
        let tensor = preprocess_batch(&[], 8, Normalization::SIGLIP).unwrap();
        assert_eq!(tensor.shape(), &[0, 3, 8, 8]);
    }
}
