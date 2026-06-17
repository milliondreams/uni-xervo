// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Team

//! DBNet text-detection postprocessing — probability map to text boxes.
//!
//! Pure Rust, no ONNX/IO: a DBNet-style detector emits a per-pixel text
//! probability map; this module turns it into axis-aligned text boxes. The
//! steps mirror the standard PP-OCR fast-postprocessing (binarize → connected
//! components → box score → unclip), but ported with no OpenCV/clipper
//! dependency. Boxes are produced in *detector-map* pixel coordinates; the
//! caller scales them back to the original image via [`DetBox::scale_to`].
//!
//! Axis-aligned boxes (v1) handle horizontal document text — the common case.
//! Rotated min-area-rect + polygon unclip is a deliberate future refinement.

use ndarray::ArrayView2;

/// Tunable thresholds for DBNet box extraction.
///
/// Defaults match common PP-OCR settings; all are overridable via the OCR
/// alias options.
#[derive(Debug, Clone, Copy)]
pub(super) struct DetParams {
    /// Probability at/above which a pixel is foreground (text). Inclusive.
    pub bin_threshold: f32,
    /// Minimum mean component probability for a box to be kept.
    pub box_score_threshold: f32,
    /// Polygon dilation ratio applied during unclip (PP-OCR default 1.5).
    pub unclip_ratio: f32,
    /// Minimum box side, in detector-map pixels, below which a box is dropped.
    pub min_box_size: u32,
}

impl Default for DetParams {
    fn default() -> Self {
        Self {
            // PP-OCR defaults: a 0.3 segmentation threshold, 0.6 box score,
            // 1.5 unclip, and a 3px minimum side. Tuned per corpus if needed.
            bin_threshold: 0.3,
            box_score_threshold: 0.6,
            unclip_ratio: 1.5,
            min_box_size: 3,
        }
    }
}

/// An axis-aligned detected text box with its mean-probability score.
///
/// Coordinates are a half-open rectangle `[x0, x1) × [y0, y1)` in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DetBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

impl DetBox {
    /// Box width in pixels.
    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }

    /// Box height in pixels.
    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }

    /// Map this box from detector-map coords to original-image pixels.
    ///
    /// `scale_x`/`scale_y` are `det_dim / orig_dim` (as returned by
    /// `preprocess_det`), so original coordinates are `det / scale`. The result
    /// is floored on the top-left, ceiled on the bottom-right, and clamped to
    /// `[0, max_w] × [0, max_h]` so it never falls outside the source image.
    pub fn scale_to(&self, scale_x: f32, scale_y: f32, max_w: f32, max_h: f32) -> DetBox {
        DetBox {
            x0: (self.x0 / scale_x).floor().clamp(0.0, max_w),
            y0: (self.y0 / scale_y).floor().clamp(0.0, max_h),
            x1: (self.x1 / scale_x).ceil().clamp(0.0, max_w),
            y1: (self.y1 / scale_y).ceil().clamp(0.0, max_h),
            score: self.score,
        }
    }
}

/// Extract text boxes from a detector probability map.
///
/// Foreground is `prob >= bin_threshold`. Connected components are found with an
/// iterative 4-connectivity flood fill (no recursion — a fully-foreground map
/// cannot overflow the stack). Each component yields its axis-aligned bounding
/// box and a score equal to the mean probability over the component's pixels;
/// boxes below `box_score_threshold` or smaller than `min_box_size` on either
/// side are dropped. Surviving boxes are unclipped (dilated) and clamped to the
/// map bounds.
pub(super) fn postprocess(prob: ArrayView2<f32>, params: &DetParams) -> Vec<DetBox> {
    let h = prob.nrows();
    let w = prob.ncols();
    if h == 0 || w == 0 {
        return Vec::new();
    }
    let thr = params.bin_threshold;
    let mut visited = vec![false; h * w];
    let mut boxes = Vec::new();

    for sy in 0..h {
        for sx in 0..w {
            if visited[sy * w + sx] || prob[[sy, sx]] < thr {
                continue;
            }
            // Iterative flood fill over the connected foreground component.
            let mut stack = vec![(sy, sx)];
            visited[sy * w + sx] = true;
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (sx, sy, sx, sy);
            let mut sum = 0f64;
            let mut count = 0u64;

            while let Some((y, x)) = stack.pop() {
                sum += f64::from(prob[[y, x]]);
                count += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);

                // 4-connectivity: up, down, left, right.
                let neighbors = [
                    (y.wrapping_sub(1), x, y > 0),
                    (y + 1, x, y + 1 < h),
                    (y, x.wrapping_sub(1), x > 0),
                    (y, x + 1, x + 1 < w),
                ];
                for (ny, nx, in_bounds) in neighbors {
                    if !in_bounds {
                        continue;
                    }
                    let i = ny * w + nx;
                    if !visited[i] && prob[[ny, nx]] >= thr {
                        visited[i] = true;
                        stack.push((ny, nx));
                    }
                }
            }

            // Inclusive pixel extent -> side length is (max - min + 1).
            let box_w = (max_x - min_x + 1) as u32;
            let box_h = (max_y - min_y + 1) as u32;
            if box_w < params.min_box_size || box_h < params.min_box_size {
                continue;
            }
            let score = (sum / count as f64) as f32;
            if score < params.box_score_threshold {
                continue;
            }

            // Half-open box: x1 = max_x + 1, y1 = max_y + 1.
            let raw = DetBox {
                x0: min_x as f32,
                y0: min_y as f32,
                x1: (max_x + 1) as f32,
                y1: (max_y + 1) as f32,
                score,
            };
            boxes.push(unclip(&raw, params.unclip_ratio, w as f32, h as f32));
        }
    }

    boxes
}

/// Dilate a box outward by `D = area * ratio / perimeter`, clamped to bounds.
///
/// This is the axis-aligned analogue of the DBNet/Vatti unclip step. A
/// degenerate (zero-perimeter) box is returned unchanged rather than dividing
/// by zero.
fn unclip(b: &DetBox, ratio: f32, max_w: f32, max_h: f32) -> DetBox {
    let w = b.width();
    let h = b.height();
    let perimeter = 2.0 * (w + h);
    let distance = if perimeter > 0.0 {
        (w * h) * ratio / perimeter
    } else {
        0.0
    };
    DetBox {
        x0: (b.x0 - distance).clamp(0.0, max_w),
        y0: (b.y0 - distance).clamp(0.0, max_h),
        x1: (b.x1 + distance).clamp(0.0, max_w),
        y1: (b.y1 + distance).clamp(0.0, max_h),
        score: b.score,
    }
}

/// Sort boxes into natural reading order (top-to-bottom, then left-to-right).
///
/// Boxes are first ordered by top edge, then grouped into lines: consecutive
/// boxes whose top edges are within `line_threshold` pixels of the line's first
/// box form one line and are sorted left-to-right. This yields a total,
/// deterministic order (independent of detection order). True multi-column
/// reading order would need an XY-cut and is left as a future refinement.
pub(super) fn sort_reading_order(boxes: &mut [DetBox], line_threshold: f32) {
    boxes.sort_by(|a, b| a.y0.partial_cmp(&b.y0).unwrap_or(std::cmp::Ordering::Equal));
    let mut i = 0;
    while i < boxes.len() {
        let line_y = boxes[i].y0;
        let mut j = i + 1;
        while j < boxes.len() && (boxes[j].y0 - line_y).abs() <= line_threshold {
            j += 1;
        }
        boxes[i..j].sort_by(|a, b| a.x0.partial_cmp(&b.x0).unwrap_or(std::cmp::Ordering::Equal));
        i = j;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Build an H×W probability map and set the given rectangle to `value`.
    fn map_with_rect(
        h: usize,
        w: usize,
        rect: (usize, usize, usize, usize),
        value: f32,
    ) -> Array2<f32> {
        let mut m = Array2::<f32>::zeros((h, w));
        let (y0, x0, y1, x1) = rect;
        for y in y0..y1 {
            for x in x0..x1 {
                m[[y, x]] = value;
            }
        }
        m
    }

    fn lenient_params() -> DetParams {
        // Accept small synthetic blobs.
        DetParams {
            bin_threshold: 0.3,
            box_score_threshold: 0.5,
            unclip_ratio: 0.0, // tests that care about unclip set it explicitly
            min_box_size: 1,
        }
    }

    // ---- connected components ----

    #[test]
    fn single_blob_one_box() {
        let m = map_with_rect(20, 20, (2, 3, 7, 9), 0.9);
        let boxes = postprocess(m.view(), &lenient_params());
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (3.0, 2.0, 9.0, 7.0));
    }

    #[test]
    fn two_separated_blobs_two_boxes() {
        let mut m = map_with_rect(20, 20, (1, 1, 4, 4), 0.9);
        for y in 10..14 {
            for x in 10..14 {
                m[[y, x]] = 0.9;
            }
        }
        let boxes = postprocess(m.view(), &lenient_params());
        assert_eq!(boxes.len(), 2);
    }

    #[test]
    fn diagonal_touch_is_two_components_4conn() {
        let mut m = Array2::<f32>::zeros((6, 6));
        // Two 2x2 blobs touching only at a corner.
        for (y, x) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            m[[y, x]] = 0.9;
        }
        for (y, x) in [(2, 2), (2, 3), (3, 2), (3, 3)] {
            m[[y, x]] = 0.9;
        }
        let boxes = postprocess(m.view(), &lenient_params());
        assert_eq!(boxes.len(), 2, "4-connectivity must split diagonal touch");
    }

    #[test]
    fn empty_map_no_boxes() {
        let m = Array2::<f32>::zeros((16, 16));
        assert!(postprocess(m.view(), &lenient_params()).is_empty());
    }

    #[test]
    fn full_foreground_one_box_no_overflow() {
        // Large fully-foreground map: proves the iterative fill doesn't recurse.
        let m = Array2::<f32>::from_elem((512, 512), 0.9);
        let boxes = postprocess(m.view(), &lenient_params());
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        assert_eq!((b.x0, b.y0, b.x1, b.y1), (0.0, 0.0, 512.0, 512.0));
    }

    #[test]
    fn sub_min_size_blob_filtered() {
        let m = map_with_rect(20, 20, (5, 5, 7, 7), 0.9); // 2x2
        let params = DetParams {
            min_box_size: 3,
            ..lenient_params()
        };
        assert!(postprocess(m.view(), &params).is_empty());
    }

    // ---- binarize / score / filter ----

    #[test]
    fn threshold_is_inclusive() {
        let m = map_with_rect(10, 10, (1, 1, 5, 5), 0.3); // exactly bin_threshold
        // Score threshold below 0.3 so the binarize boundary is what's tested,
        // not the (equal-to-0.3) mean score.
        let params = DetParams {
            bin_threshold: 0.3,
            box_score_threshold: 0.2,
            min_box_size: 1,
            unclip_ratio: 0.0,
        };
        let boxes = postprocess(m.view(), &params);
        assert_eq!(
            boxes.len(),
            1,
            "prob == bin_threshold must count as foreground"
        );
    }

    #[test]
    fn low_mean_score_dropped() {
        // A blob above bin_threshold (0.3) but mean below box_score_threshold.
        let m = map_with_rect(10, 10, (1, 1, 6, 6), 0.4);
        let params = DetParams {
            bin_threshold: 0.3,
            box_score_threshold: 0.6,
            min_box_size: 1,
            unclip_ratio: 0.0,
        };
        assert!(postprocess(m.view(), &params).is_empty());
    }

    #[test]
    fn score_is_mean_over_component() {
        // 2x1 component with probs 0.6 and 0.8 -> mean 0.7.
        let mut m = Array2::<f32>::zeros((4, 4));
        m[[1, 1]] = 0.6;
        m[[1, 2]] = 0.8;
        let params = DetParams {
            bin_threshold: 0.3,
            box_score_threshold: 0.5,
            min_box_size: 1,
            unclip_ratio: 0.0,
        };
        let boxes = postprocess(m.view(), &params);
        assert_eq!(boxes.len(), 1);
        assert!((boxes[0].score - 0.7).abs() < 1e-5);
    }

    // ---- unclip ----

    #[test]
    fn unclip_expands_interior_box_by_distance() {
        // Interior 10x20 box (w=10,h=20): area=200, perim=60, D=200*1.5/60=5.
        let raw = DetBox {
            x0: 20.0,
            y0: 20.0,
            x1: 30.0,
            y1: 40.0,
            score: 1.0,
        };
        let out = unclip(&raw, 1.5, 1000.0, 1000.0);
        assert_eq!((out.x0, out.y0, out.x1, out.y1), (15.0, 15.0, 35.0, 45.0));
    }

    #[test]
    fn unclip_zero_ratio_unchanged() {
        let raw = DetBox {
            x0: 5.0,
            y0: 5.0,
            x1: 9.0,
            y1: 12.0,
            score: 1.0,
        };
        let out = unclip(&raw, 0.0, 100.0, 100.0);
        assert_eq!((out.x0, out.y0, out.x1, out.y1), (5.0, 5.0, 9.0, 12.0));
    }

    #[test]
    fn unclip_clamps_to_bounds() {
        let raw = DetBox {
            x0: 1.0,
            y0: 1.0,
            x1: 9.0,
            y1: 9.0,
            score: 1.0,
        };
        let out = unclip(&raw, 5.0, 10.0, 10.0);
        assert!(out.x0 >= 0.0 && out.y0 >= 0.0);
        assert!(out.x1 <= 10.0 && out.y1 <= 10.0);
    }

    #[test]
    fn unclip_degenerate_box_no_div_by_zero() {
        let raw = DetBox {
            x0: 3.0,
            y0: 3.0,
            x1: 3.0,
            y1: 3.0,
            score: 1.0,
        };
        let out = unclip(&raw, 1.5, 100.0, 100.0);
        assert_eq!((out.x0, out.y0, out.x1, out.y1), (3.0, 3.0, 3.0, 3.0));
    }

    // ---- scaling ----

    #[test]
    fn scale_to_maps_back_exactly() {
        // det 960x540 from orig 1920x1080 -> scale 0.5/0.5; box doubles.
        let b = DetBox {
            x0: 100.0,
            y0: 50.0,
            x1: 200.0,
            y1: 100.0,
            score: 1.0,
        };
        let o = b.scale_to(0.5, 0.5, 1920.0, 1080.0);
        assert_eq!((o.x0, o.y0, o.x1, o.y1), (200.0, 100.0, 400.0, 200.0));
    }

    #[test]
    fn scale_to_non_uniform_and_clamped() {
        let b = DetBox {
            x0: 10.0,
            y0: 10.0,
            x1: 50.0,
            y1: 20.0,
            score: 1.0,
        };
        let o = b.scale_to(2.0, 0.5, 100.0, 100.0);
        // x divided by 2, y divided by 0.5 (=*2), all within bounds.
        assert_eq!((o.x0, o.y0, o.x1, o.y1), (5.0, 20.0, 25.0, 40.0));
    }

    // ---- reading order ----

    fn b(x0: f32, y0: f32) -> DetBox {
        DetBox {
            x0,
            y0,
            x1: x0 + 5.0,
            y1: y0 + 5.0,
            score: 1.0,
        }
    }

    #[test]
    fn reading_order_top_to_bottom() {
        let mut v = vec![b(0.0, 30.0), b(0.0, 0.0), b(0.0, 15.0)];
        sort_reading_order(&mut v, 3.0);
        assert_eq!(
            v.iter().map(|x| x.y0).collect::<Vec<_>>(),
            vec![0.0, 15.0, 30.0]
        );
    }

    #[test]
    fn reading_order_left_to_right_same_line() {
        let mut v = vec![b(20.0, 1.0), b(0.0, 0.0), b(10.0, 2.0)];
        sort_reading_order(&mut v, 5.0); // all within one line
        assert_eq!(
            v.iter().map(|x| x.x0).collect::<Vec<_>>(),
            vec![0.0, 10.0, 20.0]
        );
    }

    #[test]
    fn reading_order_independent_of_input_order() {
        let make = || vec![b(10.0, 0.0), b(0.0, 0.0), b(0.0, 20.0), b(10.0, 20.0)];
        let mut a = make();
        let mut c = make();
        c.reverse();
        sort_reading_order(&mut a, 3.0);
        sort_reading_order(&mut c, 3.0);
        assert_eq!(a, c);
        // Expected: row0 (x0,x10), then row20 (x0,x10).
        assert_eq!(
            a.iter().map(|x| (x.x0, x.y0)).collect::<Vec<_>>(),
            vec![(0.0, 0.0), (10.0, 0.0), (0.0, 20.0), (10.0, 20.0)]
        );
    }

    #[test]
    fn reading_order_empty_and_single_no_panic() {
        let mut empty: Vec<DetBox> = Vec::new();
        sort_reading_order(&mut empty, 3.0);
        assert!(empty.is_empty());
        let mut one = vec![b(1.0, 1.0)];
        sort_reading_order(&mut one, 3.0);
        assert_eq!(one.len(), 1);
    }
}
