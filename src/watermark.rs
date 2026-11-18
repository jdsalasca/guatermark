use ab_glyph::{point, Font, FontVec, PxScale, ScaleFont};
use anyhow::{bail, Context, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};

const FONT_DATA: &[u8] = include_bytes!("../assets/IBMPlexSans-Regular.ttf");

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Style {
    /// One line in the bottom-right corner.
    Corner,
    /// Repeating diagonal text across the whole image.
    Tile,
    /// Large centred brand mark.
    Logo,
}

impl Style {
    pub fn label(self) -> &'static str {
        match self {
            Style::Corner => "corner",
            Style::Tile => "tile",
            Style::Logo => "logo",
        }
    }
}

#[derive(Clone, Debug)]
pub struct WatermarkSpec {
    pub text: String,
    pub style: Style,
    pub alpha: f32,
    pub color: [u8; 3],
}

/// Add a synthetic watermark and return the image plus the exact affected-pixel mask.
pub fn stamp(img: &RgbImage, spec: &WatermarkSpec) -> Result<(RgbImage, GrayImage)> {
    let (w, h) = (img.width(), img.height());
    if spec.text.trim().is_empty() {
        bail!("watermark text is empty");
    }
    if !(0.0..=1.0).contains(&spec.alpha) || spec.alpha == 0.0 {
        bail!("alpha must be in (0, 1]");
    }
    if w < 32 || h < 32 {
        bail!("image is too small to stamp a watermark");
    }
    let mut coverage = vec![0f32; (w * h) as usize];

    match spec.style {
        Style::Corner => {
            let mut px = (h as f32 / 13.0).clamp(14.0, 96.0);
            let mut tile = render_text(&spec.text, px)?;
            // Keep the stamp under 45% of the image width.
            for _ in 0..3 {
                if tile.width() <= w * 45 / 100 {
                    break;
                }
                px *= 0.8;
                tile = render_text(&spec.text, px)?;
            }
            let margin = ((w.min(h) as f32) * 0.035).round() as u32;
            let ox = w.saturating_sub(tile.width() + margin);
            let oy = h.saturating_sub(tile.height() + margin);
            for (x, y, p) in tile.enumerate_pixels() {
                if ox + x < w && oy + y < h {
                    coverage[((oy + y) * w + ox + x) as usize] = p.0[0] as f32 / 255.0;
                }
            }
        }
        Style::Tile => {
            let px = (h as f32 / 17.0).clamp(14.0, 64.0);
            let tile = render_text(&spec.text, px)?;
            let (tw, th) = (tile.width() as f32, tile.height() as f32);
            let gap_x = tw * 0.9;
            let gap_y = th * 2.1;
            let angle = 30f32.to_radians();
            let (sin, cos) = angle.sin_cos();
            for y in 0..h {
                for x in 0..w {
                    let u = (x as f32 * cos - y as f32 * sin).rem_euclid(tw + gap_x);
                    let v = (x as f32 * sin + y as f32 * cos).rem_euclid(th + gap_y);
                    if u < tw && v < th {
                        coverage[(y * w + x) as usize] = sample(&tile, u, v);
                    }
                }
            }
        }
        Style::Logo => {
            let mut px = (h as f32 / 3.2).clamp(36.0, 200.0);
            let mut tile = render_text(&spec.text, px)?;
            for _ in 0..5 {
                if tile.width() <= w * 75 / 100 {
                    break;
                }
                px *= 0.8;
                tile = render_text(&spec.text, px)?;
            }
            let ox = (w.saturating_sub(tile.width())) / 2;
            let oy = (h.saturating_sub(tile.height())) / 2;
            for (x, y, p) in tile.enumerate_pixels() {
                if ox + x < w && oy + y < h {
                    coverage[((oy + y) * w + ox + x) as usize] = p.0[0] as f32 / 255.0;
                }
            }
        }
    }

    let mut out = img.clone();
    let mut mask = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let a = (spec.alpha * coverage[(y * w + x) as usize]).clamp(0.0, 1.0);
            if a > 0.01 {
                let p = img.get_pixel(x, y).0;
                let mut c = [0u8; 3];
                for k in 0..3 {
                    c[k] = (p[k] as f32 * (1.0 - a) + spec.color[k] as f32 * a)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
                out.put_pixel(x, y, Rgb(c));
                mask.put_pixel(x, y, Luma([255]));
            }
        }
    }
    Ok((out, mask))
}

fn render_text(text: &str, px: f32) -> Result<GrayImage> {
    let font = FontVec::try_from_vec(FONT_DATA.to_vec()).context("cannot parse font")?;
    let scaled = font.as_scaled(PxScale::from(px));
    let ascent = scaled.ascent();
    let baseline = ascent + 4.0;
    let mut width = 8.0f32;
    let mut prev = None;
    for ch in text.chars() {
        let id = font.glyph_id(ch);
        if let Some(p) = prev {
            width += scaled.kern(p, id);
        }
        width += scaled.h_advance(id);
        prev = Some(id);
    }
    let height = (ascent - scaled.descent()).ceil() as u32 + 8;
    let mut canvas = GrayImage::new(width.ceil().max(1.0) as u32, height.max(1));

    let mut caret = 4.0f32;
    let mut prev = None;
    for ch in text.chars() {
        let id = font.glyph_id(ch);
        if let Some(p) = prev {
            caret += scaled.kern(p, id);
        }
        if ch != ' ' {
            let glyph = id.with_scale_and_position(PxScale::from(px), point(caret, baseline));
            if let Some(og) = font.outline_glyph(glyph) {
                let bounds = og.px_bounds();
                og.draw(|gx, gy, cov| {
                    let x = bounds.min.x.round() as i32 + gx as i32;
                    let y = bounds.min.y.round() as i32 + gy as i32;
                    if x >= 0 && y >= 0 && x < canvas.width() as i32 && y < canvas.height() as i32 {
                        let v = (cov * 255.0).round().clamp(0.0, 255.0) as u8;
                        let old = canvas.get_pixel_mut(x as u32, y as u32);
                        old.0[0] = old.0[0].max(v);
                    }
                });
            }
        }
        caret += scaled.h_advance(id);
        prev = Some(id);
    }
    Ok(canvas)
}

fn sample(tile: &GrayImage, x: f32, y: f32) -> f32 {
    let (tw, th) = (tile.width() as i32, tile.height() as i32);
    if x < 0.0 || y < 0.0 {
        return 0.0;
    }
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    if x0 >= tw || y0 >= th {
        return 0.0;
    }
    let x1 = (x0 + 1).min(tw - 1);
    let y1 = (y0 + 1).min(th - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p = |px: i32, py: i32| tile.get_pixel(px as u32, py as u32).0[0] as f32;
    let a = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let b = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    (a * (1.0 - fy) + b * fy) / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn photo() -> RgbImage {
        RgbImage::from_fn(120, 90, |x, y| Rgb([(x * 2) as u8, (y * 2) as u8, 120]))
    }

    #[test]
    fn corner_stamp_affects_only_its_mask() {
        let img = photo();
        let spec = WatermarkSpec {
            text: "GUATERMARK".into(),
            style: Style::Corner,
            alpha: 0.4,
            color: [255, 255, 255],
        };
        let (out, mask) = stamp(&img, &spec).unwrap();
        let mut marked = 0;
        for (x, y, p) in out.enumerate_pixels() {
            if mask.get_pixel(x, y).0[0] > 127 {
                marked += 1;
                assert_ne!(p, img.get_pixel(x, y));
            } else {
                assert_eq!(p, img.get_pixel(x, y), "pixel outside mask changed");
            }
        }
        assert!(marked > 100, "mask too small: {marked}");
    }

    #[test]
    fn tile_covers_a_fraction_of_the_image() {
        let img = photo();
        let spec = WatermarkSpec {
            text: "SAMPLE".into(),
            style: Style::Tile,
            alpha: 0.25,
            color: [255, 255, 255],
        };
        let (out, mask) = stamp(&img, &spec).unwrap();
        let marked = mask.pixels().filter(|p| p.0[0] > 127).count();
        let total = (img.width() * img.height()) as usize;
        assert!(marked > total / 100);
        assert!(marked < total / 2);
        assert_eq!(out.dimensions(), img.dimensions());
    }
}
