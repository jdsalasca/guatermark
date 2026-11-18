use anyhow::{Context, Result};
use image::{GrayImage, Luma};
use std::path::Path;

/// Load a mask image. Alpha is used when present, otherwise luminance.
/// White (or opaque) means "remove this pixel".
pub fn load_mask(path: &Path) -> Result<GrayImage> {
    let img = image::open(path).with_context(|| format!("cannot open mask {}", path.display()))?;
    if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        Ok(GrayImage::from_fn(rgba.width(), rgba.height(), |x, y| {
            Luma([rgba.get_pixel(x, y).0[3]])
        }))
    } else {
        Ok(img.to_luma8())
    }
}

pub fn save_mask(mask: &GrayImage, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    mask.save(path)
        .with_context(|| format!("cannot save mask {}", path.display()))
}

/// Grow a mask by `radius` pixels (square kernel).
pub fn dilate(mask: &GrayImage, radius: u32) -> GrayImage {
    if radius == 0 {
        return mask.clone();
    }
    let (w, h) = mask.dimensions();
    let r = radius as i64;
    GrayImage::from_fn(w, h, |x, y| {
        let mut hit = false;
        'window: for dy in -r..=r {
            let sy = y as i64 + dy;
            if sy < 0 || sy >= h as i64 {
                continue;
            }
            for dx in -r..=r {
                let sx = x as i64 + dx;
                if sx < 0 || sx >= w as i64 {
                    continue;
                }
                if mask.get_pixel(sx as u32, sy as u32).0[0] > 127 {
                    hit = true;
                    break 'window;
                }
            }
        }
        Luma([if hit { 255 } else { 0 }])
    })
}
