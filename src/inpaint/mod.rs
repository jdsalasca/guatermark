use anyhow::{bail, Result};
use image::{GrayImage, RgbImage};

mod diffusion;
mod patch;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algo {
    /// Multi-scale diffusion. Very fast, soft fill; ideal for thin text.
    Fast,
    /// Exemplar-based synthesis grown from the boundary. Slower, keeps texture.
    Patch,
}

pub(crate) const PATCH_RADIUS: i32 = 3;
pub(crate) const SEED: u64 = 0x6775_6174_6572_6d6b;
pub(crate) const RANDOM_TRIES: usize = 32;

/// Fill every masked pixel of `img`. Known pixels are never modified.
pub fn inpaint(img: &RgbImage, mask: &GrayImage, algo: Algo) -> Result<RgbImage> {
    if img.dimensions() != mask.dimensions() {
        bail!(
            "image is {}x{} but mask is {}x{}",
            img.width(),
            img.height(),
            mask.width(),
            mask.height()
        );
    }
    let mask = mask.as_raw().iter().map(|&v| v > 127).collect::<Vec<_>>();
    if !mask.iter().any(|&m| m) {
        return Ok(img.clone());
    }
    let base = diffusion::fill(img, &mask);
    Ok(match algo {
        Algo::Fast => base,
        Algo::Patch => patch::fill(img, &mask, &base),
    })
}

pub(crate) fn to_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}
