use anyhow::{bail, Result};
use image::{GrayImage, RgbImage};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algo {
    /// Multi-scale diffusion. Very fast, soft fill; ideal for thin text.
    Fast,
    /// Exemplar-based synthesis grown from the boundary. Slower, keeps texture.
    Patch,
}

const PATCH_RADIUS: i32 = 3;
const SEED: u64 = 0x6775_6174_6572_6d6b;
const RANDOM_TRIES: usize = 32;

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
    let base = diffusion_fill(img, &mask);
    Ok(match algo {
        Algo::Fast => base,
        Algo::Patch => patch_fill(img, &mask, &base),
    })
}

// ---------------------------------------------------------------- diffusion

struct FillLevel {
    w: usize,
    h: usize,
    color: Vec<[f32; 3]>,
    mask: Vec<bool>,
}

fn diffusion_fill(img: &RgbImage, mask: &[bool]) -> RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut levels = Vec::new();
    let mut cur = FillLevel {
        w,
        h,
        color: img
            .pixels()
            .map(|p| [p.0[0] as f32, p.0[1] as f32, p.0[2] as f32])
            .collect(),
        mask: mask.to_vec(),
    };
    loop {
        let stop = cur.w <= 24 || cur.h <= 24 || levels.len() >= 6;
        if stop {
            levels.push(cur);
            break;
        }
        let next = downsample_fill(&cur);
        levels.push(cur);
        cur = next;
    }

    // Coarsest level: flat average of the known pixels.
    {
        let last = levels.last_mut().expect("levels");
        let mut sum = [0f64; 3];
        let mut n = 0u64;
        for (i, c) in last.color.iter().enumerate() {
            if !last.mask[i] {
                for k in 0..3 {
                    sum[k] += c[k] as f64;
                }
                n += 1;
            }
        }
        let avg = if n > 0 {
            [
                (sum[0] / n as f64) as f32,
                (sum[1] / n as f64) as f32,
                (sum[2] / n as f64) as f32,
            ]
        } else {
            [128.0; 3]
        };
        for i in 0..last.color.len() {
            if last.mask[i] {
                last.color[i] = avg;
            }
        }
    }

    for idx in (0..levels.len().saturating_sub(1)).rev() {
        let up = {
            let coarse = &levels[idx + 1];
            let fine = &levels[idx];
            upsample_color(&coarse.color, coarse.w, coarse.h, fine.w, fine.h)
        };
        let (masked, mut colors) = {
            let lvl = &levels[idx];
            (lvl.mask.clone(), lvl.color.clone())
        };
        for i in 0..colors.len() {
            if masked[i] {
                colors[i] = up[i];
            }
        }
        let (lw, lh) = (levels[idx].w, levels[idx].h);
        let mut next = colors.clone();
        for _ in 0..8 {
            for y in 0..lh {
                for x in 0..lw {
                    let i = y * lw + x;
                    if !masked[i] {
                        continue;
                    }
                    let mut sum = [0f32; 3];
                    let mut n = 0.0f32;
                    for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let sx = x as i32 + dx;
                        let sy = y as i32 + dy;
                        if sx < 0 || sy < 0 || sx >= lw as i32 || sy >= lh as i32 {
                            continue;
                        }
                        let c = colors[sy as usize * lw + sx as usize];
                        for k in 0..3 {
                            sum[k] += c[k];
                        }
                        n += 1.0;
                    }
                    if n > 0.0 {
                        next[i] = [sum[0] / n, sum[1] / n, sum[2] / n];
                    }
                }
            }
            std::mem::swap(&mut colors, &mut next);
        }
        levels[idx].color = colors;
    }

    let fine = &levels[0];
    RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let c = fine.color[y as usize * w + x as usize];
        image::Rgb([to_u8(c[0]), to_u8(c[1]), to_u8(c[2])])
    })
}

fn downsample_fill(src: &FillLevel) -> FillLevel {
    let nw = src.w / 2;
    let nh = src.h / 2;
    let mut color = vec![[128f32; 3]; nw * nh];
    let mut mask = vec![false; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = [0f32; 3];
            let mut n = 0.0f32;
            let mut any = false;
            for dy in 0..2 {
                for dx in 0..2 {
                    let si = (y * 2 + dy) * src.w + x * 2 + dx;
                    if src.mask[si] {
                        any = true;
                    } else {
                        for (s, c) in sum.iter_mut().zip(src.color[si]) {
                            *s += c;
                        }
                        n += 1.0;
                    }
                }
            }
            let di = y * nw + x;
            if n > 0.0 {
                for k in 0..3 {
                    color[di][k] = sum[k] / n;
                }
            }
            mask[di] = any;
        }
    }
    FillLevel {
        w: nw,
        h: nh,
        color,
        mask,
    }
}

fn upsample_color(src: &[[f32; 3]], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0f32; 3]; dw * dh];
    for y in 0..dh {
        let sy = ((y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5).clamp(0.0, sh as f32 - 1.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = sy - y0 as f32;
        for x in 0..dw {
            let sx = ((x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5).clamp(0.0, sw as f32 - 1.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = sx - x0 as f32;
            let mut c = [0f32; 3];
            for k in 0..3 {
                let a = src[y0 * sw + x0][k] * (1.0 - fx) + src[y0 * sw + x1][k] * fx;
                let b = src[y1 * sw + x0][k] * (1.0 - fx) + src[y1 * sw + x1][k] * fx;
                c[k] = a * (1.0 - fy) + b * fy;
            }
            out[y * dw + x] = c;
        }
    }
    out
}

// ------------------------------------------------------ exemplar-based fill

/// Criminisi-style inpainting: repeatedly pick the boundary pixel with the most
/// known context, copy the best matching source patch, and blend the new pixels
/// into the fill. Texture comes from the image itself; no models involved.
fn patch_fill(img: &RgbImage, mask: &[bool], base: &RgbImage) -> RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let n = w * h;
    let img_px: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
    let mut recon: Vec<[u8; 3]> = base.pixels().map(|p| p.0).collect();
    let mut valid = mask.iter().map(|&m| !m).collect::<Vec<bool>>();

    // Source patch centres: fully in bounds and with no masked pixel.
    let mut source_ok = vec![false; n];
    let mut sources = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let mut ok = true;
            'patch: for dy in -PATCH_RADIUS..=PATCH_RADIUS {
                for dx in -PATCH_RADIUS..=PATCH_RADIUS {
                    let sx = x as i32 + dx;
                    let sy = y as i32 + dy;
                    if sx < 0
                        || sy < 0
                        || sx >= w as i32
                        || sy >= h as i32
                        || mask[sy as usize * w + sx as usize]
                    {
                        ok = false;
                        break 'patch;
                    }
                }
            }
            if ok {
                source_ok[y * w + x] = true;
                sources.push((x, y));
            }
        }
    }
    if sources.is_empty() {
        return base.clone();
    }

    let mut offset: Vec<Option<(i32, i32)>> = vec![None; n];
    let mut queued = vec![false; n];
    let mut queue = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask[i] && has_valid_neighbour(&valid, w, h, x, y) {
                queued[i] = true;
                queue.push_back(i);
            }
        }
    }
    let mut remaining = mask.iter().filter(|&&m| m).count();

    while remaining > 0 {
        let Some(i) = queue.pop_front() else {
            break;
        };
        if valid[i] {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let mut best_d = f32::INFINITY;
        let mut best_src = None;
        {
            let consider = |sx: i32, sy: i32, bd: &mut f32, bs: &mut Option<(i32, i32)>| {
                if sx < PATCH_RADIUS
                    || sy < PATCH_RADIUS
                    || sx + PATCH_RADIUS >= w as i32
                    || sy + PATCH_RADIUS >= h as i32
                    || !source_ok[sy as usize * w + sx as usize]
                {
                    return;
                }
                let d = patch_dist(
                    &img_px, &recon, mask, &valid, w, h, x as i32, y as i32, sx, sy, *bd,
                );
                if d < *bd {
                    *bd = d;
                    *bs = Some((sx, sy));
                }
            };

            // Neighbouring patches have already chosen offsets; reusing them
            // keeps texture and structures continuous across the hole.
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    if let Some((ox, oy)) = offset[ny as usize * w + nx as usize] {
                        consider(x as i32 + ox, y as i32 + oy, &mut best_d, &mut best_src);
                    }
                }
            }
            let mut state =
                SEED ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((remaining as u64) << 19);
            for _ in 0..RANDOM_TRIES {
                let s = sources[(splitmix(&mut state) % sources.len() as u64) as usize];
                consider(s.0 as i32, s.1 as i32, &mut best_d, &mut best_src);
                if best_d == 0.0 {
                    break;
                }
            }
            // Local refinement around the best match so far.
            if let Some((bx, by)) = best_src {
                let mut radius = 32i32;
                while radius >= 1 {
                    let ox = (splitmix(&mut state) % (2 * radius as u64 + 1)) as i32 - radius;
                    let oy = (splitmix(&mut state) % (2 * radius as u64 + 1)) as i32 - radius;
                    consider(bx + ox, by + oy, &mut best_d, &mut best_src);
                    radius /= 2;
                }
            }
        }

        let Some((sx, sy)) = best_src else {
            recon[i] = base.get_pixel(x as u32, y as u32).0;
            valid[i] = true;
            remaining -= 1;
            continue;
        };

        // Copy the centre of the best patch. Copying only one pixel per step
        // keeps block seams out; coherence comes from neighbour offsets.
        recon[i] = img_px[sy as usize * w + sx as usize];
        offset[i] = Some((sx - x as i32, sy - y as i32));
        valid[i] = true;
        remaining -= 1;
        for (ndx, ndy) in [
            (-1i32, 0i32),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ] {
            let nx = x as i32 + ndx;
            let ny = y as i32 + ndy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let ni = ny as usize * w + nx as usize;
            if mask[ni] && !valid[ni] && !queued[ni] {
                queued[ni] = true;
                queue.push_back(ni);
            }
        }
    }

    // Feather the seams of the synthesized blocks: a 3x3 average restricted to
    // the hole, weighted 25%, hides hard patch borders without blurring the photo.
    feather(&mut recon, mask, w, h, 1);

    RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let i = y as usize * w + x as usize;
        image::Rgb(if mask[i] { recon[i] } else { img_px[i] })
    })
}

fn feather(recon: &mut [[u8; 3]], mask: &[bool], w: usize, h: usize, rounds: u32) {
    let mut src = recon.to_vec();
    for _ in 0..rounds {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if !mask[i] {
                    continue;
                }
                let mut sum = [0u32; 3];
                let mut n = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let c = src[ny as usize * w + nx as usize];
                        for k in 0..3 {
                            sum[k] += c[k] as u32;
                        }
                        n += 1;
                    }
                }
                for k in 0..3 {
                    let avg = sum[k] as f32 / n as f32;
                    let cur = src[i][k] as f32;
                    recon[i][k] = to_u8(cur * 0.75 + avg * 0.25);
                }
            }
        }
        src.copy_from_slice(recon);
    }
}

fn has_valid_neighbour(valid: &[bool], w: usize, h: usize, x: usize, y: usize) -> bool {
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0
                && ny >= 0
                && nx < w as i32
                && ny < h as i32
                && valid[ny as usize * w + nx as usize]
            {
                return true;
            }
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn patch_dist(
    img: &[[u8; 3]],
    recon: &[[u8; 3]],
    mask: &[bool],
    valid: &[bool],
    w: usize,
    h: usize,
    tx: i32,
    ty: i32,
    sx: i32,
    sy: i32,
    cutoff: f32,
) -> f32 {
    let mut sum = 0f32;
    let mut count = 0f32;
    for dy in -PATCH_RADIUS..=PATCH_RADIUS {
        for dx in -PATCH_RADIUS..=PATCH_RADIUS {
            let x = tx + dx;
            let y = ty + dy;
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                continue;
            }
            let ax = sx + dx;
            let ay = sy + dy;
            if ax < 0 || ay < 0 || ax >= w as i32 || ay >= h as i32 {
                continue;
            }
            let ti = y as usize * w + x as usize;
            if mask[ti] && !valid[ti] {
                continue;
            }
            let a = if mask[ti] { recon[ti] } else { img[ti] };
            let b = img[ay as usize * w + ax as usize];
            for k in 0..3 {
                let d = a[k] as f32 - b[k] as f32;
                sum += d * d;
            }
            count += 3.0;
            if sum > cutoff * count {
                return sum / count;
            }
        }
    }
    if count == 0.0 {
        f32::INFINITY
    } else {
        sum / count
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn to_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    fn textured(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            Rgb([
                ((x * 7 + y * 3) % 255) as u8,
                ((x * 3 + y * 11) % 255) as u8,
                90,
            ])
        })
    }

    #[test]
    fn fills_hole_and_keeps_known_pixels() {
        let mut img = textured(64, 48);
        for y in 10..20 {
            for x in 15..35 {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
        let mut mask = GrayImage::new(64, 48);
        for y in 10..20 {
            for x in 15..35 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        for algo in [Algo::Fast, Algo::Patch] {
            let out = inpaint(&img, &mask, algo).unwrap();
            assert_eq!(out.dimensions(), img.dimensions());
            let mut changed = 0;
            for (x, y, p) in mask.enumerate_pixels() {
                if p.0[0] > 127 {
                    if out.get_pixel(x, y) != img.get_pixel(x, y) {
                        changed += 1;
                    }
                } else {
                    assert_eq!(
                        out.get_pixel(x, y),
                        img.get_pixel(x, y),
                        "known pixel changed"
                    );
                }
            }
            assert!(changed > 0, "nothing was replaced");
        }
    }

    #[test]
    fn patch_is_deterministic() {
        let img = textured(80, 60);
        let mut mask = GrayImage::new(80, 60);
        for y in 20..35 {
            for x in 25..45 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let a = inpaint(&img, &mask, Algo::Patch).unwrap();
        let b = inpaint(&img, &mask, Algo::Patch).unwrap();
        assert_eq!(a.as_raw(), b.as_raw());
    }
}
