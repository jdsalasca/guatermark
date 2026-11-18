use image::{Rgb, RgbImage};
use rayon::prelude::*;
use std::f32::consts::PI;

/// Denoise -> Lanczos3 upscale -> unsharp mask.
pub fn enhance(img: &RgbImage, denoise: f32, sharpen: f32, scale: u32) -> RgbImage {
    let mut out = if denoise > 0.0 {
        bilateral(img, 2, 8.0 + 52.0 * denoise.clamp(0.0, 1.0))
    } else {
        img.clone()
    };
    if scale > 1 {
        out = lanczos3(&out, scale);
    }
    if sharpen > 0.0 {
        out = unsharp(&out, 1.4 * sharpen.clamp(0.0, 2.0));
    }
    out
}

pub fn bilateral(img: &RgbImage, radius: u32, sigma_color: f32) -> RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let r = radius as i32;
    let sigma_space = 2.0f32;
    let mut spatial = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
    for dy in -r..=r {
        for dx in -r..=r {
            spatial.push((-(dx * dx + dy * dy) as f32 / (2.0 * sigma_space * sigma_space)).exp());
        }
    }
    let pixels: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
    let inv_color = 1.0 / (2.0 * sigma_color * sigma_color);
    let rows: Vec<Vec<[u8; 3]>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(w);
            for x in 0..w {
                let center = pixels[y * w + x];
                let mut sum = [0f32; 3];
                let mut wsum = 0f32;
                let mut k = 0usize;
                for dy in -r..=r {
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    for dx in -r..=r {
                        let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                        let c = pixels[sy * w + sx];
                        let mut cd = 0f32;
                        for ch in 0..3 {
                            let d = c[ch] as f32 - center[ch] as f32;
                            cd += d * d;
                        }
                        let weight = spatial[k] * (-cd * inv_color).exp();
                        for ch in 0..3 {
                            sum[ch] += c[ch] as f32 * weight;
                        }
                        wsum += weight;
                        k += 1;
                    }
                }
                row.push([
                    (sum[0] / wsum).round().clamp(0.0, 255.0) as u8,
                    (sum[1] / wsum).round().clamp(0.0, 255.0) as u8,
                    (sum[2] / wsum).round().clamp(0.0, 255.0) as u8,
                ]);
            }
            row
        })
        .collect();
    let mut out = RgbImage::new(w as u32, h as u32);
    for (y, row) in rows.into_iter().enumerate() {
        for (x, c) in row.into_iter().enumerate() {
            out.put_pixel(x as u32, y as u32, Rgb(c));
        }
    }
    out
}

pub fn lanczos3(img: &RgbImage, scale: u32) -> RgbImage {
    if scale <= 1 {
        return img.clone();
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    let dw = w * scale as usize;
    let dh = h * scale as usize;
    let src: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();

    // Horizontal pass.
    let inter: Vec<Vec<[f32; 3]>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![[0f32; 3]; dw];
            for (dx, slot) in row.iter_mut().enumerate() {
                let sx = (dx as f32 + 0.5) / scale as f32 - 0.5;
                let i0 = (sx - 3.0).ceil() as i32;
                let i1 = (sx + 3.0).floor() as i32;
                let mut weights = Vec::new();
                let mut wsum = 0f32;
                for i in i0..=i1 {
                    let wt = lanczos(sx - i as f32);
                    weights.push((i, wt));
                    wsum += wt;
                }
                let mut acc = [0f32; 3];
                for (i, wt) in weights {
                    let ix = i.clamp(0, w as i32 - 1) as usize;
                    let c = src[y * w + ix];
                    for k in 0..3 {
                        acc[k] += c[k] as f32 * wt;
                    }
                }
                for k in 0..3 {
                    slot[k] = acc[k] / wsum;
                }
            }
            row
        })
        .collect();

    // Vertical pass.
    let rows: Vec<Vec<[u8; 3]>> = (0..dh)
        .into_par_iter()
        .map(|dy| {
            let sy = (dy as f32 + 0.5) / scale as f32 - 0.5;
            let i0 = (sy - 3.0).ceil() as i32;
            let i1 = (sy + 3.0).floor() as i32;
            let mut weights = Vec::new();
            let mut wsum = 0f32;
            for i in i0..=i1 {
                let wt = lanczos(sy - i as f32);
                weights.push((i, wt));
                wsum += wt;
            }
            let mut row = vec![[0u8; 3]; dw];
            for (x, slot) in row.iter_mut().enumerate() {
                let mut acc = [0f32; 3];
                for &(i, wt) in &weights {
                    let iy = i.clamp(0, h as i32 - 1) as usize;
                    let c = inter[iy][x];
                    for k in 0..3 {
                        acc[k] += c[k] * wt;
                    }
                }
                for k in 0..3 {
                    slot[k] = (acc[k] / wsum).round().clamp(0.0, 255.0) as u8;
                }
            }
            row
        })
        .collect();
    let mut out = RgbImage::new(dw as u32, dh as u32);
    for (y, row) in rows.into_iter().enumerate() {
        for (x, c) in row.into_iter().enumerate() {
            out.put_pixel(x as u32, y as u32, Rgb(c));
        }
    }
    out
}

fn lanczos(x: f32) -> f32 {
    let x = x.abs();
    if x < 1e-6 {
        1.0
    } else if x >= 3.0 {
        0.0
    } else {
        let px = PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    }
}

pub fn unsharp(img: &RgbImage, amount: f32) -> RgbImage {
    if amount <= 0.0 {
        return img.clone();
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    let luma: Vec<f32> = img
        .pixels()
        .map(|p| 0.2126 * p.0[0] as f32 + 0.7152 * p.0[1] as f32 + 0.0722 * p.0[2] as f32)
        .collect();
    let blurred = gaussian3_twice(&luma, w, h);
    let mut out = img.clone();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let delta = (luma[i] - blurred[i]) * amount;
            let p = img.get_pixel(x as u32, y as u32).0;
            out.put_pixel(
                x as u32,
                y as u32,
                Rgb([
                    (p[0] as f32 + delta).round().clamp(0.0, 255.0) as u8,
                    (p[1] as f32 + delta).round().clamp(0.0, 255.0) as u8,
                    (p[2] as f32 + delta).round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    out
}

fn gaussian3_twice(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut a = src.to_vec();
    let mut b = vec![0f32; w * h];
    for _ in 0..2 {
        for y in 0..h {
            for x in 0..w {
                let l = a[y * w + x.saturating_sub(1)];
                let c = a[y * w + x];
                let r = a[y * w + (x + 1).min(w - 1)];
                b[y * w + x] = (l + 2.0 * c + r) / 4.0;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let t = b[y.saturating_sub(1) * w + x];
                let c = b[y * w + x];
                let d = b[(y + 1).min(h - 1) * w + x];
                a[y * w + x] = (t + 2.0 * c + d) / 4.0;
            }
        }
    }
    a
}

/// Simulate a low-quality source: box downscale + gaussian noise.
pub fn degrade(img: &RgbImage, factor: u32, noise_sigma: f32, seed: u64) -> RgbImage {
    let factor = factor.max(1);
    let w = img.width() / factor;
    let h = img.height() / factor;
    let mut out = RgbImage::new(w, h);
    let mut state = seed;
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0f32; 3];
            let mut n = 0.0;
            for dy in 0..factor {
                for dx in 0..factor {
                    let p = img.get_pixel(x * factor + dx, y * factor + dy).0;
                    for k in 0..3 {
                        sum[k] += p[k] as f32;
                    }
                    n += 1.0;
                }
            }
            let mut c = [0u8; 3];
            for k in 0..3 {
                let base = sum[k] / n;
                let noise = gaussian(&mut state) * noise_sigma;
                c[k] = (base + noise).round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, Rgb(c));
        }
    }
    out
}

pub fn nearest(img: &RgbImage, scale: u32) -> RgbImage {
    let scale = scale.max(1);
    let mut out = RgbImage::new(img.width() * scale, img.height() * scale);
    for y in 0..out.height() {
        for x in 0..out.width() {
            out.put_pixel(x, y, *img.get_pixel(x / scale, y / scale));
        }
    }
    out
}

fn gaussian(state: &mut u64) -> f32 {
    let u1 = ((splitmix(state) >> 11) as f32) / ((1u64 << 53) as f32);
    let u2 = ((splitmix(state) >> 11) as f32) / ((1u64 << 53) as f32);
    (-2.0 * u1.max(1e-9).ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
