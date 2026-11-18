use image::RgbImage;

use super::to_u8;
struct FillLevel {
    w: usize,
    h: usize,
    color: Vec<[f32; 3]>,
    mask: Vec<bool>,
}

pub(super) fn fill(img: &RgbImage, mask: &[bool]) -> RgbImage {
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
    if let Some(last) = levels.last_mut() {
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
