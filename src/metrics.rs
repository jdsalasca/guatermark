use image::{GrayImage, RgbImage};

/// Rectangle in pixels, `x/y` inclusive, `w/h` in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub fn padded(self, pad: u32, img_w: u32, img_h: u32) -> Self {
        let x = self.x.saturating_sub(pad);
        let y = self.y.saturating_sub(pad);
        let x1 = (self.x + self.w - 1 + pad).min(img_w - 1);
        let y1 = (self.y + self.h - 1 + pad).min(img_h - 1);
        Rect {
            x,
            y,
            w: x1 - x + 1,
            h: y1 - y + 1,
        }
    }

    fn x1(self) -> u32 {
        self.x + self.w
    }

    fn y1(self) -> u32 {
        self.y + self.h
    }
}

/// Bounding box of the white pixels of a mask, expanded by `pad`.
pub fn mask_bbox(mask: &GrayImage, pad: u32) -> Option<Rect> {
    let (w, h) = mask.dimensions();
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;
    for (x, y, p) in mask.enumerate_pixels() {
        if p.0[0] > 127 {
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !found {
        return None;
    }
    Some(
        Rect {
            x: min_x,
            y: min_y,
            w: max_x - min_x + 1,
            h: max_y - min_y + 1,
        }
        .padded(pad, w, h),
    )
}

fn luma(img: &RgbImage) -> Vec<f32> {
    img.pixels()
        .map(|p| 0.2126 * p.0[0] as f32 + 0.7152 * p.0[1] as f32 + 0.0722 * p.0[2] as f32)
        .collect()
}

/// RGB PSNR in dB, capped at 100 for identical images. `roi` restricts the area.
pub fn psnr(a: &RgbImage, b: &RgbImage, roi: Option<Rect>) -> f64 {
    assert_eq!(
        a.dimensions(),
        b.dimensions(),
        "psnr needs equal dimensions"
    );
    let (w, h) = a.dimensions();
    let rect = roi.unwrap_or(Rect { x: 0, y: 0, w, h });
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for y in rect.y..rect.y1().min(h) {
        for x in rect.x..rect.x1().min(w) {
            let pa = a.get_pixel(x, y).0;
            let pb = b.get_pixel(x, y).0;
            for c in 0..3 {
                let d = pa[c] as f64 - pb[c] as f64;
                sum += d * d;
            }
            count += 3;
        }
    }
    mse_to_psnr(sum / count.max(1) as f64)
}

/// PSNR computed only where the mask is white.
pub fn psnr_masked(a: &RgbImage, b: &RgbImage, mask: &GrayImage) -> f64 {
    assert_eq!(a.dimensions(), b.dimensions());
    assert_eq!(a.dimensions(), mask.dimensions());
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for (x, y, m) in mask.enumerate_pixels() {
        if m.0[0] > 127 {
            let pa = a.get_pixel(x, y).0;
            let pb = b.get_pixel(x, y).0;
            for c in 0..3 {
                let d = pa[c] as f64 - pb[c] as f64;
                sum += d * d;
                count += 1;
            }
        }
    }
    if count == 0 {
        return 100.0;
    }
    mse_to_psnr(sum / count as f64)
}

fn mse_to_psnr(mse: f64) -> f64 {
    if mse <= 0.0 {
        100.0
    } else {
        (10.0 * (255.0 * 255.0 / mse).log10()).min(100.0)
    }
}

/// Mean structural similarity (11x11 Gaussian, sigma 1.5) on luminance.
pub fn ssim(a: &RgbImage, b: &RgbImage, roi: Option<Rect>) -> f64 {
    assert_eq!(
        a.dimensions(),
        b.dimensions(),
        "ssim needs equal dimensions"
    );
    let (w, h) = a.dimensions();
    let (w, h) = (w as usize, h as usize);
    let rect = roi.unwrap_or(Rect {
        x: 0,
        y: 0,
        w: w as u32,
        h: h as u32,
    });
    let la = luma(a).into_iter().map(|v| v as f64).collect::<Vec<_>>();
    let lb = luma(b).into_iter().map(|v| v as f64).collect::<Vec<_>>();
    let kernel = gaussian_kernel();
    let mu1 = blur(&la, w, h, &kernel);
    let mu2 = blur(&lb, w, h, &kernel);
    let e11 = blur(&la.iter().map(|v| v * v).collect::<Vec<_>>(), w, h, &kernel);
    let e22 = blur(&lb.iter().map(|v| v * v).collect::<Vec<_>>(), w, h, &kernel);
    let e12 = blur(
        &la.iter().zip(&lb).map(|(x, y)| x * y).collect::<Vec<_>>(),
        w,
        h,
        &kernel,
    );
    const C1: f64 = 6.5025; // (0.01 * 255)^2
    const C2: f64 = 58.5225; // (0.03 * 255)^2
    let mut total = 0.0f64;
    let mut n = 0u64;
    for y in rect.y..rect.y1().min(h as u32) {
        for x in rect.x..rect.x1().min(w as u32) {
            let i = y as usize * w + x as usize;
            let (m1, m2) = (mu1[i], mu2[i]);
            let var1 = e11[i] - m1 * m1;
            let var2 = e22[i] - m2 * m2;
            let cov = e12[i] - m1 * m2;
            let num = (2.0 * m1 * m2 + C1) * (2.0 * cov + C2);
            let den = (m1 * m1 + m2 * m2 + C1) * (var1 + var2 + C2);
            total += num / den;
            n += 1;
        }
    }
    if n == 0 {
        return 1.0;
    }
    (total / n as f64).clamp(-1.0, 1.0)
}

fn blur(src: &[f64], w: usize, h: usize, kernel: &[f64]) -> Vec<f64> {
    let radius = (kernel.len() / 2) as i32;
    let mut tmp = vec![0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (k, &wt) in kernel.iter().enumerate() {
                let sx = (x as i32 + k as i32 - radius).clamp(0, w as i32 - 1) as usize;
                sum += src[y * w + sx] * wt;
            }
            tmp[y * w + x] = sum;
        }
    }
    let mut out = vec![0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (k, &wt) in kernel.iter().enumerate() {
                let sy = (y as i32 + k as i32 - radius).clamp(0, h as i32 - 1) as usize;
                sum += tmp[sy * w + x] * wt;
            }
            out[y * w + x] = sum;
        }
    }
    out
}

fn gaussian_kernel() -> Vec<f64> {
    let sigma = 1.5f64;
    let mut k = Vec::with_capacity(11);
    let mut sum = 0.0;
    for x in -5i32..=5 {
        let v = (-(x * x) as f64 / (2.0 * sigma * sigma)).exp();
        k.push(v);
        sum += v;
    }
    k.into_iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    fn gradient(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            Rgb([
                (x * 3) as u8 % 255,
                (y * 5) as u8 % 255,
                ((x + y) * 2) as u8 % 255,
            ])
        })
    }

    #[test]
    fn identical_images_score_perfectly() {
        let img = gradient(32, 24);
        assert_eq!(psnr(&img, &img, None), 100.0);
        assert!((ssim(&img, &img, None) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn noise_lowers_metrics() {
        let a = gradient(32, 24);
        let mut b = a.clone();
        b.put_pixel(5, 5, Rgb([255, 0, 255]));
        assert!(psnr(&a, &b, None) < 100.0);
        assert!(ssim(&a, &b, None) < 1.0);
    }

    #[test]
    fn bbox_and_masked_psnr() {
        let a = gradient(32, 24);
        let mut mask = GrayImage::new(32, 24);
        mask.put_pixel(10, 8, Luma([255]));
        let rect = mask_bbox(&mask, 2).unwrap();
        assert_eq!((rect.x, rect.y), (8, 6));
        let mut b = a.clone();
        b.put_pixel(10, 8, Rgb([0, 0, 0]));
        assert!(psnr_masked(&a, &b, &mask) < 100.0);
        b.put_pixel(12, 8, Rgb([0, 0, 0]));
        let mut outside = b.clone();
        outside.put_pixel(10, 8, *a.get_pixel(10, 8));
        assert_eq!(psnr_masked(&a, &outside, &mask), 100.0);
    }
}
