use crate::enhance;
use crate::inpaint::{self, Algo};
use crate::mask;
use crate::metrics::{self, Rect};
use crate::watermark::{self, Style, WatermarkSpec};
use anyhow::{bail, Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{imageops, ImageEncoder, RgbImage};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod report;

use report::render_html;

#[derive(Serialize)]
pub struct RemovalCase {
    pub image: String,
    pub style: String,
    pub algo: String,
    pub mask: String,
    pub input: String,
    pub clean: String,
    pub output: String,
    pub psnr_full_before: f64,
    pub psnr_full_after: f64,
    pub psnr_roi_before: f64,
    pub psnr_roi_after: f64,
    pub psnr_masked_before: f64,
    pub psnr_masked_after: f64,
    pub ssim_roi_before: f64,
    pub ssim_roi_after: f64,
    pub seconds: f64,
}

#[derive(Serialize)]
pub struct EnhanceCase {
    pub image: String,
    pub clean: String,
    pub degraded: String,
    pub method: String,
    pub output: String,
    pub psnr: f64,
    pub ssim: f64,
    pub seconds: f64,
}

#[derive(Serialize)]
pub struct Group {
    pub label: String,
    pub cases: usize,
    pub psnr: f64,
    pub ssim: f64,
    pub gain_db: f64,
    pub seconds: f64,
}

#[derive(Serialize)]
pub struct Summary {
    pub images: usize,
    pub removal: Vec<Group>,
    pub enhancement: Vec<Group>,
}

#[derive(Serialize)]
pub struct Report {
    pub cases: Vec<RemovalCase>,
    pub enhancement: Vec<EnhanceCase>,
    pub summary: Summary,
}

/// Reproducible benchmark: stamps known watermarks on `dir/clean`, removes them
/// with both algorithms and writes metrics + an HTML report.
pub fn run(dir: &Path, examples: &Path) -> Result<Report> {
    let clean_dir = dir.join("clean");
    if !clean_dir.is_dir() {
        bail!(
            "{} does not exist; put clean ground-truth images there",
            clean_dir.display()
        );
    }
    let out_dir = dir.join("out");
    fs::create_dir_all(&out_dir)?;
    fs::create_dir_all(examples.join("masks"))?;

    let mut files: Vec<PathBuf> = fs::read_dir(&clean_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
                Some(ref e) if matches!(e.as_str(), "jpg" | "jpeg" | "png")
            )
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no images in {}", clean_dir.display());
    }

    let examples_rel = relative_path(dir, examples);
    let mut cases = Vec::new();
    let mut enhancement = Vec::new();

    for path in &files {
        let clean = load_rgb(path)?;
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
            .to_string();
        let Some(file_name) = path.file_name() else {
            continue;
        };
        let clean_rel = format!("clean/{}", file_name.to_string_lossy());

        for style in [Style::Corner, Style::Tile, Style::Logo] {
            let (text, alpha) = match style {
                Style::Corner => ("GUATERMARK", 0.38),
                Style::Tile => ("GUATERMARK · SAMPLE", 0.24),
                Style::Logo => ("GUATERMARK", 0.55),
            };
            let spec = WatermarkSpec {
                text: text.to_string(),
                style,
                alpha,
                color: [255, 255, 255],
            };
            let (stamped, truth) = watermark::stamp(&clean, &spec)?;
            let style_name = style.label();
            let input_name = format!("{stem}-{style_name}.jpg");
            let input_path = examples.join(&input_name);
            save_jpeg(&stamped, &input_path, 92)?;
            mask::save_mask(
                &truth,
                &examples
                    .join("masks")
                    .join(format!("{stem}-{style_name}.png")),
            )?;

            // Evaluate the file users actually get, compression included.
            let wm = load_rgb(&input_path)?;
            let grown = mask::dilate(&truth, 2);
            let roi = metrics::mask_bbox(&truth, 4).context("empty watermark mask")?;
            let before = Before {
                full: metrics::psnr(&wm, &clean, None),
                roi: metrics::psnr(&wm, &clean, Some(roi)),
                masked: metrics::psnr_masked(&wm, &clean, &truth),
                ssim: metrics::ssim(&wm, &clean, Some(roi)),
            };
            let input_rel = format!("{examples_rel}/{input_name}");

            for (algo, algo_name) in [(Algo::Fast, "fast"), (Algo::Patch, "patch")] {
                for (mask_name, mask_img) in [("exact", &truth), ("grown", &grown)] {
                    let start = Instant::now();
                    let restored = inpaint::inpaint(&wm, mask_img, algo)?;
                    let seconds = start.elapsed().as_secs_f64();
                    let file = format!("{stem}-{style_name}-{algo_name}-{mask_name}.jpg");
                    save_jpeg(&restored, &out_dir.join(&file), 92)?;

                    cases.push(RemovalCase {
                        image: stem.clone(),
                        style: style_name.to_string(),
                        algo: algo_name.to_string(),
                        mask: mask_name.to_string(),
                        input: input_rel.clone(),
                        clean: clean_rel.clone(),
                        output: format!("out/{file}"),
                        psnr_full_before: before.full,
                        psnr_full_after: metrics::psnr(&restored, &clean, None),
                        psnr_roi_before: before.roi,
                        psnr_roi_after: metrics::psnr(&restored, &clean, Some(roi)),
                        psnr_masked_before: before.masked,
                        psnr_masked_after: metrics::psnr_masked(&restored, &clean, &truth),
                        ssim_roi_before: before.ssim,
                        ssim_roi_after: metrics::ssim(&restored, &clean, Some(roi)),
                        seconds,
                    });
                }
            }
        }

        // Enhancement benchmark: clean -> degraded -> restore.
        let degraded = enhance::degrade(&clean, 2, 5.0, 0x5EED_1234);
        let degraded_name = format!("{stem}-enh-degraded.jpg");
        save_jpeg(&degraded, &out_dir.join(&degraded_name), 92)?;
        let degraded_rel = format!("out/{degraded_name}");
        let methods: [(&str, RgbImage, f64); 3] = {
            let start = Instant::now();
            let nearest = enhance::nearest(&degraded, 2);
            let t_nearest = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let lanczos = enhance::lanczos3(&degraded, 2);
            let t_lanczos = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let ours = enhance::enhance(&degraded, 0.5, 0.65, 2);
            let t_ours = start.elapsed().as_secs_f64();
            [
                ("nearest", nearest, t_nearest),
                ("lanczos3", lanczos, t_lanczos),
                ("guatermark", ours, t_ours),
            ]
        };
        for (method, img, seconds) in methods {
            let file = format!("{stem}-enh-{method}.jpg");
            save_jpeg(&img, &out_dir.join(&file), 92)?;
            if method == "guatermark" {
                let (iw, ih) = img.dimensions();
                let cw = 240.min(iw);
                let ch = 180.min(ih);
                let roi = Rect {
                    x: (iw - cw) / 2,
                    y: (ih - ch) / 2,
                    w: cw,
                    h: ch,
                };
                save_zoom(&img, roi, &out_dir.join(format!("{stem}-enh-zoom.jpg")), 2)?;
            }
            enhancement.push(EnhanceCase {
                image: stem.clone(),
                clean: clean_rel.clone(),
                degraded: degraded_rel.clone(),
                method: method.to_string(),
                output: format!("out/{file}"),
                psnr: metrics::psnr(&img, &clean, None),
                ssim: metrics::ssim(&img, &clean, None),
                seconds,
            });
        }
    }

    let report = Report {
        summary: summarize(&cases, &enhancement, files.len()),
        cases,
        enhancement,
    };

    let json_path = dir.join("results.json");
    fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
    let html_path = dir.join("report.html");
    fs::write(&html_path, render_html(&report, examples_rel.as_str())?)?;
    print_summary(&report);
    println!("\nJSON:   {}", json_path.display());
    println!("Report: {}", html_path.display());
    Ok(report)
}

struct Before {
    full: f64,
    roi: f64,
    masked: f64,
    ssim: f64,
}

fn summarize(cases: &[RemovalCase], enhancement: &[EnhanceCase], images: usize) -> Summary {
    let mut removal = Vec::new();
    for (algo, mask_name) in [
        ("fast", "exact"),
        ("patch", "exact"),
        ("fast", "grown"),
        ("patch", "grown"),
    ] {
        let group: Vec<&RemovalCase> = cases
            .iter()
            .filter(|c| c.algo == algo && c.mask == mask_name)
            .collect();
        removal.push(Group {
            label: format!("{algo} + {mask_name} mask"),
            cases: group.len(),
            psnr: mean(group.iter().map(|c| c.psnr_roi_after)),
            ssim: mean(group.iter().map(|c| c.ssim_roi_after)),
            gain_db: mean(group.iter().map(|c| c.psnr_roi_after - c.psnr_roi_before)),
            seconds: mean(group.iter().map(|c| c.seconds)),
        });
    }
    let mut enhancement_groups = Vec::new();
    for method in ["nearest", "lanczos3", "guatermark"] {
        let group: Vec<&EnhanceCase> = enhancement.iter().filter(|c| c.method == method).collect();
        enhancement_groups.push(Group {
            label: method.to_string(),
            cases: group.len(),
            psnr: mean(group.iter().map(|c| c.psnr)),
            ssim: mean(group.iter().map(|c| c.ssim)),
            gain_db: 0.0,
            seconds: mean(group.iter().map(|c| c.seconds)),
        });
    }
    Summary {
        images,
        removal,
        enhancement: enhancement_groups,
    }
}

fn mean<I: Iterator<Item = f64>>(it: I) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for v in it {
        sum += v;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn print_summary(report: &Report) {
    println!(
        "\n=== Watermark removal ({} images, {} cases) ===",
        report.summary.images,
        report.cases.len()
    );
    println!(
        "{:<28} {:>6}  {:>8}  {:>8}  {:>8}  {:>7}",
        "configuration", "cases", "PSNR roi", "gain dB", "SSIM", "time s"
    );
    for g in &report.summary.removal {
        println!(
            "{:<28} {:>6}  {:>8.2}  {:>+8.2}  {:>8.4}  {:>7.2}",
            g.label, g.cases, g.psnr, g.gain_db, g.ssim, g.seconds
        );
    }
    println!("\n=== Enhancement (degraded 2x + noise) ===");
    println!(
        "{:<28} {:>6}  {:>8}  {:>8}  {:>7}",
        "method", "cases", "PSNR", "SSIM", "time s"
    );
    for g in &report.summary.enhancement {
        println!(
            "{:<28} {:>6}  {:>8.2}  {:>8.4}  {:>7.2}",
            g.label, g.cases, g.psnr, g.ssim, g.seconds
        );
    }
}

fn save_zoom(img: &RgbImage, roi: Rect, path: &Path, scale: u32) -> Result<()> {
    let (w, h) = img.dimensions();
    let x = roi.x.min(w.saturating_sub(1));
    let y = roi.y.min(h.saturating_sub(1));
    let cw = roi.w.min(w - x);
    let ch = roi.h.min(h - y);
    let crop = imageops::crop_imm(img, x, y, cw, ch).to_image();
    let zoom = imageops::resize(&crop, cw * scale, ch * scale, imageops::FilterType::Nearest);
    save_jpeg(&zoom, path, 90)
}

fn save_jpeg(img: &RgbImage, path: &Path, quality: u8) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    JpegEncoder::new_with_quality(&mut writer, quality)
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

fn load_rgb(path: &Path) -> Result<RgbImage> {
    Ok(image::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?
        .to_rgb8())
}

fn relative_path(base: &Path, target: &Path) -> String {
    let base = base.canonicalize().unwrap_or_else(|_| base.to_path_buf());
    let target = target
        .canonicalize()
        .unwrap_or_else(|_| target.to_path_buf());
    let mut b: Vec<_> = base.components().collect();
    let mut t: Vec<_> = target.components().collect();
    while !b.is_empty() && !t.is_empty() && b[0] == t[0] {
        b.remove(0);
        t.remove(0);
    }
    let mut out = String::new();
    for _ in 0..b.len() {
        out.push_str("../");
    }
    let tail = t
        .iter()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    out.push_str(&tail);
    if out.is_empty() {
        ".".into()
    } else {
        out
    }
}
