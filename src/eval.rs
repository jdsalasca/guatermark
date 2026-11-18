use crate::enhance;
use crate::inpaint::{self, Algo};
use crate::mask;
use crate::metrics::{self, Rect};
use crate::watermark::{self, Style, WatermarkSpec};
use anyhow::{bail, Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{imageops, ImageEncoder, RgbImage};
use serde::Serialize;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
        let clean_rel = format!("clean/{}", path.file_name().unwrap().to_string_lossy());

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
    fs::write(&html_path, render_html(&report, examples_rel.as_str()))?;
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

fn render_html(report: &Report, examples_rel: &str) -> String {
    let mut h = String::new();
    let s = &report.summary;
    let best = s
        .removal
        .iter()
        .find(|g| g.label.starts_with("fast + exact"))
        .unwrap_or(&s.removal[0]);
    let other = s
        .removal
        .iter()
        .find(|g| g.label.starts_with("patch + exact"))
        .unwrap_or(&s.removal[0]);
    let ours = s
        .enhancement
        .iter()
        .find(|g| g.label == "guatermark")
        .unwrap_or(&s.enhancement[0]);
    let lanczos = s
        .enhancement
        .iter()
        .find(|g| g.label == "lanczos3")
        .unwrap_or(&s.enhancement[0]);
    let mean_before = report
        .cases
        .iter()
        .filter(|c| c.algo == "fast" && c.mask == "exact")
        .map(|c| c.psnr_roi_before)
        .sum::<f64>()
        / report
            .cases
            .iter()
            .filter(|c| c.algo == "fast" && c.mask == "exact")
            .count()
            .max(1) as f64;
    write!(
        h,
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>guatermark benchmark</title>
<style>
:root{{--bg:#0b0e13;--card:#141922;--line:#253040;--fg:#e7edf5;--muted:#93a1b5;--accent:#5eead4}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:2.1rem;margin:0 0 6px;letter-spacing:-.02em}}
h2{{margin:52px 0 6px;font-size:1.35rem;border-bottom:1px solid var(--line);padding-bottom:10px}}
h3{{margin:34px 0 12px;font-size:1.02rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.06em}}
p{{color:var(--muted);max-width:78ch}}
a{{color:var(--accent)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:26px 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.stat b{{display:block;font-size:1.55rem;color:var(--accent);letter-spacing:-.02em}}
.stat span{{color:var(--muted);font-size:.85rem}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin:18px 0}}
.shots{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}}
.shots figure{{margin:0}}
.shots img{{width:100%;border-radius:8px;display:block;background:#000}}
.shots figcaption{{font-size:.75rem;color:var(--muted);margin-top:6px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{text-align:right;padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
th{{color:var(--muted);font-weight:500}}
td.good{{color:var(--accent)}}
.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.82rem}}
.tag{{display:inline-block;padding:2px 7px;border-radius:999px;font-size:.72rem;border:1px solid var(--line);color:var(--muted);margin-right:6px}}
.note{{border-left:3px solid var(--accent);padding:2px 0 2px 14px;margin:20px 0;color:var(--muted)}}
footer{{margin-top:60px;color:var(--muted);font-size:.8rem}}
@media(max-width:820px){{.shots{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>guatermark <span style="color:var(--muted);font-weight:400">benchmark</span></h1>
<p>Watermark removal and quality enhancement, implemented 100% in Rust. Every number below comes from
<code>guatermark eval</code> on {images} clean ground-truth photos ({cases} removal cases). The watermark,
its exact pixel mask and the degraded input are generated deterministically, so the benchmark is reproducible.</p>
<div class="stats">
<div class="stat"><b>{psnr:.1} dB</b><span>ROI PSNR after diffusion inpainting (exact mask, from {before:.1} dB)</span></div>
<div class="stat"><b>{ssim:.3}</b><span>ROI SSIM after removal</span></div>
<div class="stat"><b>{other_psnr:.1} dB</b><span>patch synthesis with the same mask</span></div>
<div class="stat"><b>{enh_psnr:.2} dB</b><span>enhancement PSNR (Lanczos3 baseline {lanc_psnr:.2} dB)</span></div>
</div>
<p class="note">Removal is mask-based, like IOPaint or Photoshop: automatic detection of arbitrary watermarks is not
solved. “exact mask” is the true watermark footprint; “grown mask” adds 2 px to cover soft edges and JPEG halos,
as <code>--grow 2</code> would. ROI PSNR/SSIM are measured around the watermark, where the change happens.</p>
<h2>Watermark removal</h2>
<p>Images are stamped, saved as JPEG (q92) and then restored. <span class="tag">fast</span> multi-scale diffusion
(recommended), <span class="tag">patch</span> exemplar-based synthesis. Restored images are shown in full.</p>
"#,
        images = s.images,
        cases = report.cases.len(),
        psnr = best.psnr,
        before = mean_before,
        ssim = best.ssim,
        other_psnr = other.psnr,
        enh_psnr = ours.psnr,
        lanc_psnr = lanczos.psnr,
    )
    .unwrap();

    let mut keys: Vec<(String, String)> = report
        .cases
        .iter()
        .map(|c| (c.image.clone(), c.style.clone()))
        .collect();
    keys.sort();
    keys.dedup();
    for (image, style) in keys {
        let group: Vec<&RemovalCase> = report
            .cases
            .iter()
            .filter(|c| c.image == image && c.style == style)
            .collect();
        let first = group[0];
        let fast_exact = group
            .iter()
            .find(|c| c.algo == "fast" && c.mask == "exact")
            .unwrap();
        let patch_exact = group
            .iter()
            .find(|c| c.algo == "patch" && c.mask == "exact")
            .unwrap();
        write!(
            h,
            r#"<div class="case"><h3>{image} · {style}</h3>
<div class="shots">
<figure><img loading="lazy" src="{examples_rel}/{image}-{style}.jpg" alt="watermarked"><figcaption>watermarked (input)</figcaption></figure>
<figure><img loading="lazy" src="{clean}" alt="clean"><figcaption>clean (ground truth)</figcaption></figure>
<figure><img loading="lazy" src="{out1}" alt="fast exact"><figcaption>fast + exact mask</figcaption></figure>
<figure><img loading="lazy" src="{out2}" alt="patch exact"><figcaption>patch + exact mask</figcaption></figure>
</div>
<table><thead><tr><th>algorithm</th><th>mask</th><th>PSNR on mark before</th><th>after</th><th>ROI PSNR</th><th>SSIM ROI</th><th>time</th></tr></thead><tbody>
"#,
            image = image,
            style = style,
            examples_rel = examples_rel,
            clean = first.clean,
            out1 = fast_exact.output,
            out2 = patch_exact.output,
        )
        .unwrap();
        for c in &group {
            write!(
                h,
                "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td class=\"good\">{:.2}</td><td>{:.2}</td><td>{:.4}</td><td>{:.2}s</td></tr>",
                c.algo, c.mask, c.psnr_masked_before, c.psnr_masked_after, c.psnr_roi_after, c.ssim_roi_after, c.seconds
            )
            .unwrap();
        }
        write!(h, "</tbody></table></div>").unwrap();
    }

    write!(
        h,
        r#"<h2>Quality enhancement</h2>
<p>The clean photo is downscaled 2x and given Gaussian noise (σ=5), then restored to the original size.
The baseline is a plain Lanczos3 upscale; <b>guatermark</b> adds bilateral denoise and luma unsharp masking.</p>
"#
    )
    .unwrap();
    let mut images: Vec<String> = report.enhancement.iter().map(|c| c.image.clone()).collect();
    images.sort();
    images.dedup();
    for image in images {
        let group: Vec<&EnhanceCase> = report
            .enhancement
            .iter()
            .filter(|c| c.image == image)
            .collect();
        let clean = &group[0].clean;
        let degraded = &group[0].degraded;
        let get = |m: &str| group.iter().find(|c| c.method == m).unwrap();
        let nearest = get("nearest");
        let lanc = get("lanczos3");
        let ours = get("guatermark");
        let zoom = format!("out/{image}-enh-zoom.jpg");
        write!(
            h,
            r#"<div class="case"><h3>{image}</h3>
<div class="shots">
<figure><img loading="lazy" src="{clean}" alt="clean"><figcaption>clean</figcaption></figure>
<figure><img loading="lazy" src="{degraded}" alt="degraded"><figcaption>degraded 2x + noise</figcaption></figure>
<figure><img loading="lazy" src="{near}" alt="nearest"><figcaption>nearest {near_psnr:.2} dB</figcaption></figure>
<figure><img loading="lazy" src="{lanc}" alt="lanczos"><figcaption>Lanczos3 {lanc_psnr:.2} dB</figcaption></figure>
</div>
<div class="shots" style="grid-template-columns:repeat(2,1fr)">
<figure><img loading="lazy" src="{ours}" alt="guatermark"><figcaption>guatermark {ours_psnr:.2} dB · SSIM {ours_ssim:.4} · {secs:.2}s</figcaption></figure>
<figure><img loading="lazy" src="{zoom}" alt="detail"><figcaption>detail: guatermark</figcaption></figure>
</div></div>
"#,
            image = image,
            clean = clean,
            degraded = degraded,
            near = nearest.output,
            lanc = lanc.output,
            ours = ours.output,
            near_psnr = nearest.psnr,
            lanc_psnr = lanc.psnr,
            ours_psnr = ours.psnr,
            ours_ssim = ours.ssim,
            secs = ours.seconds,
            zoom = zoom,
        )
        .unwrap();
    }

    write!(
        h,
        r#"<h2>How to reproduce</h2>
<p class="mono">cargo run --release -- eval</p>
<p>Source images: <a href="https://picsum.photos">Lorem Picsum</a> / Unsplash (see ATTRIBUTION.md).
Watermarks are synthetic and applied at evaluation time. Metrics: RGB PSNR and luminance SSIM with an
11×11 Gaussian window (σ=1.5); “ROI” is the watermark bounding box, “on mark” is computed only on the
watermark pixels.</p>
<footer>guatermark · MIT · 100% Rust (no Python, no ONNX, no external models)</footer>
</main></body></html>
"#
    )
    .unwrap();
    h
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
