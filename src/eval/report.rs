use anyhow::{Context, Result};
use std::fmt::Write as _;

use super::{EnhanceCase, RemovalCase, Report};
pub(super) fn render_html(report: &Report, examples_rel: &str) -> Result<String> {
    let mut h = String::new();
    let s = &report.summary;
    let best = s
        .removal
        .iter()
        .find(|g| g.label.starts_with("fast + exact"))
        .context("missing fast+exact group")?;
    let other = s
        .removal
        .iter()
        .find(|g| g.label.starts_with("patch + exact"))
        .context("missing patch+exact group")?;
    let ours = s
        .enhancement
        .iter()
        .find(|g| g.label == "guatermark")
        .context("missing enhancement group")?;
    let lanczos = s
        .enhancement
        .iter()
        .find(|g| g.label == "lanczos3")
        .context("missing lanczos group")?;
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
    )?;

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
        let Some(first) = group.first() else {
            continue;
        };
        let fast_exact = group
            .iter()
            .find(|c| c.algo == "fast" && c.mask == "exact")
            .context("missing fast exact case")?;
        let patch_exact = group
            .iter()
            .find(|c| c.algo == "patch" && c.mask == "exact")
            .context("missing patch exact case")?;
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
        )?;
        for c in &group {
            write!(
                h,
                "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td class=\"good\">{:.2}</td><td>{:.2}</td><td>{:.4}</td><td>{:.2}s</td></tr>",
                c.algo, c.mask, c.psnr_masked_before, c.psnr_masked_after, c.psnr_roi_after, c.ssim_roi_after, c.seconds
            )
            ?;
        }
        write!(h, "</tbody></table></div>")?;
    }

    write!(
        h,
        r#"<h2>Quality enhancement</h2>
<p>The clean photo is downscaled 2x and given Gaussian noise (σ=5), then restored to the original size.
The baseline is a plain Lanczos3 upscale; <b>guatermark</b> adds bilateral denoise and luma unsharp masking.</p>
"#
    )?;
    let mut images: Vec<String> = report.enhancement.iter().map(|c| c.image.clone()).collect();
    images.sort();
    images.dedup();
    for image in images {
        let group: Vec<&EnhanceCase> = report
            .enhancement
            .iter()
            .filter(|c| c.image == image)
            .collect();
        let Some(first) = group.first() else {
            continue;
        };
        let clean = &first.clean;
        let degraded = &first.degraded;
        let nearest = group
            .iter()
            .find(|c| c.method == "nearest")
            .context("missing nearest case")?;
        let lanc = group
            .iter()
            .find(|c| c.method == "lanczos3")
            .context("missing lanczos case")?;
        let ours = group
            .iter()
            .find(|c| c.method == "guatermark")
            .context("missing guatermark case")?;
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
        )?;
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
    )?;
    Ok(h)
}
