use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use guatermark::enhance;
use guatermark::inpaint::{self, Algo};
use guatermark::mask;
use guatermark::watermark::{self, Style, WatermarkSpec};
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Watermark removal and image quality enhancement, 100% Rust.
#[derive(Parser)]
#[command(name = "guatermark", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Remove a watermark using a mask
    Remove(RemoveArgs),
    /// Denoise, upscale (Lanczos3) and sharpen an image
    Enhance(EnhanceArgs),
    /// Stamp a synthetic watermark (used to build the examples)
    Stamp(StampArgs),
    /// Run the reproducible benchmark in eval/
    Eval(EvalArgs),
}

#[derive(Args)]
struct RemoveArgs {
    /// Input image (jpg, png, webp)
    input: PathBuf,
    /// Output image
    output: PathBuf,
    /// Mask image: white or opaque = remove (PNG alpha or luminance)
    #[arg(long)]
    mask: PathBuf,
    /// Grow the mask by this many pixels to cover anti-aliased and JPEG halos
    #[arg(long, default_value_t = 0)]
    grow: u32,
    /// Inpainting algorithm
    #[arg(long, value_enum, default_value_t = AlgoArg::Fast)]
    algo: AlgoArg,
}

#[derive(Args)]
struct EnhanceArgs {
    input: PathBuf,
    output: PathBuf,
    /// Bilateral denoise strength, 0 disables it
    #[arg(long, default_value_t = 0.3)]
    denoise: f32,
    /// Unsharp mask amount, 0 disables it
    #[arg(long, default_value_t = 0.5)]
    sharpen: f32,
    /// Integer upscale factor (Lanczos3)
    #[arg(long, default_value_t = 1)]
    scale: u32,
}

#[derive(Args)]
struct StampArgs {
    input: PathBuf,
    output: PathBuf,
    #[arg(long, default_value = "GUATERMARK")]
    text: String,
    #[arg(long, value_enum, default_value_t = Style::Corner)]
    style: Style,
    #[arg(long, default_value_t = 0.35)]
    alpha: f32,
    /// Write the exact affected-pixel mask too
    #[arg(long)]
    save_mask: Option<PathBuf>,
}

#[derive(Args)]
struct EvalArgs {
    /// Directory with clean/ photos and where report.html is written
    #[arg(long, default_value = "eval")]
    dir: PathBuf,
    /// Directory for the generated watermarked examples
    #[arg(long, default_value = "input-examples")]
    examples: PathBuf,
}

#[derive(Clone, Copy, ValueEnum)]
enum AlgoArg {
    Fast,
    Patch,
}

impl From<AlgoArg> for Algo {
    fn from(value: AlgoArg) -> Self {
        match value {
            AlgoArg::Fast => Algo::Fast,
            AlgoArg::Patch => Algo::Patch,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Remove(a) => remove(a),
        Command::Enhance(a) => enhance_image(a),
        Command::Stamp(a) => stamp(a),
        Command::Eval(a) => guatermark::eval::run(&a.dir, &a.examples).map(|_| ()),
    }
}

fn remove(a: RemoveArgs) -> Result<()> {
    let img = load_rgb(&a.input)?;
    let mut mask_img = mask::load_mask(&a.mask)?;
    if a.grow > 0 {
        mask_img = mask::dilate(&mask_img, a.grow);
    }
    let covered = mask_img.pixels().filter(|p| p.0[0] > 127).count();
    if covered == 0 {
        bail!("the mask is empty; nothing to remove");
    }
    let total = (img.width() * img.height()) as usize;
    let start = Instant::now();
    let out = inpaint::inpaint(&img, &mask_img, a.algo.into())?;
    let seconds = start.elapsed().as_secs_f64();
    save_image(&out, &a.output)?;
    println!(
        "removed {} pixels ({:.1}% of the image) with {:?} in {:.2}s -> {}",
        covered,
        100.0 * covered as f64 / total as f64,
        match a.algo {
            AlgoArg::Fast => Algo::Fast,
            AlgoArg::Patch => Algo::Patch,
        },
        seconds,
        a.output.display()
    );
    Ok(())
}

fn enhance_image(a: EnhanceArgs) -> Result<()> {
    if !(0.0..=1.0).contains(&a.denoise) {
        bail!("--denoise must be between 0 and 1");
    }
    if !(0.0..=2.0).contains(&a.sharpen) {
        bail!("--sharpen must be between 0 and 2");
    }
    if !(1..=8).contains(&a.scale) {
        bail!("--scale must be between 1 and 8");
    }
    let img = load_rgb(&a.input)?;
    let start = Instant::now();
    let out = enhance::enhance(&img, a.denoise, a.sharpen, a.scale);
    let seconds = start.elapsed().as_secs_f64();
    save_image(&out, &a.output)?;
    println!(
        "{}x{} -> {}x{} in {:.2}s -> {}",
        img.width(),
        img.height(),
        out.width(),
        out.height(),
        seconds,
        a.output.display()
    );
    Ok(())
}

fn stamp(a: StampArgs) -> Result<()> {
    let img = load_rgb(&a.input)?;
    let spec = WatermarkSpec {
        text: a.text,
        style: a.style,
        alpha: a.alpha,
        color: [255, 255, 255],
    };
    let (out, mask_img) = watermark::stamp(&img, &spec)?;
    save_image(&out, &a.output)?;
    if let Some(path) = &a.save_mask {
        mask::save_mask(&mask_img, path)?;
    }
    println!(
        "{} watermark with alpha {} -> {}",
        a.style.label(),
        a.alpha,
        a.output.display()
    );
    Ok(())
}

fn load_rgb(path: &Path) -> Result<RgbImage> {
    Ok(image::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?
        .to_rgb8())
}

fn save_image(img: &RgbImage, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(path)
        .with_context(|| format!("cannot save {}", path.display()))
}
