use guatermark::enhance::{bilateral, degrade, lanczos3, nearest};
use guatermark::inpaint::{inpaint, Algo};
use guatermark::mask::dilate;
use guatermark::metrics::{mask_bbox, psnr, psnr_masked, ssim};
use guatermark::watermark::{stamp, Style, WatermarkSpec};
use image::{GrayImage, Luma, Rgb, RgbImage};

fn gradient(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        Rgb([
            (x * 3) as u8 % 255,
            (y * 5) as u8 % 255,
            ((x + y) * 2) as u8 % 255,
        ])
    })
}

fn textured(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        Rgb([
            ((x * 7 + y * 3) % 255) as u8,
            ((x * 3 + y * 11) % 255) as u8,
            90,
        ])
    })
}

fn photo() -> RgbImage {
    RgbImage::from_fn(120, 90, |x, y| Rgb([(x * 2) as u8, (y * 2) as u8, 120]))
}

#[test]
fn metricas_de_imagenes_identicas_son_perfectas() {
    let img = gradient(32, 24);
    assert_eq!(psnr(&img, &img, None), 100.0);
    assert!((ssim(&img, &img, None) - 1.0).abs() < 1e-6);
}

#[test]
fn el_ruido_baja_las_metricas() {
    let a = gradient(32, 24);
    let mut b = a.clone();
    b.put_pixel(5, 5, Rgb([255, 0, 255]));
    assert!(psnr(&a, &b, None) < 100.0);
    assert!(ssim(&a, &b, None) < 1.0);
}

#[test]
fn la_caja_de_la_mascara_y_el_psnr_mapeado_funcionan() {
    let a = gradient(32, 24);
    let mut mask = GrayImage::new(32, 24);
    mask.put_pixel(10, 8, Luma([255]));
    let rect = mask_bbox(&mask, 2).expect("mask bbox");
    assert_eq!((rect.x, rect.y), (8, 6));
    let mut b = a.clone();
    b.put_pixel(10, 8, Rgb([0, 0, 0]));
    assert!(psnr_masked(&a, &b, &mask) < 100.0);
    let mut outside = b.clone();
    outside.put_pixel(10, 8, *a.get_pixel(10, 8));
    assert_eq!(psnr_masked(&a, &outside, &mask), 100.0);
}

#[test]
fn la_dilatacion_crece_la_mascara() {
    let mut mask = GrayImage::new(20, 20);
    mask.put_pixel(10, 10, Luma([255]));
    let grown = dilate(&mask, 2);
    let count = grown.pixels().filter(|p| p.0[0] > 127).count();
    assert_eq!(count, 25);
    assert_eq!(grown.get_pixel(10, 10).0[0], 255);
    assert_eq!(grown.get_pixel(13, 10).0[0], 0);
}

#[test]
fn el_upscale_llega_al_tamano_exacto() {
    let img = textured(30, 20);
    assert_eq!(lanczos3(&img, 2).dimensions(), (60, 40));
    assert_eq!(nearest(&img, 3).dimensions(), (90, 60));
}

#[test]
fn lanczos_conserva_un_color_plano() {
    let img = RgbImage::from_pixel(20, 20, Rgb([77, 88, 99]));
    let up = lanczos3(&img, 2);
    for p in up.pixels() {
        assert_eq!(p.0, [77, 88, 99]);
    }
}

#[test]
fn el_denoise_reduce_el_ruido() {
    let img = RgbImage::from_pixel(40, 40, Rgb([100, 100, 100]));
    let noisy = degrade(&img, 1, 25.0, 7);
    let smooth = bilateral(&noisy, 2, 60.0);
    let var = |i: &RgbImage| {
        let mut s = 0.0;
        for p in i.pixels() {
            let d = p.0[0] as f32 - 100.0;
            s += d * d;
        }
        s / (i.width() * i.height()) as f32
    };
    assert!(var(&smooth) < var(&noisy));
}

#[test]
fn la_degradacion_conserva_el_tamano_esperado() {
    let img = textured(40, 30);
    let low = degrade(&img, 2, 6.0, 1);
    assert_eq!(low.dimensions(), (20, 15));
}

#[test]
fn el_relleno_cambia_el_hueco_y_respeta_lo_conocido() {
    let mut img = textured(64, 48);
    let mut mask = GrayImage::new(64, 48);
    for y in 10..20 {
        for x in 15..35 {
            img.put_pixel(x, y, Rgb([255, 255, 255]));
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    for algo in [Algo::Fast, Algo::Patch] {
        let out = inpaint(&img, &mask, algo).expect("inpaint");
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
fn patch_es_deterministico() {
    let img = textured(80, 60);
    let mut mask = GrayImage::new(80, 60);
    for y in 20..35 {
        for x in 25..45 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    let a = inpaint(&img, &mask, Algo::Patch).expect("first");
    let b = inpaint(&img, &mask, Algo::Patch).expect("second");
    assert_eq!(a.as_raw(), b.as_raw());
}

fn corner_spec() -> WatermarkSpec {
    WatermarkSpec {
        text: "GUATERMARK".into(),
        style: Style::Corner,
        alpha: 0.4,
        color: [255, 255, 255],
    }
}

#[test]
fn la_marca_en_esquina_solo_afecta_su_mascara() {
    let img = photo();
    let (out, mask) = stamp(&img, &corner_spec()).expect("stamp");
    let mut marked = 0;
    for (x, y, p) in out.enumerate_pixels() {
        if mask.get_pixel(x, y).0[0] > 127 {
            marked += 1;
            assert_ne!(p, img.get_pixel(x, y));
        } else {
            assert_eq!(p, img.get_pixel(x, y), "pixel outside mask changed");
        }
    }
    assert!(marked > 100, "mask too small: {marked}");
}

#[test]
fn la_marca_en_mosaico_cubre_una_fraccion() {
    let img = photo();
    let spec = WatermarkSpec {
        text: "SAMPLE".into(),
        style: Style::Tile,
        alpha: 0.25,
        color: [255, 255, 255],
    };
    let (out, mask) = stamp(&img, &spec).expect("stamp");
    let marked = mask.pixels().filter(|p| p.0[0] > 127).count();
    let total = (img.width() * img.height()) as usize;
    assert!(marked > total / 100);
    assert!(marked < total / 2);
    assert_eq!(out.dimensions(), img.dimensions());
}
