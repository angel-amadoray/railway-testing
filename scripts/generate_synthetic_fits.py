# scripts/generate_synthetic_fits.py
"""
Genera un MEF sintético con la estructura de un Spectral Image de SPHEREx (Level 2).

Uso:
    python scripts/generate_synthetic_fits.py \
        --output data/sample_fits/orion_nebula/2025-04/det1_synthetic.fits \
        --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def make_stars(image, n_stars=30, size=256, seed=0):
    """Añade estrellas gaussianas aleatorias a la imagen."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_stars):
        cx, cy = rng.uniform(0, size, size=2)
        amp = rng.uniform(5, 50)
        sigma = rng.uniform(0.8, 2.5)
        image += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return image


def make_asteroid(image, cx, cy, amp=20, sigma=2.0):
    """Coloca un 'asteroide' gaussiano en una posición dada."""
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    image += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return image


def build_spectral_wcs(n_pix=256, lam_min=0.75, lam_max=1.10):
    """
    Crea un WCS espectral lineal simple para 256x256.
    Cada fila (eje Y) mapea a una longitud de onda distinta.
    """
    w = WCS(naxis=2)
    w.wcs.ctype = ["WAVE", "LINEAR"]
    w.wcs.cunit = ["um", ""]
    w.wcs.cdelt = [-(lam_max - lam_min) / n_pix, 1.0]
    w.wcs.crpix = [1, 1]
    w.wcs.crval = [lam_max, 0]
    return w


def build_wcs_wave_table(n_pix=256, lam_min=0.75, lam_max=1.10):
    """
    Construye la extensión WCS-WAVE como una BinTable.
    Columnas: WAVELENGTH (E), BANDWIDTH (E), PIXEL (J)
    """
    lam = np.linspace(lam_max, lam_min, n_pix)
    bandwidth = np.full(n_pix, (lam_max - lam_min) / n_pix)
    pixel = np.arange(n_pix)
    col1 = fits.Column(name="WAVELENGTH", format="E", array=lam)
    col2 = fits.Column(name="BANDWIDTH", format="E", array=bandwidth)
    col3 = fits.Column(name="PIXEL", format="J", array=pixel)
    return fits.BinTableHDU.from_columns([col1, col2, col3], name="WCS-WAVE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Ruta del FITS de salida")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=256, help="Tamaño espacial (default 256)")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = args.size
    rng = np.random.default_rng(args.seed)

    # 1) IMAGE: fondo + estrellas + asteroide
    image = np.full((n, n), 1.0, dtype=np.float32)  # fondo
    image = make_stars(image, n_stars=25, size=n, seed=args.seed)
    # Asteroide desplazado según la semilla (para que dos épocas difieran)
    ax = n * 0.3 + (args.seed % 5) * 3
    ay = n * 0.7 + (args.seed % 7) * 2
    image = make_asteroid(image, ax, ay, amp=15, sigma=1.8)

    # 2) FLAGS: todo válido
    flags = np.zeros((n, n), dtype=np.int32)

    # 3) VARIANCE: ruido constante
    variance = np.full((n, n), 0.01, dtype=np.float32)

    # 4) ZODI: cero
    zodi = np.zeros((n, n), dtype=np.float32)

    # 5) PSF: 11x11x5 gaussianas sintéticas
    psf = np.zeros((11, 11, 5), dtype=np.float32)
    yy, xx = np.mgrid[0:11, 0:11]
    for i in range(5):
        psf[:, :, i] = np.exp(-((xx - 5) ** 2 + (yy - 5) ** 2) / (2 * 1.5 ** 2))

    # 6) WCS-WAVE
    wcs_wave_hdu = build_wcs_wave_table(n, lam_min=0.75, lam_max=1.10)

    # Encabezado primario con WCS espectral alternativo (key='W')
    pri_hdr = fits.Header()
    pri_hdr["TELESCOP"] = "SPHEREx"
    pri_hdr["INSTRUME"] = "SPHEREx"
    pri_hdr["DETECTOR"] = 1
    pri_hdr["BANDW"] = 1
    # WCS espectral lineal accesible con WCS(..., key='W')
    w_spectral = build_spectral_wcs(n, 0.75, 1.10)
    pri_hdr.update(w_spectral.to_header(key="W"))

    primary = fits.PrimaryHDU(header=pri_hdr)
    hdu_image = fits.ImageHDU(data=image, name="IMAGE")
    hdu_flags = fits.ImageHDU(data=flags, name="FLAGS")
    hdu_var = fits.ImageHDU(data=variance, name="VARIANCE")
    hdu_zodi = fits.ImageHDU(data=zodi, name="ZODI")
    hdu_psf = fits.ImageHDU(data=psf, name="PSF")

    hdul = fits.HDUList([
        primary,
        hdu_image,
        hdu_flags,
        hdu_var,
        hdu_zodi,
        hdu_psf,
        wcs_wave_hdu,
    ])
    hdul.writeto(out_path, overwrite=True)
    print(f"✅ FITS sintético generado: {out_path} ({out_path.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()