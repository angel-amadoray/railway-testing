from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import glob

folder = Path('data/sample_fits/orion_nebula/2025-09')
files = sorted(glob.glob(str(folder / '*.fits')))

def check(ra, dec, label):
    print(f'\n=== {label} → RA={ra:.3f}, Dec={dec:.3f} ===')
    visible = []
    for f in files:
        with fits.open(f) as hdul:
            hdr = hdul['IMAGE'].header
            det = hdr.get('DETECTOR')
            wcs = WCS(hdr, fobj=hdul)
            try:
                x, y = wcs.all_world2pix(ra, dec, 0)
                ny, nx = hdul['IMAGE'].data.shape
                in_frame = (0 <= x < nx) and (0 <= y < ny)
                status = f'in_frame ({x:.0f},{y:.0f})' if in_frame else f'FUERA  ({x:.0f},{y:.0f})'
                print(f'  D{det}: {status}')
                if in_frame:
                    visible.append(det)
            except Exception as e:
                print(f'  D{det}: error ({e})')
    print(f'  → Visible en {len(visible)}/6 detectores: {visible}')
    return visible

# Test 1: centro de D1 (donde probablemente clicaste)
check(88.3123, -6.7469, 'Centro de D1')

# Test 2: centro de D2 (zona de overlap)
check(84.4267, -6.9269, 'Centro de D2')

# Test 3: punto medio entre D1 y D3
check(84.4200, -6.8500, 'Zona de overlap (RA~84.4)')