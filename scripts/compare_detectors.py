from pathlib import Path
from astropy.io import fits
import glob

folder = Path('data/sample_fits/orion_nebula/2025-09')
files = sorted(glob.glob(str(folder / '*.fits')))

print(f'Analizando {len(files)} FITS...\n')
print(f'{"Detector":<10} {"RA":<12} {"Dec":<12} {"ObsId":<18} {"Date-OBS"}')
print('-' * 80)

for f in files:
    with fits.open(f) as h:
        hdr = h['IMAGE'].header
        det = hdr.get('DETECTOR', '?')
        ra = hdr.get('CRVAL1', 0)
        dec = hdr.get('CRVAL2', 0)
        obsid = hdr.get('OBSID', '?')
        date = hdr.get('DATE-OBS', '?')[:19]
        print(f'D{det:<9} {ra:<12.4f} {dec:<12.4f} {obsid:<18} {date}')