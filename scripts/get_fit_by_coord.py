from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
import astropy.units as u

# 1. Define las coordenadas y el radio de búsqueda

# Orion Nebula
#ra, dec = 83.82, -5.39

# Rho Ophiuchi
#ra, dec = 247.02, -24.54

# Galactic Center
ra, dec = 266.42, -29.01
search_radius = 30 * u.arcsec

coord = SkyCoord(ra, dec, unit='deg')

# 2. Consulta la colección pública de QR2
print(f"Buscando FITS cerca de RA={ra}, Dec={dec}...")
results = Irsa.query_sia(pos=(coord, search_radius), collection='spherex_qr2')

# 3. Muestra los nombres de archivo y metadatos clave
print(f"\nSe encontraron {len(results)} archivos:")
for row in results:
    # El nombre del archivo está en la columna 'access_url' o 'obs_id'
    # La API devuelve una URL completa; nos quedamos con el nombre final.
    url = row['access_url']
    filename = url.split('/')[-1]
    print(f"  - Archivo: {filename}")
    print(f"    Obs ID:  {row['obs_id']}")
    print(f"    RA/Dec:  {row['s_ra']:.4f}, {row['s_dec']:.4f}")
    print(f"    Fecha:   {row['t_min']}")