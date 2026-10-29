import requests
import json

URL = "https://ssd-api.jpl.nasa.gov/sb_ident.api"

# Vector de estado del FITS (posición + velocidad, ICRF)
XOBS_FULL = "3357.540286915801,-2950.2970881889973,-5442.019747466086,3.1889978886327373,-4.953464629399665,4.66464957446033"
XOBS_POS_ONLY = "3357.540286915801,-2950.2970881889973,-5442.019747466086"

# Campo de prueba: centro en la eclíptica (RA 10h, Dec +10°), radio 3°
BASE = {
    "fov-ra-center": "10-00-00",
    "fov-dec-center": "10-00-00",
    "fov-ra-hwidth": 3.0,
    "fov-dec-hwidth": 3.0,
    "sb-kind": "a",
    "req-elem": "false",
}

configs = [
    {
        "name": "A) two-pass=false, mag-required=false, xobs completo",
        "params": {**BASE, "obs-time": "2025-05-01_12:15:52",
                   "two-pass": "false", "mag-required": "false", "xobs": XOBS_FULL},
    },
    {
        "name": "B) two-pass=false, mag-required=false, solo posición",
        "params": {**BASE, "obs-time": "2025-05-01_12:15:52",
                   "two-pass": "false", "mag-required": "false", "xobs": XOBS_POS_ONLY},
    },
    {
        "name": "C) two-pass=true, mag-required=false, xobs completo",
        "params": {**BASE, "obs-time": "2025-05-01_12:15:52",
                   "two-pass": "true", "suppress-first-pass": "true",
                   "mag-required": "false", "xobs": XOBS_FULL},
    },
]

for cfg in configs:
    print(f"\n{'='*70}\n{cfg['name']}\n{'='*70}")
    try:
        r = requests.get(URL, params=cfg["params"], timeout=120)
        print(f"HTTP {r.status_code}")
        data = r.json()
        # Imprimir solo lo relevante
        print(f"n_first_pass:  {data.get('n_first_pass', 'N/A')}")
        print(f"n_second_pass: {data.get('n_second_pass', 'N/A')}")
        print(f"n_objects:     {data.get('n_objects', 'N/A')}")
        if data.get("data_first_pass"):
            print(f"Primeros 3 objetos: {data['data_first_pass'][:3]}")
        if data.get("data_second_pass"):
            print(f"Primeros 3 objetos (2nd): {data['data_second_pass'][:3]}")
        # Si hay error, mostrarlo
        if "error" in data:
            print(f"ERROR: {data['error']}")
    except Exception as e:
        print(f"Excepción: {e}")