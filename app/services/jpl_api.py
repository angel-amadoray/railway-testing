"""
Cliente para la API Small-Body Identification de JPL.
https://ssd-api.jpl.nasa.gov/doc/sb_ident.html

Hallazgos empíricos:
- two_pass=false → respuesta en ~2-5s con miles de objetos (viable para HTTP).
- two_pass=true  → >120s, NO viable para endpoint HTTP.
- mag_required=false → JPL devuelve objetos FUERA del FOV (comportamiento
  conservador por incertidumbre orbital). Hay que filtrar server-side.
- mag_required=true  → SÍ respeta el FOV pero descarta objetos sin magnitud.
- HTTP 500 puede aparecer transitoriamente → reintentar.
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

API_URL = "https://ssd-api.jpl.nasa.gov/sb_ident.api"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 2
RETRY_DELAY_S = 2.0


class JPLAPIError(Exception):
    """Error de comunicación o parseo con la API de JPL."""


def _deg_to_hms(ra_deg: float) -> str:
    hours = ra_deg / 15.0
    h = int(hours)
    m = int((hours - h) * 60)
    s = ((hours - h) * 60 - m) * 60
    return f"{h:02d}-{m:02d}-{s:05.2f}"


def _deg_to_dms(dec_deg: float) -> str:
    sign = "M" if dec_deg < 0 else ""
    dec_abs = abs(dec_deg)
    d = int(dec_abs)
    m = int((dec_abs - d) * 60)
    s = ((dec_abs - d) * 60 - m) * 60
    return f"{sign}{d:02d}-{m:02d}-{s:05.2f}"


def _parse_rows(fields: list[str], rows: list[list]) -> list[dict]:
    return [dict(zip(fields, row)) for row in rows]


def _safe_float(value) -> float | None:
    """Convierte 'n.a.', '', '3.E5', '-2.E5' a float. Devuelve None si no es posible."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("n.a.", "na", "nan", "--", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _request_with_retry(params: dict, timeout: int, retries: int) -> dict:
    """GET con reintentos para errores transitorios (5xx, timeouts)."""
    last_err = None
    for attempt in range(1, retries + 2):  # intentos = retries + 1
        try:
            resp = requests.get(API_URL, params=params, timeout=timeout)
            if resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning("JPL attempt %d/%d: %s", attempt, retries + 1, last_err)
                if attempt <= retries:
                    time.sleep(RETRY_DELAY_S)
                    continue
                raise JPLAPIError(f"JPL API {last_err}")
            if resp.status_code != 200:
                raise JPLAPIError(f"JPL API HTTP {resp.status_code}: {resp.text[:500]}")
            try:
                return resp.json()
            except ValueError:
                raise JPLAPIError(f"JPL non-JSON: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            last_err = f"timeout after {timeout}s"
            logger.warning("JPL attempt %d/%d: %s", attempt, retries + 1, last_err)
            if attempt <= retries:
                time.sleep(RETRY_DELAY_S)
                continue
            raise JPLAPIError(f"JPL API {last_err}")
        except requests.exceptions.RequestException as e:
            raise JPLAPIError(f"JPL API request failed: {e}")
    raise JPLAPIError(f"JPL API failed after retries: {last_err}")


def find_known_objects(
    ra_center_deg: float,
    dec_center_deg: float,
    radius_deg: float,
    obs_time_iso: str,
    observer: dict | None = None,
    two_pass: bool = False,
    mag_required: bool = False,
    sb_kind: str = "a",
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    filter_by_offset: bool = True,
    offset_margin: float = 1.4142,          # diagonal del cuadrado (√2)
    max_vmag: float | None = 20.0,          # brillo límite (None = sin filtro)
    max_objects: int | None = 200,          # límite de resultados devueltos
    sort_by: str = "vmag",                  # "vmag" | "offset" | "name"
) -> dict:
    """
    Devuelve los asteroides conocidos dentro del FOV en el instante dado.

    Filtros aplicados (en orden):
        1. Descarta objetos con |offset| > radius_deg * 3600 * offset_margin
        2. Descarta objetos con V > max_vmag (si max_vmag no es None)
        3. Ordena por sort_by ("vmag" ascendente, "offset" ascendente, "name")
        4. Trunca a max_objects (si max_objects no es None)

    Reporta:
        n_objects_raw         → lo que JPL devolvió
        n_objects_after_offset → tras filtrar por offset
        n_objects_after_vmag  → tras filtrar por magnitud
        n_objects             → tras truncar (lo que finalmente se devuelve)
    """
    if not observer:
        raise JPLAPIError(
            "JPL requires an observer. Pasa {'xobs': ...} o {'mpc_code': ...}."
        )

    obs_time = obs_time_iso.replace("T", "_")

    params = {
        "obs-time": obs_time,
        "fov-ra-center": _deg_to_hms(ra_center_deg),
        "fov-dec-center": _deg_to_dms(dec_center_deg),
        "fov-ra-hwidth": radius_deg,
        "fov-dec-hwidth": radius_deg,
        "two-pass": "true" if two_pass else "false",
        "mag-required": "true" if mag_required else "false",
        "sb-kind": sb_kind,
        "req-elem": "false",
    }

    if "xobs" in observer:
        params["xobs"] = observer["xobs"]
    elif "mpc_code" in observer:
        params["mpc-code"] = observer["mpc_code"]
    elif "lat" in observer and "lon" in observer:
        params["lat"] = observer["lat"]
        params["lon"] = observer["lon"]
        params["alt"] = observer.get("alt", 0)
    else:
        raise JPLAPIError(f"Invalid observer dict: {observer}")

    if two_pass:
        params["suppress-first-pass"] = "true"

    logger.info(
        "JPL SB_Ident: two_pass=%s mag_required=%s radius=%.2f° "
        "offset_margin=%.3f max_vmag=%s max_objects=%s",
        two_pass, mag_required, radius_deg, offset_margin, max_vmag, max_objects,
    )

    data = _request_with_retry(params, timeout=timeout, retries=retries)

    if "error" in data:
        raise JPLAPIError(f"JPL API error: {data['error']}")

    summary = data.get("summary", {})
    n_first = summary.get("n_first_pass") or data.get("n_first_pass") or 0
    n_second = summary.get("n_second_pass") or data.get("n_second_pass")

    if two_pass and n_second is not None and n_second > 0:
        fields = data.get("fields_second") or []
        rows = data.get("data_second_pass") or []
        mode = "second_pass"
    else:
        fields = data.get("fields_first") or []
        rows = data.get("data_first_pass") or []
        mode = "first_pass"

    n_objects_raw = len(rows) if rows else 0
    logger.info(
        "JPL raw: mode=%s n_first=%s n_second=%s n_rows=%d",
        mode, n_first, n_second, n_objects_raw,
    )

    base_result = {
        "n_objects_raw": n_objects_raw,
        "n_first_pass": int(n_first or 0),
        "n_second_pass": int(n_second) if n_second is not None else None,
        "mode": mode,
        "fields": fields,
        "observer": data.get("observer", {}),
        "raw_summary": summary,
        "max_offset_arcsec_allowed": (
            radius_deg * 3600 * offset_margin if filter_by_offset else None
        ),
        "max_vmag": max_vmag,
        "max_objects": max_objects,
    }

    if not fields or not rows:
        return {
            **base_result,
            "n_objects": 0,
            "n_objects_after_offset": 0,
            "n_objects_after_vmag": 0,
            "filtered_out_offset": 0,
            "filtered_out_vmag": 0,
            "truncated": 0,
            "objects": [],
        }

    objects = _parse_rows(fields, rows)

    # PASO 1: filtro por offset
    max_offset = radius_deg * 3600 * offset_margin if filter_by_offset else None
    if filter_by_offset and max_offset:
        kept_offset = []
        for obj in objects:
            offset = _safe_float(obj.get('Dist. from center Norm (")'))
            if offset is None or abs(offset) <= max_offset:
                kept_offset.append(obj)
        filtered_out_offset = len(objects) - len(kept_offset)
        objects = kept_offset
    else:
        filtered_out_offset = 0
    n_after_offset = len(objects)

    # PASO 2: filtro por magnitud
    filtered_out_vmag = 0
    if max_vmag is not None:
        kept_vmag = []
        for obj in objects:
            v = _safe_float(obj.get("Visual magnitude (V)"))
            if v is None:
                # Sin magnitud: conservar (podría ser detectable)
                kept_vmag.append(obj)
            elif v <= max_vmag:
                kept_vmag.append(obj)
            else:
                filtered_out_vmag += 1
        objects = kept_vmag
    n_after_vmag = len(objects)

    # PASO 3: ordenar
    if sort_by == "vmag":
        def key_v(o):
            v = _safe_float(o.get("Visual magnitude (V)"))
            return (v is None, v if v is not None else 999.0)
        objects.sort(key=key_v)
    elif sort_by == "offset":
        def key_o(o):
            x = _safe_float(o.get('Dist. from center Norm (")'))
            return (x is None, abs(x) if x is not None else 1e18)
        objects.sort(key=key_o)
    elif sort_by == "name":
        objects.sort(key=lambda o: str(o.get("Object name", "")))
    else:
        raise JPLAPIError(f"sort_by must be vmag|offset|name, got {sort_by!r}")

    # PASO 4: truncar
    truncated = 0
    if max_objects is not None and len(objects) > max_objects:
        truncated = len(objects) - max_objects
        objects = objects[:max_objects]

    n_final = len(objects)
    logger.info(
        "JPL filtered: raw=%d → offset=%d (−%d) → vmag=%d (−%d) → final=%d "
        "(truncated %d, offset_max=%.0f arcsec, max_vmag=%s)",
        n_objects_raw, n_after_offset, filtered_out_offset,
        n_after_vmag, filtered_out_vmag, n_final, truncated,
        max_offset or 0, max_vmag,
    )

    return {
        **base_result,
        "n_objects": n_final,
        "n_objects_after_offset": n_after_offset,
        "n_objects_after_vmag": n_after_vmag,
        "filtered_out_offset": filtered_out_offset,
        "filtered_out_vmag": filtered_out_vmag,
        "truncated": truncated,
        "objects": objects,
    }