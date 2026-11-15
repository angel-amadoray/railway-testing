"""
Client for the JPL Small-Body Identification API.
https://ssd-api.jpl.nasa.gov/doc/sb_ident.html

Empirical findings:
- two_pass=false → response in ~2-5s with thousands of objects (viable for HTTP).
- two_pass=true  → >120s, NOT viable for an HTTP endpoint.
- mag_required=false → JPL returns objects OUTSIDE the FOV (conservative
  behavior due to orbital uncertainty). Server-side filtering is required.
- mag_required=true  → DOES respect the FOV but discards objects without magnitude.
- HTTP 500 may appear transiently → retry.
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
    """Error specific to JPL API communication, responses, or parsing."""


def _deg_to_hms(ra_deg: float) -> str:
    """Convert decimal right ascension to the hour format required by JPL.

    Right ascension is interpreted in degrees and divided by 15 to obtain
    hours. The result uses ``HH-MM-SS.ss``.

    Args:
        ra_deg: Right ascension in degrees.

    Returns:
        String with hours, minutes, and seconds separated by hyphens.
    """
    hours = ra_deg / 15.0
    h = int(hours)
    m = int((hours - h) * 60)
    s = ((hours - h) * 60 - m) * 60
    return f"{h:02d}-{m:02d}-{s:05.2f}"


def _deg_to_dms(dec_deg: float) -> str:
    """Convert decimal declination to JPL's sexagesimal format.

    Args:
        dec_deg: Declination in degrees; negative values preserve their sign.

    Returns:
        ``DD-MM-SS.ss`` string; negative values use the ``M`` prefix expected
        by the API format.
    """
    sign = "M" if dec_deg < 0 else ""
    dec_abs = abs(dec_deg)
    d = int(dec_abs)
    m = int((dec_abs - d) * 60)
    s = ((dec_abs - d) * 60 - m) * 60
    return f"{sign}{d:02d}-{m:02d}-{s:05.2f}"


def _parse_rows(fields: list[str], rows: list[list]) -> list[dict]:
    """Transform JPL tabular rows into a list of dictionaries.

    Each element of ``fields`` is paired with the value at the same position
    in each row using ``zip``.

    Args:
        fields: Column names returned by JPL.
        rows: Data rows in the order of those columns.

    Returns:
        A list where each row is represented as a ``dict``.
    """
    return [dict(zip(fields, row)) for row in rows]


def _safe_float(value) -> float | None:
    """Tolerantly convert a JPL value to ``float``.

    Accept numbers and scientific-notation strings, and treat markers such as
    ``n.a.``, ``nan``, ``--``, or the empty string as missing data.

    Args:
        value: Value received from the JPL response.

    Returns:
        Floating-point number or ``None`` if missing or not convertible.
    """
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
    """Perform the GET request to JPL with controlled retries.

    Retry HTTP 5xx responses and timeouts up to ``retries`` times, waiting
    ``RETRY_DELAY_S`` between attempts. Permanent unsuccessful responses,
    network errors, and non-JSON bodies are converted to ``JPLAPIError``.

    Args:
        params: Query parameters sent to the API.
        timeout: Maximum wait time per request, in seconds.
        retries: Number of additional retries after the first attempt.

    Returns:
        Dictionary decoded from the response JSON.

    Raises:
        JPLAPIError: If the API fails, retries are exhausted, or invalid JSON is returned.
    """
    last_err = None
    for attempt in range(1, retries + 2):  # attempts = retries + 1
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
          offset_margin: float = 1.4142,          # square diagonal (√2)
    max_vmag: float | None = 20.0,          # brightness limit (None = no filter)
    max_objects: int | None = 200,          # returned-results limit
    sort_by: str = "vmag",                  # "vmag" | "offset" | "name"
) -> dict:
    """
    Query JPL and return known minor objects within a field of view.

    Build the query with center, radius, time, and observer; select results from
    the available JPL pass and convert them to dictionaries. Then apply local
    distance and magnitude filters, sort, and limit the final number of objects.

    Args:
        ra_center_deg: FOV center right ascension, in degrees.
        dec_center_deg: FOV center declination, in degrees.
        radius_deg: FOV half-width in degrees for RA and Dec.
        obs_time_iso: Observation time in ISO format; adapted to JPL format.
        observer: Observatorio como ``xobs``, ``mpc_code`` o ``lat``/``lon`` y
            opcionalmente ``alt``.
        two_pass: Request JPL's two-pass mode.
        mag_required: Ask JPL to require visual magnitude.
        sb_kind: Minor-object type requested from JPL.
        timeout: Timeout for each HTTP request, in seconds.
        retries: Retries for transient failures.
        filter_by_offset: Enable the local center-distance filter.
        offset_margin: Radius multiplier to cover the FOV diagonal.
        max_vmag: Maximum visual magnitude; ``None`` disables the filter.
        max_objects: Maximum returned objects; ``None`` imposes no limit.
        sort_by: Sort order: ``vmag``, ``offset``, or ``name``.

    Local filters applied (in order):
        1. Discard objects with |offset| > radius_deg * 3600 * offset_margin
        2. Discard objects with V > max_vmag (if max_vmag is not None)
        3. Sort by sort_by ("vmag" ascending, "offset" ascending, "name")
        4. Truncate to max_objects (if max_objects is not None)

    Returns:
        Dictionary with final objects, fields, JPL summary, observer, pass mode,
        raw counts, and counts after each filter.

    Raises:
        JPLAPIError: If the observer is missing or invalid, the API returns an
            error, or ``sort_by`` is unsupported.
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

    # STEP 1: offset filter
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

    # STEP 2: magnitude filter
    filtered_out_vmag = 0
    if max_vmag is not None:
        kept_vmag = []
        for obj in objects:
            v = _safe_float(obj.get("Visual magnitude (V)"))
            if v is None:
                # No magnitude: keep (it may be detectable)
                kept_vmag.append(obj)
            elif v <= max_vmag:
                kept_vmag.append(obj)
            else:
                filtered_out_vmag += 1
        objects = kept_vmag
    n_after_vmag = len(objects)

    # STEP 3: sort
    if sort_by == "vmag":
        def key_v(o):
            """Return the sort key for visual magnitude."""
            v = _safe_float(o.get("Visual magnitude (V)"))
            return (v is None, v if v is not None else 999.0)
        objects.sort(key=key_v)
    elif sort_by == "offset":
        def key_o(o):
            """Return the sort key for distance from the center."""
            x = _safe_float(o.get('Dist. from center Norm (")'))
            return (x is None, abs(x) if x is not None else 1e18)
        objects.sort(key=key_o)
    elif sort_by == "name":
        objects.sort(key=lambda o: str(o.get("Object name", "")))
    else:
        raise JPLAPIError(f"sort_by must be vmag|offset|name, got {sort_by!r}")

    # STEP 4: truncate
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
