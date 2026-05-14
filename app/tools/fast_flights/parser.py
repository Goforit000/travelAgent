from __future__ import annotations

import json

from selectolax.lexbor import LexborHTMLParser

from .model import (
    Airline,
    Airport,
    Alliance,
    CarbonEmission,
    Flights,
    JsMetadata,
    SimpleDatetime,
    SingleFlight,
)


class MetaList(list[Flights]):
    metadata: JsMetadata


def parse(html: str) -> MetaList:
    parser = LexborHTMLParser(html)
    script = parser.css_first(r"script.ds\:1")
    if script is None:
        all_scripts = parser.css("script")
        script_info = []
        for i, s in enumerate(all_scripts[:5]):
            classes = s.attrs.get("class", "") if s.attrs else ""
            text_preview = (s.text() or "")[:150] if s.text() else "(empty)"
            script_info.append(f"#{i} class={classes!r} text={text_preview}")
        raise RuntimeError(
            "Google Flights page structure changed or request blocked "
            f"(no script.ds:1 found, got {len(all_scripts)} scripts: {'; '.join(script_info)})"
        )
    return parse_js(script.text())


def _safe_get(arr, idx, default=None):
    if arr is None:
        return default
    try:
        return arr[idx]
    except (IndexError, TypeError):
        return default


def _safe_time_pair(raw, idx):
    try:
        val = raw[idx]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return (int(val[0]), int(val[1]))
        if isinstance(val, (list, tuple)) and len(val) == 1:
            return (int(val[0]), 0)
        if isinstance(val, (int, float)):
            h, m = divmod(int(val), 60)
            return (h, m)
    except (ValueError, TypeError, IndexError):
        pass
    return (0, 0)


def parse_js(js: str):
    data = js.split("data:", 1)[1].rsplit(",", 1)[0]
    payload = json.loads(data)

    # ── airlines/alliances metadata ──
    alliances = []
    airlines = []
    try:
        alliances_data_raw = _safe_get(payload, 7, [])
        alliances_data = _safe_get(alliances_data_raw, 1, [[], []])
        alliances_raw = _safe_get(alliances_data, 0, [])
        airlines_raw = _safe_get(alliances_data, 1, [])
        for item in (alliances_raw or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                alliances.append(Alliance(code=str(item[0]), name=str(item[1])))
        for item in (airlines_raw or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                airlines.append(Airline(code=str(item[0]), name=str(item[1])))
    except Exception:
        pass

    meta = JsMetadata(alliances=alliances, airlines=airlines)

    # ── flights ──
    flights = MetaList()
    payload_3 = _safe_get(payload, 3)
    if payload_3 is None:
        return flights

    flight_groups = _safe_get(payload_3, 0)
    if flight_groups is None:
        return flights

    for k_idx, k in enumerate(flight_groups):
        try:
            flight = k[0]

            price = _safe_get(_safe_get(k, 1, []), 0, [0, 0])
            price = _safe_get(price, 1, 0) if isinstance(price, (list, tuple)) else 0

            typ = _safe_get(flight, 0, "unknown")
            airline_codes = _safe_get(flight, 1, [])

            sg_flights = []
            segments = _safe_get(flight, 2, [])
            for single_flight in segments:
                code_from = str(_safe_get(single_flight, 3, "???"))
                name_from = str(_safe_get(single_flight, 4, "Unknown"))
                name_to = str(_safe_get(single_flight, 5, "Unknown"))
                code_to = str(_safe_get(single_flight, 6, "???"))
                from_airport = Airport(code=code_from, name=name_from)
                to_airport = Airport(code=code_to, name=name_to)

                departure_time = _safe_time_pair(single_flight, 8)
                departure_date_raw = _safe_get(single_flight, 20, (0, 0, 0))
                if not isinstance(departure_date_raw, (list, tuple)) or len(departure_date_raw) < 3:
                    departure_date_raw = (0, 0, 0)
                departure = SimpleDatetime(date=departure_date_raw, time=departure_time)

                arrival_time = _safe_time_pair(single_flight, 10)
                arrival_date_raw = _safe_get(single_flight, 21, (0, 0, 0))
                if not isinstance(arrival_date_raw, (list, tuple)) or len(arrival_date_raw) < 3:
                    arrival_date_raw = (0, 0, 0)
                arrival = SimpleDatetime(date=arrival_date_raw, time=arrival_time)

                plane_type = str(_safe_get(single_flight, 17, ""))
                duration = int(_safe_get(single_flight, 11, 0) or 0)

                sg_flights.append(
                    SingleFlight(
                        from_airport=from_airport,
                        to_airport=to_airport,
                        departure=departure,
                        arrival=arrival,
                        duration=duration,
                        plane_type=plane_type,
                    )
                )

            extras = _safe_get(flight, 22, [])
            carbon_emission = int(_safe_get(extras, 7, 0) or 0)
            typical_carbon_emission = int(_safe_get(extras, 8, 0) or 0)

            flights.append(
                Flights(
                    type=typ,
                    price=int(price or 0),
                    airlines=airline_codes,
                    flights=sg_flights,
                    carbon=CarbonEmission(
                        typical_on_route=typical_carbon_emission,
                        emission=carbon_emission,
                    ),
                )
            )
        except Exception:
            continue

    flights.metadata = meta
    return flights
