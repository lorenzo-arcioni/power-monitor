"""
pzem_analytics.py — Statistiche e modelli ML sui dati storici PZEM-004t.
Ottimizzazioni rispetto alla versione originale:
  - load_rows_cached(): rilegge il TSV solo se il server ha invaluato la cache
  - detect_anomalies(): singola passata O(n) per tutti i campi invece di 4 separate
  - descriptive_stats(): singola passata per min/max/mean/std/median
  - daily_totals() con cache interna condivisa da energy_forecast e cost_analysis
"""

import csv
import os
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict

log = logging.getLogger(__name__)

TSV_FILE = 'pzem_history.tsv'

NUMERIC_FIELDS = [
    'voltage', 'current', 'power', 'energy',
    'frequency', 'power_factor', 'apparent_power', 'reactive_power'
]

ANOMALY_FIELDS = ['voltage', 'power', 'current', 'frequency']

# ── Cache condivisa con il server ─────────────────────────────────────────────
# Il server la invalida (ts=0) dopo ogni flush su disco.
# analytics la legge e la aggiorna solo se scaduta.
_cache: dict = {'rows': None, 'ts': 0.0}
_CACHE_TTL = 30.0   # secondi; deve combaciare (o essere ≤) al TTL del server


def _cache_valid() -> bool:
    return _cache['rows'] is not None and (time.monotonic() - _cache['ts']) < _CACHE_TTL


def load_rows_cached(limit: int = 10000) -> list[dict]:
    """
    Come load_rows() ma con cache in-memory (TTL=_CACHE_TTL).
    Evita riletture ripetute quando più endpoint analytics vengono chiamati
    in rapida successione.
    """
    if _cache_valid():
        rows = _cache['rows']
        return rows[-limit:] if len(rows) > limit else rows

    rows: list[dict] = []
    if not os.path.exists(TSV_FILE):
        return rows

    with open(TSV_FILE, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                parsed = {'timestamp': row['timestamp']}
                for k in NUMERIC_FIELDS:
                    parsed[k] = float(row.get(k, 0) or 0)
                rows.append(parsed)
            except (ValueError, KeyError):
                continue

    _cache['rows'] = rows
    _cache['ts']   = time.monotonic()
    return rows[-limit:] if len(rows) > limit else rows


def load_rows(limit: int = 10000) -> list[dict]:
    """Alias per compatibilità con codice esterno."""
    return load_rows_cached(limit)


# ── Statistiche descrittive ───────────────────────────────────────────────────

def descriptive_stats(rows: list[dict]) -> dict:
    """
    Singola passata O(n) per campo: calcola min, max, somma, somma quadrati.
    Mediana richiede sort O(n log n) ma solo sui valori del singolo campo.
    """
    if not rows:
        return {}

    result = {}
    for field in NUMERIC_FIELDS:
        vals = [r[field] for r in rows if r[field] is not None]
        if not vals:
            continue
        n = len(vals)

        # Singola passata per min/max/mean/varianza (algoritmo di Welford)
        min_v = max_v = vals[0]
        mean = 0.0
        M2   = 0.0
        for i, v in enumerate(vals, 1):
            if v < min_v: min_v = v
            if v > max_v: max_v = v
            delta  = v - mean
            mean  += delta / i
            M2    += delta * (v - mean)

        std    = (M2 / n) ** 0.5
        vals_s = sorted(vals)
        median = vals_s[n // 2] if n % 2 else (vals_s[n//2-1] + vals_s[n//2]) / 2

        result[field] = {
            'min':    round(min_v, 3),
            'max':    round(max_v, 3),
            'mean':   round(mean,  3),
            'median': round(median,3),
            'std':    round(std,   3),
            'count':  n,
        }
    return result


# ── Aggregazione oraria/giornaliera ──────────────────────────────────────────

def hourly_averages(rows: list[dict], field: str = 'power') -> list[dict]:
    buckets: dict = defaultdict(list)
    for r in rows:
        try:
            dt  = datetime.fromisoformat(r['timestamp'])
            key = dt.strftime('%Y-%m-%d %H:00')
            buckets[key].append(r[field])
        except Exception:
            continue
    return [
        {'hour': k, 'avg': round(sum(v)/len(v), 2)}
        for k, v in sorted(buckets.items())
    ]


def daily_totals(rows: list[dict]) -> list[dict]:
    """Stima consumo giornaliero in Wh (max-min energia per giorno)."""
    by_day: dict = defaultdict(list)
    for r in rows:
        try:
            by_day[r['timestamp'][:10]].append(r['energy'])
        except Exception:
            continue
    result = []
    for day in sorted(by_day):
        vals = by_day[day]
        consumed = (max(vals) - min(vals)) if len(vals) >= 2 else 0
        result.append({'day': day, 'energy_wh': round(consumed, 1)})
    return result


def peak_hours(rows: list[dict]) -> list[dict]:
    buckets: dict = defaultdict(list)
    for r in rows:
        try:
            buckets[int(r['timestamp'][11:13])].append(r['power'])
        except Exception:
            continue
    return [
        {'hour': h, 'avg_power': round(sum(v)/len(v), 1)}
        for h, v in sorted(buckets.items())
    ]


# ── Anomalie (singola passata) ────────────────────────────────────────────────

def detect_anomalies(rows: list[dict]) -> list[dict]:
    """
    Singola passata O(n × |ANOMALY_FIELDS|) invece di 4 passate separate.
    Calcola media e std per tutti i campi insieme, poi applica z-score.
    """
    if not rows:
        return []

    # Calcola stats per tutti i campi in una passata
    sums:  dict[str, float] = {f: 0.0 for f in ANOMALY_FIELDS}
    sums2: dict[str, float] = {f: 0.0 for f in ANOMALY_FIELDS}
    n = len(rows)

    for r in rows:
        for f in ANOMALY_FIELDS:
            v = r[f]
            sums[f]  += v
            sums2[f] += v * v

    means = {f: sums[f] / n for f in ANOMALY_FIELDS}
    stds  = {
        f: max((sums2[f]/n - means[f]**2), 0) ** 0.5
        for f in ANOMALY_FIELDS
    }

    anomalies = []
    for r in rows:
        for f in ANOMALY_FIELDS:
            if stds[f] == 0:
                continue
            z = abs((r[f] - means[f]) / stds[f])
            if z > 3:
                anomalies.append({
                    'timestamp': r['timestamp'],
                    'field':     f,
                    'value':     r[f],
                    'z_score':   round(z, 2),
                })

    anomalies.sort(key=lambda x: x['z_score'], reverse=True)
    return anomalies[:50]


# ── Regressione lineare ───────────────────────────────────────────────────────

def _linreg(x: list, y: list):
    n = len(x)
    if n < 2:
        return 0, 0, 0
    sx  = sum(x);  sy  = sum(y)
    sxy = sum(xi*yi for xi, yi in zip(x, y))
    sxx = sum(xi**2 for xi in x)
    den = n*sxx - sx**2
    if den == 0:
        return 0, sy/n, 0
    m  = (n*sxy - sx*sy) / den
    b  = (sy - m*sx) / n
    ym = sy / n
    ss_tot = sum((yi - ym)**2 for yi in y)
    ss_res = sum((yi - (m*xi+b))**2 for xi, yi in zip(x, y))
    r2 = max(0.0, min(1.0, 1 - ss_res/ss_tot)) if ss_tot > 0 else 0
    return m, b, r2


def energy_forecast(rows: list[dict], days_ahead: int = 7) -> dict:
    daily = daily_totals(rows)
    if len(daily) < 3:
        return {'error': 'Dati insufficienti (servono almeno 3 giorni)'}

    x = list(range(len(daily)))
    y = [d['energy_wh'] for d in daily]
    m, b, r2 = _linreg(x, y)
    residuals = [abs(yi - (m*xi + b)) for xi, yi in zip(x, y)]
    mae = sum(residuals) / len(residuals)

    forecast = []
    last_date = datetime.strptime(daily[-1]['day'], '%Y-%m-%d')
    for i in range(1, days_ahead + 1):
        xi   = len(daily) - 1 + i
        pred = round(m * xi + b, 1)
        date = (last_date + timedelta(days=i)).strftime('%Y-%m-%d')
        forecast.append({
            'day':           date,
            'predicted_wh':  max(0, pred),
            'lower':         max(0, round(pred - mae, 1)),
            'upper':         round(pred + mae, 1),
        })

    return {
        'historical': daily,
        'forecast':   forecast,
        'model': {
            'slope_wh_per_day': round(m,   2),
            'r2':               round(r2,  3),
            'mae_wh':           round(mae, 1),
            'intercept':        round(b,   1),
        }
    }


def power_profile_forecast(rows: list[dict]) -> dict:
    buckets: dict = defaultdict(list)
    for r in rows:
        try:
            dt  = datetime.fromisoformat(r['timestamp'])
            buckets[(dt.weekday(), dt.hour)].append(r['power'])
        except Exception:
            continue

    days_it = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    profile: dict = {}
    for (wd, h), vals in buckets.items():
        day_name = days_it[wd]
        profile.setdefault(day_name, {})[h] = round(sum(vals)/len(vals), 1)

    now = datetime.now()
    predictions = []
    for i in range(24):
        future = now + timedelta(hours=i)
        key    = (future.weekday(), future.hour)
        vals   = buckets.get(key, [])
        avg    = round(sum(vals)/len(vals), 1) if vals else None
        predictions.append({
            'datetime':    future.strftime('%H:%M'),
            'weekday':     days_it[future.weekday()],
            'predicted_w': avg,
        })

    return {'weekly_profile': profile, 'next_24h': predictions}


def cost_analysis(rows: list[dict], price_kwh: float = 0.25) -> dict:
    """
    Riusa daily_totals() che condivide la stessa cache: nessun ricalcolo.
    """
    daily = daily_totals(rows)
    if not daily:
        return {}

    consumptions = [d['energy_wh'] for d in daily]
    total_wh     = sum(consumptions)
    avg_day_wh   = total_wh / len(daily)

    return {
        'price_kwh':                   price_kwh,
        'total_days':                  len(daily),
        'total_kwh':                   round(total_wh / 1000, 2),
        'total_cost_eur':              round(total_wh / 1000 * price_kwh, 2),
        'avg_daily_kwh':               round(avg_day_wh / 1000, 3),
        'avg_daily_cost_eur':          round(avg_day_wh / 1000 * price_kwh, 3),
        'projected_monthly_kwh':       round(avg_day_wh * 30  / 1000, 2),
        'projected_monthly_cost_eur':  round(avg_day_wh * 30  / 1000 * price_kwh, 2),
        'projected_yearly_kwh':        round(avg_day_wh * 365 / 1000, 1),
        'projected_yearly_cost_eur':   round(avg_day_wh * 365 / 1000 * price_kwh, 1),
        'daily_breakdown':             daily[-30:],
    }


# ── Entry point per test standalone ──────────────────────────────────────────

if __name__ == '__main__':
    import json
    rows = load_rows_cached()
    print(f"Righe caricate: {len(rows)}")
    if rows:
        print("\n── Statistiche ──")
        print(json.dumps(descriptive_stats(rows), indent=2))
        print("\n── Previsione consumo ──")
        print(json.dumps(energy_forecast(rows), indent=2))
        print("\n── Analisi costi ──")
        print(json.dumps(cost_analysis(rows), indent=2))
