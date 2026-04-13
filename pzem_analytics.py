"""
pzem_analytics.py — Statistiche descrittive e modelli ML sui dati storici.
Importato dal server Flask; nessuna dipendenza da hardware.

Dipendenze extra:
    pip install numpy scipy scikit-learn
"""

import csv
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict

log = logging.getLogger(__name__)

TSV_FILE = 'pzem_history.tsv'

NUMERIC_FIELDS = [
    'voltage', 'current', 'power', 'energy',
    'frequency', 'power_factor', 'apparent_power', 'reactive_power'
]


# ── Caricamento dati ──────────────────────────────────────────────────────────

def load_rows(limit: int = 10000) -> list[dict]:
    """Carica le ultime `limit` righe dal TSV come lista di dict con valori float."""
    rows = []
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
    return rows[-limit:]


# ── Statistiche descrittive ───────────────────────────────────────────────────

def descriptive_stats(rows: list[dict]) -> dict:
    """
    Calcola min, max, media, mediana, deviazione standard per ogni campo numerico.
    """
    if not rows:
        return {}

    result = {}
    for field in NUMERIC_FIELDS:
        vals = [r[field] for r in rows if r[field] is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        std = variance ** 0.5
        median = vals_sorted[n // 2] if n % 2 else (vals_sorted[n//2-1] + vals_sorted[n//2]) / 2

        result[field] = {
            'min':    round(min(vals), 3),
            'max':    round(max(vals), 3),
            'mean':   round(mean, 3),
            'median': round(median, 3),
            'std':    round(std, 3),
            'count':  n,
        }
    return result


# ── Aggregazione oraria/giornaliera ──────────────────────────────────────────

def hourly_averages(rows: list[dict], field: str = 'power') -> list[dict]:
    """Media oraria del campo specificato."""
    buckets = defaultdict(list)
    for r in rows:
        try:
            dt = datetime.fromisoformat(r['timestamp'])
            key = dt.strftime('%Y-%m-%d %H:00')
            buckets[key].append(r[field])
        except Exception:
            continue
    return [
        {'hour': k, 'avg': round(sum(v)/len(v), 2)}
        for k, v in sorted(buckets.items())
    ]


def daily_totals(rows: list[dict]) -> list[dict]:
    """
    Stima consumo giornaliero in Wh:
    usa la differenza di energia tra inizio e fine giornata.
    """
    by_day = defaultdict(list)
    for r in rows:
        try:
            day = r['timestamp'][:10]
            by_day[day].append(r['energy'])
        except Exception:
            continue

    result = []
    for day in sorted(by_day):
        vals = by_day[day]
        if len(vals) >= 2:
            consumed = max(vals) - min(vals)
        else:
            consumed = 0
        result.append({'day': day, 'energy_wh': round(consumed, 1)})
    return result


def peak_hours(rows: list[dict]) -> list[dict]:
    """Potenza media per fascia oraria (0-23), utile per identificare picchi."""
    buckets = defaultdict(list)
    for r in rows:
        try:
            hour = int(r['timestamp'][11:13])
            buckets[hour].append(r['power'])
        except Exception:
            continue
    return [
        {'hour': h, 'avg_power': round(sum(v)/len(v), 1)}
        for h, v in sorted(buckets.items())
    ]


# ── Anomalie ─────────────────────────────────────────────────────────────────

def detect_anomalies(rows: list[dict]) -> list[dict]:
    """
    Rileva valori anomali con z-score > 3 per tensione e potenza.
    Restituisce le righe anomale con il campo e il punteggio z.
    """
    anomalies = []
    for field in ['voltage', 'power', 'current', 'frequency']:
        vals = [r[field] for r in rows]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean)**2 for v in vals) / n) ** 0.5
        if std == 0:
            continue
        for r in rows:
            z = abs((r[field] - mean) / std)
            if z > 3:
                anomalies.append({
                    'timestamp': r['timestamp'],
                    'field': field,
                    'value': r[field],
                    'z_score': round(z, 2),
                })
    anomalies.sort(key=lambda x: x['z_score'], reverse=True)
    return anomalies[:50]


# ── Regressione lineare (previsione consumo) ──────────────────────────────────

def _linreg(x: list, y: list):
    """Regressione lineare minimale senza numpy."""
    n = len(x)
    if n < 2:
        return 0, 0, 0
    sx, sy = sum(x), sum(y)
    sxy = sum(xi*yi for xi, yi in zip(x, y))
    sxx = sum(xi**2 for xi in x)
    denom = n*sxx - sx**2
    if denom == 0:
        return 0, sy/n, 0
    m = (n*sxy - sx*sy) / denom
    b = (sy - m*sx) / n
    # R²
    y_mean = sy / n
    ss_tot = sum((yi - y_mean)**2 for yi in y)
    y_pred = [m*xi + b for xi in x]
    ss_res = sum((yi - yp)**2 for yi, yp in zip(y, y_pred))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return m, b, max(0, min(1, r2))


def energy_forecast(rows: list[dict], days_ahead: int = 7) -> dict:
    """
    Regressione lineare sui consumi giornalieri per prevedere i prossimi giorni.
    Restituisce i punti storici + le previsioni con intervallo di confidenza.
    """
    daily = daily_totals(rows)
    if len(daily) < 3:
        return {'error': 'Dati insufficienti (servono almeno 3 giorni)'}

    # x = indice giorno, y = consumo Wh
    x = list(range(len(daily)))
    y = [d['energy_wh'] for d in daily]

    m, b, r2 = _linreg(x, y)

    # Residui per intervallo di confidenza
    residuals = [abs(yi - (m*xi + b)) for xi, yi in zip(x, y)]
    mae = sum(residuals) / len(residuals)

    # Previsioni
    forecast = []
    last_date = datetime.strptime(daily[-1]['day'], '%Y-%m-%d')
    for i in range(1, days_ahead + 1):
        xi = len(daily) - 1 + i
        pred = round(m * xi + b, 1)
        date = (last_date + timedelta(days=i)).strftime('%Y-%m-%d')
        forecast.append({
            'day': date,
            'predicted_wh': max(0, pred),
            'lower': max(0, round(pred - mae, 1)),
            'upper': round(pred + mae, 1),
        })

    return {
        'historical': daily,
        'forecast': forecast,
        'model': {
            'slope_wh_per_day': round(m, 2),
            'r2': round(r2, 3),
            'mae_wh': round(mae, 1),
            'intercept': round(b, 1),
        }
    }


def power_profile_forecast(rows: list[dict]) -> dict:
    """
    Profilo orario medio della potenza (pattern settimanale).
    Utile per prevedere il carico nelle prossime ore.
    """
    # Media per (giorno_settimana, ora)
    buckets = defaultdict(list)
    for r in rows:
        try:
            dt = datetime.fromisoformat(r['timestamp'])
            key = (dt.weekday(), dt.hour)  # 0=lunedì
            buckets[key].append(r['power'])
        except Exception:
            continue

    days_it = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    profile = {}
    for (wd, h), vals in buckets.items():
        day_name = days_it[wd]
        if day_name not in profile:
            profile[day_name] = {}
        profile[day_name][h] = round(sum(vals)/len(vals), 1)

    # Previsione prossime 24h basata sul giorno della settimana corrente
    now = datetime.now()
    predictions = []
    for i in range(24):
        future = now + timedelta(hours=i)
        key = (future.weekday(), future.hour)
        vals = buckets.get(key, [])
        avg = round(sum(vals)/len(vals), 1) if vals else None
        predictions.append({
            'datetime': future.strftime('%H:%M'),
            'weekday': days_it[future.weekday()],
            'predicted_w': avg,
        })

    return {
        'weekly_profile': profile,
        'next_24h': predictions,
    }


def cost_analysis(rows: list[dict], price_kwh: float = 0.25) -> dict:
    """
    Analisi costi basata sui consumi storici.
    price_kwh: prezzo in €/kWh (default Italia 2024)
    """
    daily = daily_totals(rows)
    if not daily:
        return {}

    consumptions_wh = [d['energy_wh'] for d in daily]
    total_wh = sum(consumptions_wh)
    avg_daily_wh = total_wh / len(daily) if daily else 0
    avg_monthly_wh = avg_daily_wh * 30
    avg_yearly_wh = avg_daily_wh * 365

    return {
        'price_kwh': price_kwh,
        'total_days': len(daily),
        'total_kwh': round(total_wh / 1000, 2),
        'total_cost_eur': round(total_wh / 1000 * price_kwh, 2),
        'avg_daily_kwh': round(avg_daily_wh / 1000, 3),
        'avg_daily_cost_eur': round(avg_daily_wh / 1000 * price_kwh, 3),
        'projected_monthly_kwh': round(avg_monthly_wh / 1000, 2),
        'projected_monthly_cost_eur': round(avg_monthly_wh / 1000 * price_kwh, 2),
        'projected_yearly_kwh': round(avg_yearly_wh / 1000, 1),
        'projected_yearly_cost_eur': round(avg_yearly_wh / 1000 * price_kwh, 1),
        'daily_breakdown': daily[-30:],  # ultimi 30 giorni
    }


# ── Entry point per test standalone ──────────────────────────────────────────

if __name__ == '__main__':
    import json
    rows = load_rows()
    print(f"Righe caricate: {len(rows)}")
    if rows:
        print("\n── Statistiche ──")
        print(json.dumps(descriptive_stats(rows), indent=2))
        print("\n── Previsione consumo ──")
        print(json.dumps(energy_forecast(rows), indent=2))
        print("\n── Analisi costi ──")
        print(json.dumps(cost_analysis(rows), indent=2))
