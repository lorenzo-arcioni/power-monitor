"""
pzem_analytics.py — Statistiche e modelli sui dati storici PZEM-004t.

Strategia prestazioni (Centrino Duo, TSV > 10 MB):
  - tail_read_tsv(): legge solo le ultime N righe dal fondo del file
    tramite seek binario — O(tail_kb) invece di O(file_size).
  - _daily_cache: aggregato giornaliero persistente su JSON, ricalcolato
    solo quando arrivano giorni nuovi. Forecast e cost non toccano mai
    il TSV intero: leggono solo il JSON (pochi KB).
  - load_rows_cached(): usata solo da stats/anomalies/peaks, con limite
    ridotto a STATS_ROWS (≈ 1-2 ore di dati) per mantenere il calcolo
    entro ~100 ms anche su hardware lento.
  - Cache con TTL lungo (300 s) per analytics pesanti.
"""

import csv
import io
import json
import os
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict

log = logging.getLogger(__name__)

TSV_FILE        = 'pzem_history.tsv'
DAILY_JSON      = 'pzem_daily.json'   # aggregato giornaliero persistente

# Quante righe leggere per statistiche / anomalie / profilo orario.
# A 2 s/campione: 3600 righe ≈ 2 ore, 7200 ≈ 4 ore.
STATS_ROWS      = 3600

# Quante righe leggere per /api/history (grafico storico nella dashboard).
# A 2 s/campione: 2000 righe ≈ 1 ora, già più che sufficienti per un grafico.
HISTORY_ROWS    = 2000

# TTL cache analytics in RAM (secondi)
ANALYTICS_TTL   = 300.0   # 5 minuti — Centrino Duo non deve rileggere spesso

NUMERIC_FIELDS = [
    'voltage', 'current', 'power', 'energy',
    'frequency', 'power_factor', 'apparent_power', 'reactive_power'
]
ANOMALY_FIELDS = ['voltage', 'power', 'current', 'frequency']

# ── Cache in-memory (condivisa con server tramite riferimento) ────────────────
_cache: dict = {'rows': None, 'ts': 0.0}


def _cache_valid() -> bool:
    return _cache['rows'] is not None and (time.monotonic() - _cache['ts']) < ANALYTICS_TTL


# ═══════════════════════════════════════════════════════════════════════════════
#  LETTURA TSV EFFICIENTE — tail da fondo file, senza leggere tutto
# ═══════════════════════════════════════════════════════════════════════════════

def tail_read_tsv(n_rows: int = STATS_ROWS) -> list[dict]:
    """
    Legge le ultime n_rows righe dal TSV senza caricare l'intero file.
    Strategia:
      1. Seek alla fine del file.
      2. Leggi a ritroso blocchi da CHUNK bytes finché non si trovano
         abbastanza newline (n_rows + 1 per l'header).
      3. Parsa solo quel frammento con csv.DictReader.

    Su un file da 14 MB con n_rows=3600 si leggono ~300-400 KB invece
    di 14 MB: ×35 più veloce su disco lento.
    """
    if not os.path.exists(TSV_FILE):
        return []

    CHUNK = 65536  # 64 KB per blocco
    needed_nl = n_rows + 2   # +1 header +1 margine

    with open(TSV_FILE, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()

        if file_size == 0:
            return []

        buf = b''
        pos = file_size
        found = 0

        while pos > 0 and found < needed_nl:
            read_size = min(CHUNK, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            buf   = chunk + buf
            found = buf.count(b'\n')

        # Se il file è più piccolo di CHUNK, rileggi dall'inizio
        if pos == 0 and found < needed_nl:
            f.seek(0)
            buf = f.read()

    # Estrai le ultime needed_nl righe (più l'header che è la prima riga)
    lines = buf.split(b'\n')
    # Rimuovi righe vuote in coda
    while lines and not lines[-1].strip():
        lines.pop()

    # Conserva header (prima riga del file) + ultime n_rows righe di dati
    # L'header lo recuperiamo dal file intero (solo la prima riga)
    header_line = _read_header()
    if not header_line:
        return []

    # Prendi le ultime n_rows righe di dati
    data_lines = lines[-(n_rows):]
    fragment   = header_line + b'\n' + b'\n'.join(data_lines)

    rows: list[dict] = []
    try:
        text = fragment.decode('utf-8', errors='replace').replace('\r', '')
        reader = csv.DictReader(io.StringIO(text), delimiter='\t')
        for row in reader:
            try:
                parsed = {'timestamp': row['timestamp']}
                for k in NUMERIC_FIELDS:
                    parsed[k] = float(row.get(k, 0) or 0)
                rows.append(parsed)
            except (ValueError, KeyError):
                continue
    except Exception as e:
        log.warning(f"tail_read_tsv parse error: {e}")

    return rows


def _read_header() -> bytes:
    """Legge solo la prima riga del TSV (header). Velocissimo."""
    try:
        with open(TSV_FILE, 'rb') as f:
            return f.readline().rstrip(b'\n\r')
    except Exception:
        return b''


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHE IN-MEMORY PER ANALYTICS VELOCI
# ═══════════════════════════════════════════════════════════════════════════════

def load_rows_cached(limit: int = STATS_ROWS) -> list[dict]:
    """
    Restituisce le ultime `limit` righe usando tail_read_tsv() + cache TTL.
    Rispetto alla versione precedente NON carica mai l'intero file.
    """
    if _cache_valid():
        rows = _cache['rows']
        return rows[-limit:] if len(rows) > limit else rows

    rows = tail_read_tsv(STATS_ROWS)
    _cache['rows'] = rows
    _cache['ts']   = time.monotonic()
    return rows[-limit:] if len(rows) > limit else rows


def load_rows(limit: int = STATS_ROWS) -> list[dict]:
    """Alias per compatibilità."""
    return load_rows_cached(limit)


def invalidate_cache():
    """Chiamata dal server dopo ogni flush TSV."""
    _cache['ts'] = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATO GIORNALIERO PERSISTENTE
#  Forecast e cost_analysis leggono questo JSON (pochi KB) invece del TSV.
# ═══════════════════════════════════════════════════════════════════════════════

def _load_daily_json() -> list[dict]:
    """Carica l'aggregato giornaliero dal JSON persistente."""
    if not os.path.exists(DAILY_JSON):
        return []
    try:
        with open(DAILY_JSON) as f:
            return json.load(f)
    except Exception:
        return []


def _save_daily_json(data: list[dict]):
    try:
        with open(DAILY_JSON, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        log.warning(f"Errore salvataggio daily JSON: {e}")


def update_daily_aggregate():
    """
    Aggiorna pzem_daily.json scansionando il TSV in modo leggero.

    Problema del tail: energy è cumulativa. Se tagli le righe a metà
    giornata, max-min del giorno parziale è sbagliato (potresti avere
    solo la seconda metà del giorno dove energy è già alta).
    Soluzione: scansione lineare dell'intero TSV leggendo SOLO le colonne
    timestamp e energy (le altre vengono ignorate). Su un file da 14 MB
    con ~250k righe e 2 colonne invece di 10, il parsing è ~5x più veloce
    e la RAM usata è minima (due float per riga, non dieci).

    Strategia incrementale:
      - Carica il JSON esistente e trova l'ultimo giorno già chiuso.
      - Un giorno è "chiuso" se NON è oggi (il giorno corrente è ancora
        in corso, quindi max-min sarebbe parziale).
      - Scansiona il TSV saltando le righe dei giorni già aggregati
        (confronto stringa sul prefisso data, O(1) per riga).
      - Ricalcola solo i giorni nuovi + ri-aggrega oggi (giorno aperto).
    """
    existing  = _load_daily_json()
    today_str = datetime.now().strftime('%Y-%m-%d')

    # Giorni già chiusi e definitivi (tutto tranne oggi)
    closed = {e['day']: e for e in existing if e['day'] < today_str}
    last_closed_day = max(closed.keys()) if closed else '1970-01-01'

    # Scansione TSV — legge solo timestamp e energy
    # by_day[giorno] = (min_energy, max_energy)
    by_day_min: dict[str, float] = {}
    by_day_max: dict[str, float] = {}

    if not os.path.exists(TSV_FILE):
        return

    ts_idx  = None
    en_idx  = None

    with open(TSV_FILE, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader((line.replace('\r', '') for line in f), delimiter='\t')
        try:
            header = next(reader)
        except StopIteration:
            return
        try:
            ts_idx = header.index('timestamp')
            en_idx = header.index('energy')
        except ValueError:
            log.error("update_daily_aggregate: colonne timestamp/energy non trovate nell'header")
            return

        for row in reader:
            try:
                day = row[ts_idx][:10]
                # Salta i giorni già chiusi e definitivi per velocità
                if day < last_closed_day:
                    continue
                val = float(row[en_idx])
                if day not in by_day_min:
                    by_day_min[day] = val
                    by_day_max[day] = val
                else:
                    if val < by_day_min[day]: by_day_min[day] = val
                    if val > by_day_max[day]: by_day_max[day] = val
            except (IndexError, ValueError):
                continue

    if not by_day_min:
        return

    # Calcola consumo per ogni giorno scansionato
    new_or_updated: dict = {}
    for day in sorted(by_day_min.keys()):
        consumed = round(by_day_max[day] - by_day_min[day], 1)
        if consumed < 0:
            # Reset del contatore PZEM durante la giornata: usa solo il max
            consumed = round(by_day_max[day], 1)
        if consumed >= 0:
            new_or_updated[day] = {'day': day, 'energy_wh': consumed}

    # Unisci: giorni chiusi dal JSON + giorni nuovi/aggiornati dalla scansione
    # I giorni chiusi già nel JSON rimangono invariati (non vengono riScansionati)
    merged_dict = {**closed, **new_or_updated}
    merged = sorted(merged_dict.values(), key=lambda x: x['day'])

    _save_daily_json(merged)
    log.info(
        f"Daily aggregate aggiornato: {len(new_or_updated)} giorni scansionati "
        f"→ totale {len(merged)} giorni"
    )


def daily_totals(rows: list[dict] = None) -> list[dict]:
    """
    Restituisce l'aggregato giornaliero.
    Se il JSON esiste lo usa direttamente (velocissimo).
    Altrimenti lo calcola dalle righe passate (fallback).
    """
    data = _load_daily_json()
    if data:
        return data

    # Fallback: calcolo dalle righe in memoria
    if not rows:
        rows = load_rows_cached()
    by_day_min: dict = {}
    by_day_max: dict = {}
    for r in rows:
        try:
            day = r['timestamp'][:10]
            val = r['energy']
            if day not in by_day_min:
                by_day_min[day] = val; by_day_max[day] = val
            else:
                if val < by_day_min[day]: by_day_min[day] = val
                if val > by_day_max[day]: by_day_max[day] = val
        except Exception:
            continue
    result = []
    for day in sorted(by_day_min.keys()):
        consumed = by_day_max[day] - by_day_min[day]
        if consumed < 0: consumed = by_day_max[day]
        result.append({'day': day, 'energy_wh': round(consumed, 1)})
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICHE DESCRITTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(rows: list[dict]) -> dict:
    """Algoritmo di Welford — singola passata O(n) per campo."""
    if not rows:
        return {}

    result = {}
    for field in NUMERIC_FIELDS:
        vals = [r[field] for r in rows if r.get(field) is not None]
        if not vals:
            continue
        n = len(vals)

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


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGAZIONI ORARIE / PROFILO DI CARICO
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  ANOMALIE — singola passata su campione ridotto
# ═══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

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


# ═══════════════════════════════════════════════════════════════════════════════
#  FORECAST — usa daily_totals() che legge il JSON (pochi KB)
# ═══════════════════════════════════════════════════════════════════════════════

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


def energy_forecast(rows: list[dict] = None, days_ahead: int = 7) -> dict:
    """
    Usa l'aggregato giornaliero (JSON) — non tocca il TSV grosso.
    rows è ignorato ma mantenuto per compatibilità firma.
    """
    daily = daily_totals()   # legge da JSON, velocissimo

    # Filtra giorni con consumo zero (possibile se PZEM era offline)
    daily = [d for d in daily if d['energy_wh'] > 0]

    if len(daily) < 3:
        return {'error': f'Dati insufficienti ({len(daily)} giorni validi, servono almeno 3)'}

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
            'day':          date,
            'predicted_wh': max(0.0, pred),
            'lower':        max(0.0, round(pred - mae, 1)),
            'upper':        round(pred + mae, 1),
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


# ═══════════════════════════════════════════════════════════════════════════════
#  COSTI — usa daily_totals() → JSON
# ═══════════════════════════════════════════════════════════════════════════════

def cost_analysis(rows: list[dict] = None, price_kwh: float = 0.25) -> dict:
    """
    rows è ignorato: legge dal JSON aggregato giornaliero.
    """
    daily = daily_totals()
    if not daily:
        return {}

    consumptions = [d['energy_wh'] for d in daily]
    total_wh     = sum(consumptions)
    avg_day_wh   = total_wh / len(daily) if daily else 0

    return {
        'price_kwh':                  price_kwh,
        'total_days':                 len(daily),
        'total_kwh':                  round(total_wh / 1000, 2),
        'total_cost_eur':             round(total_wh / 1000 * price_kwh, 2),
        'avg_daily_kwh':              round(avg_day_wh / 1000, 3),
        'avg_daily_cost_eur':         round(avg_day_wh / 1000 * price_kwh, 3),
        'projected_monthly_kwh':      round(avg_day_wh * 30  / 1000, 2),
        'projected_monthly_cost_eur': round(avg_day_wh * 30  / 1000 * price_kwh, 2),
        'projected_yearly_kwh':       round(avg_day_wh * 365 / 1000, 1),
        'projected_yearly_cost_eur':  round(avg_day_wh * 365 / 1000 * price_kwh, 1),
        'daily_breakdown':            daily[-30:],
    }


# ── Entry point per test standalone ──────────────────────────────────────────

if __name__ == '__main__':
    import json as _json
    import sys

    logging.basicConfig(level=logging.INFO)

    print("── Aggiornamento aggregato giornaliero ──")
    update_daily_aggregate()

    daily = _load_daily_json()
    print(f"Giorni aggregati: {len(daily)}")
    if daily:
        print(f"  Dal {daily[0]['day']} al {daily[-1]['day']}")

    print("\n── Caricamento righe recenti (tail) ──")
    rows = load_rows_cached()
    print(f"Righe in cache: {len(rows)}")

    if rows:
        print("\n── Statistiche (ultime righe) ──")
        print(_json.dumps(descriptive_stats(rows), indent=2))

    print("\n── Previsione consumo ──")
    fc = energy_forecast()
    if 'error' in fc:
        print(f"  Errore: {fc['error']}")
    else:
        print(_json.dumps(fc['model'], indent=2))
        print(f"  Giorni previsti: {len(fc['forecast'])}")

    print("\n── Analisi costi ──")
    print(_json.dumps(cost_analysis(), indent=2))