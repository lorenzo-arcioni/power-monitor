"""
pzem_server.py — Server Flask per la dashboard PZEM-004t.

Ottimizzazioni prestazioni (Centrino Duo, TSV > 10 MB):
  - /api/history usa tail_read_tsv(): legge solo le ultime N righe
    dal fondo del file senza caricare tutto in RAM.
  - Tutti gli endpoint analytics usano load_rows_cached() che opera
    su un campione ridotto (STATS_ROWS ≈ 3600 righe, ~2 ore).
  - Forecast e cost_analysis leggono pzem_daily.json (pochi KB)
    aggiornato ogni ora da un thread dedicato.
  - CACHE_TTL alzato a 300 s: il TSV viene riletto al massimo ogni 5 min.
  - invalidate_cache() chiamata solo dopo flush TSV, non ad ogni scrittura.
"""

import os
import csv
import time
import threading
import logging
from collections import deque
from flask import Flask, jsonify, send_file, abort, request
from flask_cors import CORS

import pzem_reader as reader
import pzem_analytics as analytics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ── Configurazione ────────────────────────────────────────────────────────────
POLL_INTERVAL     = 2      # secondi tra letture seriali
MAX_LIVE_ROWS     = 500    # campioni in RAM per /api/live
TSV_FILE          = 'pzem_history.tsv'
WRITE_BATCH       = 30     # righe accumulate prima del flush su disco
                           # (era 10: con 2s/campione = flush ogni 60s invece di 20s)
DAILY_UPDATE_SECS = 3600   # aggiorna aggregato giornaliero ogni ora

TSV_HEADERS = [
    'timestamp', 'voltage', 'current', 'power', 'energy',
    'frequency', 'power_factor', 'alarm', 'apparent_power', 'reactive_power'
]

# ── Stato globale thread-safe ─────────────────────────────────────────────────
latest        = {}
history: deque = deque(maxlen=MAX_LIVE_ROWS)
sensor_online = False
lock          = threading.Lock()

# Write-buffer per il TSV
_write_buf: list[dict] = []
_write_buf_lock = threading.Lock()

# ── TSV helpers ───────────────────────────────────────────────────────────────

def init_tsv():
    if not os.path.exists(TSV_FILE):
        with open(TSV_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=TSV_HEADERS,
                           delimiter='\t').writeheader()
        log.info(f"Creato file storico: {TSV_FILE}")


def _flush_buf(buf: list[dict]):
    """Scrive il buffer su disco e invalida la cache analytics."""
    with open(TSV_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TSV_HEADERS,
                           delimiter='\t', extrasaction='ignore')
        w.writerows(buf)
    buf.clear()
    # Invalida la cache in analytics così al prossimo accesso rilegge
    analytics.invalidate_cache()


def queue_tsv(row: dict):
    """Accoda una riga; flush ogni WRITE_BATCH righe."""
    with _write_buf_lock:
        _write_buf.append(row)
        if len(_write_buf) >= WRITE_BATCH:
            _flush_buf(_write_buf)


# ── Thread di polling sensore ─────────────────────────────────────────────────

def poll_loop():
    global sensor_online
    log.info(f"Polling su {reader.SERIAL_PORT} ogni {POLL_INTERVAL}s")
    while True:
        row = reader.read_sensor()
        with lock:
            if row:
                sensor_online = True
                latest.clear()
                latest.update(row)
                history.append(dict(row))
                queue_tsv(row)
            else:
                sensor_online = False
        time.sleep(POLL_INTERVAL)


# ── Thread aggiornamento aggregato giornaliero ────────────────────────────────

def daily_update_loop():
    """
    Aggiorna pzem_daily.json ogni DAILY_UPDATE_SECS secondi.
    Gira in background senza toccare Flask.
    Il primo aggiornamento avviene subito all'avvio (dopo 5s di attesa
    per dare tempo al server di partire).
    """
    time.sleep(5)
    while True:
        try:
            log.info("Aggiornamento aggregato giornaliero…")
            analytics.update_daily_aggregate()
        except Exception as e:
            log.error(f"Errore daily_update_loop: {e}")
        time.sleep(DAILY_UPDATE_SECS)


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)


@app.route('/')
def index():
    html_path = os.path.join(os.path.dirname(__file__), 'pzem_dashboard.html')
    if os.path.exists(html_path):
        return send_file(html_path)
    abort(404, "pzem_dashboard.html non trovata.")


# ── API dati live ─────────────────────────────────────────────────────────────

@app.route('/api/latest')
def api_latest():
    with lock:
        return jsonify({
            'online': sensor_online,
            'data':   dict(latest) if sensor_online else None,
        })


@app.route('/api/live')
def api_live():
    """
    Campioni in RAM. Con ?since=<ISO timestamp> restituisce solo il delta.
    """
    since = request.args.get('since')
    with lock:
        snap = list(history)
    if since:
        snap = [r for r in snap if r.get('timestamp', '') > since]
    return jsonify(snap)


@app.route('/api/status')
def api_status():
    with lock:
        ok  = sensor_online
        buf = len(history)
    return jsonify({
        'sensor_online': ok,
        'serial_port':   reader.SERIAL_PORT,
        'poll_every':    POLL_INTERVAL,
        'tsv_file':      TSV_FILE,
        'buffer_size':   buf,
    })


# ── API storico ───────────────────────────────────────────────────────────────

@app.route('/api/history')
def api_history():
    """
    Usa tail_read_tsv() — legge solo le ultime N righe dal fondo del file.
    Default 2000 righe (≈ 1 ora a 2s/campione), max 5000.
    Il client della dashboard usa già sottocampionamento lato JS,
    quindi 2000 punti sono più che sufficienti per il grafico storico.
    """
    limit = min(int(request.args.get('limit', analytics.HISTORY_ROWS)), 5000)
    rows  = analytics.tail_read_tsv(limit)
    return jsonify(rows)


@app.route('/api/download')
def api_download():
    """Scarica il TSV completo. Flush preventivo per dati aggiornati."""
    with _write_buf_lock:
        if _write_buf:
            _flush_buf(_write_buf)
    if os.path.exists(TSV_FILE):
        return send_file(TSV_FILE, as_attachment=True,
                         download_name='pzem_history.tsv',
                         mimetype='text/tab-separated-values')
    abort(404)


# ── API analytics ─────────────────────────────────────────────────────────────
# stats / anomalies / peaks → campione recente (tail STATS_ROWS righe, ~2 ore)
# forecast / cost           → pzem_daily.json (pochi KB, aggiornato ogni ora)

@app.route('/api/analytics/stats')
def api_stats():
    rows = analytics.load_rows_cached()
    return jsonify(analytics.descriptive_stats(rows))


@app.route('/api/analytics/hourly')
def api_hourly():
    field = request.args.get('field', 'power')
    rows  = analytics.load_rows_cached()
    return jsonify(analytics.hourly_averages(rows, field))


@app.route('/api/analytics/daily')
def api_daily():
    """Restituisce l'aggregato giornaliero dal JSON (velocissimo)."""
    return jsonify(analytics.daily_totals())


@app.route('/api/analytics/peak_hours')
def api_peak():
    rows = analytics.load_rows_cached()
    return jsonify(analytics.peak_hours(rows))


@app.route('/api/analytics/anomalies')
def api_anomalies():
    rows = analytics.load_rows_cached()
    return jsonify(analytics.detect_anomalies(rows))


@app.route('/api/analytics/forecast')
def api_forecast():
    days = int(request.args.get('days', 7))
    # rows=None: energy_forecast legge da daily_totals() → JSON
    return jsonify(analytics.energy_forecast(days_ahead=days))


@app.route('/api/analytics/power_profile')
def api_profile():
    rows = analytics.load_rows_cached()
    return jsonify(analytics.power_profile_forecast(rows))


@app.route('/api/analytics/cost')
def api_cost():
    price = float(request.args.get('price', 0.25))
    # rows=None: cost_analysis legge da daily_totals() → JSON
    return jsonify(analytics.cost_analysis(price_kwh=price))


@app.route('/api/analytics/trigger_daily_update')
def api_trigger_daily():
    """Endpoint di debug per forzare l'aggiornamento dell'aggregato."""
    try:
        analytics.update_daily_aggregate()
        daily = analytics._load_daily_json()
        return jsonify({'ok': True, 'days': len(daily)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_tsv()

    # Thread polling sensore
    t_poll = threading.Thread(target=poll_loop, daemon=True)
    t_poll.start()

    # Thread aggiornamento aggregato giornaliero
    t_daily = threading.Thread(target=daily_update_loop, daemon=True)
    t_daily.start()

    log.info("Dashboard → http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)