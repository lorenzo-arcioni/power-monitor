"""
pzem_server.py — Server Flask per la dashboard PZEM-004t.
Ottimizzazioni rispetto alla versione originale:
  - history usa collections.deque(maxlen=) → pop(0) O(1) invece di O(n)
  - /api/live accetta ?since= per restituire solo i campioni nuovi (delta)
  - write-buffer TSV: flush al disco ogni WRITE_BATCH righe anziché ad ogni campione
  - cache in-memory per load_tsv() con TTL configurabile
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
POLL_INTERVAL  = 2       # secondi tra letture
MAX_LIVE_ROWS  = 500     # campioni in RAM
TSV_FILE       = 'pzem_history.tsv'
WRITE_BATCH    = 10      # righe accumulate prima del flush su disco
CACHE_TTL      = 30      # secondi di validità della cache TSV

TSV_HEADERS = [
    'timestamp', 'voltage', 'current', 'power', 'energy',
    'frequency', 'power_factor', 'alarm', 'apparent_power', 'reactive_power'
]

# ── Stato globale thread-safe ─────────────────────────────────────────────────
latest        = {}
# deque con maxlen: il pop dell'elemento più vecchio è O(1) e automatico
history: deque = deque(maxlen=MAX_LIVE_ROWS)
sensor_online = False
lock          = threading.Lock()

# Write-buffer per il TSV
_write_buf: list[dict] = []
_write_buf_lock = threading.Lock()

# Cache per load_tsv
_tsv_cache: dict = {'rows': None, 'ts': 0.0}
_tsv_cache_lock = threading.Lock()

# ── TSV helpers ───────────────────────────────────────────────────────────────

def init_tsv():
    if not os.path.exists(TSV_FILE):
        with open(TSV_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=TSV_HEADERS,
                           delimiter='\t').writeheader()
        log.info(f"Creato file storico: {TSV_FILE}")


def _flush_buf(buf: list[dict]):
    """Scrive buf su disco in un colpo solo e svuota la lista."""
    with open(TSV_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TSV_HEADERS,
                           delimiter='\t', extrasaction='ignore')
        w.writerows(buf)
    buf.clear()
    # invalida la cache TSV così la prossima lettura è fresca
    with _tsv_cache_lock:
        _tsv_cache['ts'] = 0.0


def queue_tsv(row: dict):
    """Accoda una riga nel buffer; flush ogni WRITE_BATCH righe."""
    with _write_buf_lock:
        _write_buf.append(row)
        if len(_write_buf) >= WRITE_BATCH:
            _flush_buf(_write_buf)


def load_tsv_cached(limit: int = 5000) -> list[dict]:
    """
    Restituisce le ultime `limit` righe del TSV usando una cache in-memory
    con TTL=CACHE_TTL secondi. Evita riletture ripetute su file grandi.
    """
    now = time.monotonic()
    with _tsv_cache_lock:
        if _tsv_cache['rows'] is not None and (now - _tsv_cache['ts']) < CACHE_TTL:
            rows = _tsv_cache['rows']
            return rows[-limit:] if len(rows) > limit else rows

    # Cache scaduta o vuota: rilegge dal disco
    rows: list[dict] = []
    if os.path.exists(TSV_FILE):
        with open(TSV_FILE, newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                rows.append(row)

    with _tsv_cache_lock:
        _tsv_cache['rows'] = rows
        _tsv_cache['ts']   = now

    return rows[-limit:] if len(rows) > limit else rows


# ── Thread di polling ─────────────────────────────────────────────────────────

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
                history.append(dict(row))   # deque gestisce da solo il maxlen
                queue_tsv(row)
            else:
                sensor_online = False
        time.sleep(POLL_INTERVAL)


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
    Restituisce i campioni in RAM.
    Con ?since=<timestamp ISO> restituisce SOLO i campioni più recenti
    di quel timestamp → il client riceve solo il delta, non l'intera history.
    Esempio: GET /api/live?since=2024-05-01T12:00:00
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
    limit = min(int(request.args.get('limit', 5000)), 20000)
    return jsonify(load_tsv_cached(limit))


@app.route('/api/download')
def api_download():
    # Flush buffer prima del download per avere dati completi
    with _write_buf_lock:
        if _write_buf:
            _flush_buf(_write_buf)
    if os.path.exists(TSV_FILE):
        return send_file(TSV_FILE, as_attachment=True,
                         download_name='pzem_history.tsv',
                         mimetype='text/tab-separated-values')
    abort(404)


# ── API analytics ─────────────────────────────────────────────────────────────
# Tutte usano load_tsv_cached() tramite analytics.load_rows_cached()

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
    rows = analytics.load_rows_cached()
    return jsonify(analytics.daily_totals(rows))


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
    rows = analytics.load_rows_cached()
    return jsonify(analytics.energy_forecast(rows, days))


@app.route('/api/analytics/power_profile')
def api_profile():
    rows = analytics.load_rows_cached()
    return jsonify(analytics.power_profile_forecast(rows))


@app.route('/api/analytics/cost')
def api_cost():
    price = float(request.args.get('price', 0.25))
    rows  = analytics.load_rows_cached()
    return jsonify(analytics.cost_analysis(rows, price))


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_tsv()
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    log.info("Dashboard → http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
