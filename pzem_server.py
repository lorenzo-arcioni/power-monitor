"""
pzem_server.py — Server Flask per la dashboard PZEM-004t.
Interfaccia esclusivamente con l'hardware tramite pzem_reader.py
e con i dati storici tramite pzem_analytics.py.

Avvio:
    pip install flask flask-cors modbus-tk pyserial
    python3 pzem_server.py
"""

import os
import csv
import time
import threading
import logging
from flask import Flask, jsonify, send_file, abort, request
from flask_cors import CORS

import pzem_reader as reader
import pzem_analytics as analytics

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ── Configurazione ────────────────────────────────────────────────────────────
POLL_INTERVAL = 2       # secondi tra letture
MAX_LIVE_ROWS = 500     # campioni tenuti in RAM
TSV_FILE      = 'pzem_history.tsv'

TSV_HEADERS = [
    'timestamp', 'voltage', 'current', 'power', 'energy',
    'frequency', 'power_factor', 'alarm', 'apparent_power', 'reactive_power'
]

# ── Stato globale thread-safe ─────────────────────────────────────────────────
latest        = {}      # ultima lettura valida
history       = []      # buffer RAM letture recenti
sensor_online = False   # True se l'ultima lettura è andata a buon fine
lock          = threading.Lock()

# ── TSV helpers ───────────────────────────────────────────────────────────────

def init_tsv():
    if not os.path.exists(TSV_FILE):
        with open(TSV_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TSV_HEADERS, delimiter='\t')
            writer.writeheader()
        log.info(f"Creato file storico: {TSV_FILE}")


def append_tsv(row: dict):
    with open(TSV_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TSV_HEADERS, delimiter='\t',
                                extrasaction='ignore')
        writer.writerow(row)


def load_tsv(limit: int = 5000) -> list[dict]:
    rows = []
    if not os.path.exists(TSV_FILE):
        return rows
    with open(TSV_FILE, newline='') as f:
        reader_csv = csv.DictReader(f, delimiter='\t')
        for row in reader_csv:
            rows.append(row)
    return rows[-limit:]


# ── Thread di polling ─────────────────────────────────────────────────────────

def poll_loop():
    global sensor_online
    log.info(f"Avvio polling su {reader.SERIAL_PORT} ogni {POLL_INTERVAL}s")
    while True:
        row = reader.read_sensor()
        with lock:
            if row:
                sensor_online = True
                latest.clear()
                latest.update(row)
                history.append(dict(row))
                if len(history) > MAX_LIVE_ROWS:
                    history.pop(0)
                append_tsv(row)
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
            'data': dict(latest) if sensor_online else None,
        })


@app.route('/api/live')
def api_live():
    with lock:
        return jsonify(list(history))


@app.route('/api/status')
def api_status():
    with lock:
        ok = sensor_online
    return jsonify({
        'sensor_online': ok,
        'serial_port':   reader.SERIAL_PORT,
        'poll_every':    POLL_INTERVAL,
        'tsv_file':      TSV_FILE,
        'buffer_size':   len(history),
    })


# ── API storico ───────────────────────────────────────────────────────────────

@app.route('/api/history')
def api_history():
    limit = min(int(request.args.get('limit', 5000)), 20000)
    return jsonify(load_tsv(limit))


@app.route('/api/download')
def api_download():
    if os.path.exists(TSV_FILE):
        return send_file(TSV_FILE, as_attachment=True,
                         download_name='pzem_history.tsv',
                         mimetype='text/tab-separated-values')
    abort(404)


# ── API analytics ─────────────────────────────────────────────────────────────

@app.route('/api/analytics/stats')
def api_stats():
    rows = analytics.load_rows()
    return jsonify(analytics.descriptive_stats(rows))


@app.route('/api/analytics/hourly')
def api_hourly():
    field = request.args.get('field', 'power')
    rows = analytics.load_rows()
    return jsonify(analytics.hourly_averages(rows, field))


@app.route('/api/analytics/daily')
def api_daily():
    rows = analytics.load_rows()
    return jsonify(analytics.daily_totals(rows))


@app.route('/api/analytics/peak_hours')
def api_peak():
    rows = analytics.load_rows()
    return jsonify(analytics.peak_hours(rows))


@app.route('/api/analytics/anomalies')
def api_anomalies():
    rows = analytics.load_rows()
    return jsonify(analytics.detect_anomalies(rows))


@app.route('/api/analytics/forecast')
def api_forecast():
    days = int(request.args.get('days', 7))
    rows = analytics.load_rows()
    return jsonify(analytics.energy_forecast(rows, days))


@app.route('/api/analytics/power_profile')
def api_profile():
    rows = analytics.load_rows()
    return jsonify(analytics.power_profile_forecast(rows))


@app.route('/api/analytics/cost')
def api_cost():
    price = float(request.args.get('price', 0.25))
    rows = analytics.load_rows()
    return jsonify(analytics.cost_analysis(rows, price))


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_tsv()
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    log.info("Dashboard → http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
