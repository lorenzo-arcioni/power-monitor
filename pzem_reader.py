"""
pzem_reader.py — Lettura sensore PZEM-004t via Modbus-RTU.
Nessuna simulazione: se il sensore non è disponibile, ritorna None.
"""

import logging
from datetime import datetime

log = logging.getLogger(__name__)

SERIAL_PORT  = '/dev/ttyUSB0'
BAUD_RATE    = 9600
SLAVE_ID     = 1
TIMEOUT      = 2.0


def read_sensor() -> dict | None:
    """
    Legge i registri del PZEM-004t e restituisce un dict con tutti i valori.
    Ritorna None se la porta non esiste o la lettura fallisce.
    """
    try:
        import serial
        import modbus_tk.defines as cst
        from modbus_tk import modbus_rtu

        sensor = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=8,
            parity='N',
            stopbits=1,
            xonxoff=0,
        )
        master = modbus_rtu.RtuMaster(sensor)
        master.set_timeout(TIMEOUT)
        master.set_verbose(False)

        data = master.execute(SLAVE_ID, cst.READ_INPUT_REGISTERS, 0, 10)

        voltage      = data[0] / 10.0
        current      = (data[1] + (data[2] << 16)) / 1000.0
        power        = (data[3] + (data[4] << 16)) / 10.0
        energy       = data[5] + (data[6] << 16)
        frequency    = data[7] / 10.0
        power_factor = data[8] / 100.0
        alarm        = data[9]

        try:
            master.close()
            if sensor.is_open:
                sensor.close()
        except Exception:
            pass

        apparent = round(voltage * current, 1)
        reactive = round(
            max((apparent ** 2 - power ** 2), 0) ** 0.5, 1
        )

        return {
            'timestamp':    datetime.now().isoformat(timespec='seconds'),
            'voltage':      round(voltage, 1),
            'current':      round(current, 3),
            'power':        round(power, 1),
            'energy':       energy,
            'frequency':    round(frequency, 1),
            'power_factor': round(power_factor, 2),
            'alarm':        alarm,
            'apparent_power': apparent,
            'reactive_power': reactive,
        }

    except Exception as e:
        log.warning(f"Lettura sensore fallita: {e}")
        return None
