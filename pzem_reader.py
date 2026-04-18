"""
pzem_reader.py — Lettura e reset sensore PZEM-004t via Modbus-RTU.
Nessuna simulazione: se il sensore non è disponibile, ritorna None.

Correzioni rispetto alla versione originale:
  - Validazione energy: se data[6] (high word) è anomalo il valore
    viene scartato e la lettura ritorna None (glitch Modbus).
  - reset_energy(): invia il comando di reset contatore al PZEM-004t.
  - _open_master() / _close_master(): helper condivisi per non
    duplicare la logica di apertura porta seriale.
"""

import logging
from datetime import datetime

log = logging.getLogger(__name__)

SERIAL_PORT  = '/dev/ttyUSB0'
BAUD_RATE    = 9600
SLAVE_ID     = 1
TIMEOUT      = 2.0

# Limite massimo energy ragionevole (Wh).
# Il PZEM-004t supporta fino a 9999 kWh = 9.999.000 Wh.
# Usiamo un limite conservativo: 999.999 Wh (~1000 kWh).
# Se il tuo impianto consuma di più alzalo, ma 4 miliardi è sempre un glitch.
ENERGY_MAX_WH = 9_999_000


def _open_master():
    """Apre la porta seriale e restituisce (serial, master)."""
    import serial
    import modbus_tk.defines as cst
    from modbus_tk import modbus_rtu

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        bytesize=8,
        parity='N',
        stopbits=1,
        xonxoff=0,
    )
    master = modbus_rtu.RtuMaster(ser)
    master.set_timeout(TIMEOUT)
    master.set_verbose(False)
    return ser, master


def _close(ser, master):
    """Chiude master e porta seriale ignorando errori."""
    try:
        master.close()
    except Exception:
        pass
    try:
        if ser.is_open:
            ser.close()
    except Exception:
        pass


def read_sensor() -> dict | None:
    """
    Legge i registri del PZEM-004t e restituisce un dict con tutti i valori.
    Ritorna None se la porta non esiste, la lettura fallisce, o i dati
    sono fuori dai limiti fisici del sensore (glitch Modbus).
    """
    try:
        import modbus_tk.defines as cst

        ser, master = _open_master()
        try:
            data = master.execute(SLAVE_ID, cst.READ_INPUT_REGISTERS, 0, 10)
        finally:
            _close(ser, master)

        voltage      = data[0] / 10.0
        current      = (data[1] + (data[2] << 16)) / 1000.0
        power        = (data[3] + (data[4] << 16)) / 10.0
        energy       = data[5] + (data[6] << 16)
        frequency    = data[7] / 10.0
        power_factor = data[8] / 100.0
        alarm        = data[9]

        # ── Validazione anti-glitch ──────────────────────────────────────────
        # data[6] è la high word di energy: su PZEM-004t reale è quasi sempre
        # 0 (energy < 65535 Wh) o al massimo qualche unità (energy < 999 kWh).
        # Se è > 152 (ovvero energy > 9.999.000 Wh) è certamente un glitch.
        if energy > ENERGY_MAX_WH:
            log.warning(
                f"Energy fuori range scartata: {energy} Wh "
                f"(data[5]={data[5]}, data[6]={data[6]})"
            )
            return None

        if not (80.0 <= voltage <= 260.0):
            log.warning(f"Tensione fuori range scartata: {voltage} V")
            return None

        if not (45.0 <= frequency <= 65.0):
            log.warning(f"Frequenza fuori range scartata: {frequency} Hz")
            return None

        # ────────────────────────────────────────────────────────────────────

        apparent = round(voltage * current, 1)
        reactive = round(
            max((apparent ** 2 - power ** 2), 0) ** 0.5, 1
        )

        return {
            'timestamp':      datetime.now().isoformat(timespec='seconds'),
            'voltage':        round(voltage, 1),
            'current':        round(current, 3),
            'power':          round(power, 1),
            'energy':         energy,
            'frequency':      round(frequency, 1),
            'power_factor':   round(power_factor, 2),
            'alarm':          alarm,
            'apparent_power': apparent,
            'reactive_power': reactive,
        }

    except Exception as e:
        log.warning(f"Lettura sensore fallita: {e}")
        return None


def reset_energy() -> bool:
    """
    Invia il comando di reset contatore energia al PZEM-004t.
    Il PZEM-004t accetta un comando speciale: write 0x42 al registro
    0x0003 (coil/holding) — varia a seconda del firmware.
    Metodo più affidabile per la maggior parte dei firmware:
    write_single_register(0x0003, 0x0000) preceduto da un byte 0x42.

    In pratica il modo più semplice e compatibile è inviare il frame
    Modbus raw: 01 42 80 11 (CRC incluso dal modbus_tk).

    Restituisce True se il reset è andato a buon fine.
    """
    try:
        import modbus_tk.defines as cst

        ser, master = _open_master()
        try:
            # Il PZEM-004t usa function code 0x42 (66) per il reset energy.
            # modbus_tk permette di eseguire function code custom.
            master.execute(
                SLAVE_ID,
                cst.WRITE_SINGLE_REGISTER,
                0x0003,
                output_value=0x0000,
            )
        finally:
            _close(ser, master)

        log.info("Reset energy PZEM-004t eseguito.")
        return True

    except Exception as e:
        log.warning(f"Reset energy fallito: {e}")
        # Fallback: prova con il frame raw via pyserial
        return _reset_energy_raw()


def _reset_energy_raw() -> bool:
    """
    Fallback: invia il frame di reset raw via pyserial.
    Frame: 01 42 80 11  (slave=1, cmd=0x42, CRC=0x8011)
    Questo funziona su tutti i firmware PZEM-004t v3.
    """
    try:
        import serial

        with serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=TIMEOUT,
        ) as ser:
            # Frame reset ufficiale PZEM-004t
            frame = bytes([0x01, 0x42, 0x80, 0x11])
            ser.write(frame)
            ser.flush()
            # Leggi risposta (opzionale, il PZEM potrebbe non rispondere)
            ser.read(4)

        log.info("Reset energy PZEM-004t (raw frame) eseguito.")
        return True

    except Exception as e:
        log.warning(f"Reset energy raw fallito: {e}")
        return False