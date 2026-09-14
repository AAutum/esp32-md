#!/bin/bash
# Reboot ACM3 expert board into app mode WITHOUT DTR strap hazard.
# Board DTR strap = GPIO0 low at EN release -> every RTS-pulse reset re-enters
# ROM download mode (9/12 lesson). esptool 'run' also pulses RTS — same trap.
# Fix: drive DTR+RTS de-asserted (both high = EN released, GPIO0 high) and pulse
# the USB re-enumeration by closing cleanly; board boots app when DTR is idle.
# Fallback that ALWAYS works: full USB unplug via uhubctl if present, else
# instruct human. Here we use esptool attach (download mode) -> exit cleanly
# by closing port without pulsing (board runs whatever mode straps dictate:
# with GPIO0 held high by de-asserted DTR at EN, app boots).
python3 - <<'PYEOF'
import serial, time
p = serial.Serial()
p.port = '/dev/ttyACM_DISPATCH'
p.baudrate = 115200
p.dtr = False   # DTR de-asserted -> GPIO0 pulled high via strap
p.rts = False   # RTS de-asserted -> EN released (chip runs)
p.open()
time.sleep(0.1)
p.dtr = False
p.rts = True    # EN asserted (chip in reset), GPIO0 high
time.sleep(0.1)
p.rts = False   # EN released with GPIO0 HIGH -> normal app boot
time.sleep(1.5)
p.write(b'PING\n')
time.sleep(1.0)
data = p.read(4096)
print('PING resp:', repr(data[:300]))
p.close()
PYEOF