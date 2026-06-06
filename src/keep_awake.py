"""Cegah Windows sleep selama job panjang (mis. training Group A).

Non-invasif: pakai SetThreadExecutionState (tak mengubah power plan user).
Efek hanya selama proses ini hidup; kill -> sleep normal kembali.

    python src/keep_awake.py        # jalankan di background, kill saat job selesai
"""
import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

k32 = ctypes.windll.kernel32
k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print("keep-awake AKTIF (ES_CONTINUOUS|ES_SYSTEM_REQUIRED) — sistem tidak akan sleep")
try:
    while True:
        time.sleep(120)
        k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)  # re-assert
except KeyboardInterrupt:
    pass
finally:
    k32.SetThreadExecutionState(ES_CONTINUOUS)  # lepas, sleep normal lagi
