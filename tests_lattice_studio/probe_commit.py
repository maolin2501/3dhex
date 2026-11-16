"""Observe child native reproducer memory without importing geometry libraries."""
import ctypes
from ctypes import wintypes
import subprocess
import sys
import time


class Memory(ctypes.Structure):
    _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[(x,ctypes.c_ulonglong) for x in (
        'ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]


def main():
    process=subprocess.Popen([sys.executable,'-X','faulthandler','-u','-m','tests_lattice_studio.probe_native_trim',*sys.argv[1:]])
    kernel=ctypes.windll.kernel32
    # A Windows venv may launch a second interpreter. Do not mistake the tiny
    # launcher allocation for the native worker's process-private memory.
    minimum=1000.;last=None
    while process.poll() is None:
        memory=Memory();memory.dwLength=ctypes.sizeof(memory);kernel.GlobalMemoryStatusEx(ctypes.byref(memory))
        free=memory.ullAvailPageFile/1024**3
        minimum=min(minimum,free)
        bucket=int(free)
        if bucket!=last:
            print(f'MEM system_available_commit={free:.3f}GiB physical_free={memory.ullAvailPhys/1024**3:.3f}GiB',flush=True);last=bucket
        time.sleep(.25)
    print(f'EXIT {process.returncode}; min_available_commit={minimum:.3f}GiB',flush=True)
    raise SystemExit(process.returncode)


if __name__=='__main__':main()
