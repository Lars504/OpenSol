"""Host RAM status shared by every backend (no cupy, GUI-thread safe): the available physical memory that bounds CPU
factorisations, from psutil when installed, else the Windows / Linux system query, else None (unknown)."""
import ctypes
import sys

__all__ = ['available_bytes', 'total_bytes']


def available_bytes() -> int | None:
    """Physical RAM [bytes] the OS can hand out without swapping (psutil, else GlobalMemoryStatusEx, else
    /proc/meminfo MemAvailable); None when no query works. Never raises."""
    return _first_answer((_psutil_available, _windows_available, _linux_available))


def total_bytes() -> int | None:
    """Installed physical RAM [bytes] (the same sources as available_bytes); None when unknown. Never raises."""
    return _first_answer((_psutil_total, _windows_total, _linux_total))


# ---- helpers ----
def _first_answer(queries):
    for query in queries:
        try:
            value = query()
        except Exception:   # a missing module, API or file: try the next source
            value = None
        if value is not None and value >= 0:
            return int(value)
    return None


def _psutil_available():
    import psutil   # optional dependency
    return psutil.virtual_memory().available


def _psutil_total():
    import psutil
    return psutil.virtual_memory().total


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def _windows_status():
    if sys.platform != 'win32':
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status


def _windows_available():
    status = _windows_status()
    return None if status is None else int(status.ullAvailPhys)


def _windows_total():
    status = _windows_status()
    return None if status is None else int(status.ullTotalPhys)


def _linux_meminfo(key):
    with open('/proc/meminfo', encoding='ascii') as meminfo:
        for line in meminfo:
            if line.startswith(key):
                return int(line.split()[1]) * 1024   # the file states kB (= KiB)
    return None


def _linux_available():
    return _linux_meminfo('MemAvailable:')


def _linux_total():
    return _linux_meminfo('MemTotal:')
