"""Process memory metadata on POSIX and Windows; unrelated to metric formulas."""
try:
    from resource import RUSAGE_SELF,getrusage
except ImportError:
    from types import SimpleNamespace
    import psutil
    RUSAGE_SELF=0

    def getrusage(_who):
        info=psutil.Process().memory_info()
        peak=getattr(info,'peak_wset',info.rss)
        return SimpleNamespace(ru_maxrss=peak/1024)
