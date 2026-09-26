"""Measure one child in a fresh process, avoiding macOS time -l sysctl access."""
import datetime
import json
import pathlib
import resource
import subprocess
import sys
import time

record_path = pathlib.Path(sys.argv[1])
command = sys.argv[2:]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
start = time.monotonic()
result = subprocess.run(command)
elapsed = time.monotonic() - start
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
record = dict(command=command, started_at_utc=started,
              finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              wall_seconds=elapsed, exit_code=result.returncode,
              user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
              peak_rss_bytes_macos=usage.ru_maxrss,
              measurement="Fresh Python process, one child; monotonic wall time and getrusage(RUSAGE_CHILDREN). macOS ru_maxrss in bytes.")
record_path.write_text(json.dumps(record, indent=2) + "\n")
sys.exit(result.returncode)
