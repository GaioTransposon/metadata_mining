import time
import sys

for i in range(5):
    print(f"stdout message {i}", flush=True)
    print(f"stderr message {i}", file=sys.stderr, flush=True)
    time.sleep(1)

