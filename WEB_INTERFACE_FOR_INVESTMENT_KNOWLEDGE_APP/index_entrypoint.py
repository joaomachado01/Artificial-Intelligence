# Waits until Qdrant is ready, then runs investment_rag.py as a script,
# exactly as you would in the terminal: python investment_rag.py Documents
# Used by the "indexer" service in docker-compose.yml.
import os
import subprocess
import sys
import time
import urllib.request

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

for attempt in range(60):
    try:
        urllib.request.urlopen(f"{qdrant_url}/readyz", timeout=2)
        break
    except Exception:
        print("Waiting for Qdrant...")
        time.sleep(2)
else:
    raise SystemExit("Qdrant did not become ready in time.")

# Run the indexing script as a separate process; check=True makes this
# container fail if the script fails, so the API won't start on a bad index
subprocess.run([sys.executable, "investment_rag.py", "Documents"], check=True)