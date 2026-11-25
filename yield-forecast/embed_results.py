"""
embed_results.py
Puts forecast_output.json into the dashboard page, so the page works as a single static file.

Run (from this folder, after yield_forecast.py):
  python embed_results.py results/forecast_output.json ../crop-yield-forecast.html
"""
import re
import sys
from pathlib import Path

src, page = Path(sys.argv[1]), Path(sys.argv[2])
data = src.read_text(encoding="utf-8").strip()
html = page.read_text(encoding="utf-8")
pat = re.compile(r'(<script id="data" type="application/json">)(.*?)(</script>)', re.S)
if not pat.search(html):
    raise SystemExit(f"No <script id=\"data\"> block in {page}")
page.write_text(pat.sub(lambda m: m.group(1) + data + m.group(3), html, count=1), encoding="utf-8")
print(f"Embedded {src} ({len(data):,} characters) into {page}")
