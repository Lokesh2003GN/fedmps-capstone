#!/usr/bin/env python3
"""Screenshots of the running application for the thesis.

Runs one complete demo case and captures it:

    server_console.png      server console after the case is opened
    node_review.png         node 2's review queue with the pending alert
    confirmed_sighting.png  server console, confirmed sighting after review
    audit_log.png           server console, audit log of the whole case
    annotated_frame.png     a frame of the node's annotated output video

The demo database and scan outputs are backed up first and restored at the
end, so the team's own demo state is untouched. Chrome is driven headless
through its DevTools protocol (no extra driver download).

Usage:
    python docs/thesis/capture_screens.py
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import websocket

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PY = _sys.executable
SERVER, NODE, DEVTOOLS = 8501, 8502, 9333
DB = ROOT / "artifacts" / "fedmps.sqlite"
VIDEO = ROOT / "media" / "demo_node2_person1.mp4"
ANNOTATED = ROOT / "media" / "demo_node2_person1_annotated.mp4"


def log(msg):
    print(f"[screens] {msg}", flush=True)


class CDP:
    """Minimal Chrome DevTools Protocol client."""

    def __init__(self, port):
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except OSError:
                time.sleep(0.5)
        page = next(t for t in tabs if t["type"] == "page")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120,
                                              suppress_origin=True)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True,
                      returnByValue=True)
        return r.get("result", {}).get("value")

    def wait(self, cond, timeout=180, what="page"):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.js(cond):
                    time.sleep(1.5)          # let the last widgets paint
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"timed out waiting for {what}")

    def open(self, url, width=1400, height=1000):
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                  deviceScaleFactor=1.5, mobile=False)
        self.call("Page.navigate", url=url)

    def tidy(self):
        # Hide Streamlit's own toolbar ("Deploy", menu), which is not the app.
        self.js("for (const s of ['header[data-testid=\"stHeader\"]',"
                "'[data-testid=\"stToolbar\"]','[data-testid=\"stDecoration\"]'])"
                "{const e=document.querySelector(s); if(e) e.style.display='none';} true")

    def shot(self, path):
        data = self.call("Page.captureScreenshot", format="png")["data"]
        path.write_bytes(base64.b64decode(data))
        log(f"wrote {path.name}")


def wait_http(port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"streamlit on {port} did not start")


def annotated_frame(dst: Path):
    cap = cv2.VideoCapture(str(ANNOTATED))
    best, best_score = None, -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        # Alert boxes are drawn in red, BGR (40, 50, 200) in pipeline/match.py.
        red = int(((fr[:, :, 2] > 170) & (fr[:, :, 1] < 90) & (fr[:, :, 0] < 90)).sum())
        if red > best_score:
            best, best_score = fr, red
    if best is None:
        raise RuntimeError("annotated video is empty")
    h, w = best.shape[:2]
    scale = 1280 / max(h, w)
    cv2.imwrite(str(dst), cv2.resize(best, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA))
    log(f"wrote {dst.name}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="fedmps_screens_"))
    keep = [DB] + list((ROOT / "results").glob("video_node*.json")) \
        + list((ROOT / "media").glob("*_annotated.mp4"))
    for f in keep:
        shutil.copy2(f, backup / f.name)
    alerts = ROOT / "media" / "alerts"
    shutil.copytree(alerts, backup / "alerts")      # demo_reset clears this folder
    procs = []
    profile = Path(tempfile.mkdtemp(prefix="fedmps_chrome_"))
    try:
        subprocess.run([PY, "application/demo_reset.py"], cwd=ROOT, check=True,
                       capture_output=True)
        log("demo reset; running the demo scan")
        subprocess.run([PY, "application/run_video_scan.py", "--video", str(VIDEO),
                        "--node", "2", "--camera", "cam-C (site 2)",
                        "--enrol", "media/person-1.jpg", "--enrol-name", "Priya Sharma",
                        "--fps-cap", "0", "--max-frames", "60"],
                       cwd=ROOT, check=True, capture_output=True)
        annotated_frame(OUT / "annotated_frame.png")

        for app, port in (("server_app.py", SERVER), ("node_app.py", NODE)):
            procs.append(subprocess.Popen(
                [PY, "-m", "streamlit", "run", f"application/{app}",
                 "--server.port", str(port), "--server.headless", "true"],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        wait_http(SERVER)
        wait_http(NODE)
        procs.append(subprocess.Popen(
            [str(CHROME), "--headless=new", f"--remote-debugging-port={DEVTOOLS}",
             f"--user-data-dir={profile}", "--no-first-run", "--hide-scrollbars",
             f"--remote-allow-origins=http://127.0.0.1:{DEVTOOLS}",
             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        cdp = CDP(DEVTOOLS)
        cdp.call("Page.enable")

        # 1. server console with the open case
        cdp.open(f"http://localhost:{SERVER}/", height=1050)
        cdp.wait("document.querySelectorAll('[data-testid=\"stMetric\"]').length >= 5",
                 what="server console")
        cdp.tidy()
        cdp.shot(OUT / "server_console.png")

        # 2. node 2's review queue with the pending alert
        cdp.open(f"http://localhost:{NODE}/?node=2", height=1000)
        cdp.wait("[...document.querySelectorAll('button')].some(b => "
                 "b.innerText.includes('Confirm'))", what="pending alert")
        cdp.tidy()
        cdp.shot(OUT / "node_review.png")

        # 3. confirm it, as the operator would
        cdp.js("[...document.querySelectorAll('button')].find(b => "
               "b.innerText.includes('Confirm')).click(); true")
        cdp.wait("document.body.innerText.includes('Nothing awaiting review')",
                 what="confirmation")

        # Keep the audit rows of this case too (used for a table in Chapter 4).
        import sqlite3
        con = sqlite3.connect(DB)
        rows = con.execute("SELECT ts_utc, actor, action, detail FROM audit ORDER BY id").fetchall()
        con.close()
        (OUT / "audit_rows.json").write_text(json.dumps(rows, indent=1))

        # 4. server console: confirmed sighting, then the audit log
        for tab, name, height in (("Confirmed sightings", "confirmed_sighting.png", 900),
                                  ("Audit log", "audit_log.png", 1000)):
            cdp.open(f"http://localhost:{SERVER}/", height=height)
            cdp.wait("document.querySelectorAll('[data-testid=\"stMetric\"]').length >= 5",
                     what="server console")
            cdp.js(f"[...document.querySelectorAll('button[role=tab]')].find(b => "
                   f"b.innerText.includes('{tab}')).click(); true")
            if tab == "Audit log":
                # Data tables are drawn on a canvas; wait until it has painted.
                cdp.wait("[...document.querySelectorAll('[data-testid=\"stDataFrame\"]')]"
                         ".some(d => d.offsetHeight > 50 && d.querySelector('canvas'))",
                         what="audit table")
                time.sleep(6)
            else:
                time.sleep(3)
            cdp.tidy()
            cdp.js("window.scrollTo(0, 0); "
                   "const m = document.querySelector('[data-testid=\"stMain\"]') || "
                   "document.querySelector('section.main'); "
                   f"const t = [...document.querySelectorAll('button[role=tab]')].find(b => b.innerText.includes('{tab}')); "
                   "if (t) t.scrollIntoView({block: 'start'}); true")
            time.sleep(1.5)
            cdp.shot(OUT / name)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in keep:
            shutil.copy2(backup / f.name, f)
        for f in (ROOT / "results").glob("video_node*.json"):
            if not (backup / f.name).exists():
                f.unlink()
        for f in (ROOT / "media").glob("*_annotated.mp4"):
            if not (backup / f.name).exists():
                f.unlink()
        shutil.rmtree(alerts, ignore_errors=True)
        shutil.copytree(backup / "alerts", alerts)
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)
        log("demo database and outputs restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
