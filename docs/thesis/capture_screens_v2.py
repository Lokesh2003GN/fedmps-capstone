#!/usr/bin/env python3
"""Screenshots of the running application for the v2 thesis, node 0 case.

Expects the node-0 scan to be in the demo database already:

    python application/demo_reset.py
    python application/run_video_scan.py --video media/node0.mp4 --node 0 \
        --camera "cam-A (site 0)" --enrol media/person.jpg --enrol-name "Test Subject" --fps-cap 0

Then captures, at a width that stays readable on a printed page:

    v2_server_console.png      server console with the open case
    v2_node_review.png         node 0's review queue with the pending alerts
    v2_confirmed_sighting.png  server console, confirmed sightings after review
    v2_audit_log.png           server console, audit log of the whole case
    v2_annotated_frame.png     a frame of node 0's annotated output video

The operator's decisions are made through the console's own buttons: every
alert is confirmed except the tracks given in --reject, which a visual check of
the alert crops showed to be a different person.

Usage:
    python docs/thesis/capture_screens_v2.py --reject 49
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

import cv2

from capture_screens import CDP, CHROME, DB, OUT, PY, ROOT, log, wait_http

SERVER, NODE, DEVTOOLS = 8511, 8512, 9343
ANNOTATED = ROOT / "media" / "node0_annotated.mp4"
WIDTH = 1300                    # CSS pixels; wide enough that the alert cards do not overlap


def annotated_frame(dst: Path) -> None:
    """The frame with the most alert-red box pixels, at 1600 px wide."""
    cap = cv2.VideoCapture(str(ANNOTATED))
    best, best_score = None, -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        red = int(((fr[:, :, 2] > 170) & (fr[:, :, 1] < 90) & (fr[:, :, 0] < 90)).sum())
        if red > best_score:
            best, best_score = fr, red
    if best is None:
        raise RuntimeError("annotated video is empty")
    h, w = best.shape[:2]
    scale = 1600 / w
    cv2.imwrite(str(dst), cv2.resize(best, (1600, int(h * scale)), interpolation=cv2.INTER_AREA))
    log(f"wrote {dst.name}")


def open_page(cdp: CDP, url: str, height: int) -> None:
    cdp.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=height,
             deviceScaleFactor=2, mobile=False)
    cdp.call("Page.navigate", url=url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--reject", type=int, nargs="*", default=[],
                    help="track ids to reject; every other pending alert is confirmed")
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    pending = con.execute("SELECT alert_id, track_id FROM alerts WHERE node_id=? AND "
                          "status='pending' ORDER BY alert_id", (args.node,)).fetchall()
    con.close()
    if not pending:
        raise SystemExit("no pending alerts: run the node scan first (see the docstring)")
    log(f"{len(pending)} pending alerts; rejecting tracks {args.reject}")

    annotated_frame(OUT / "v2_annotated_frame.png")
    procs = []
    profile = Path(tempfile.mkdtemp(prefix="fedmps_chrome_"))
    try:
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
        open_page(cdp, f"http://localhost:{SERVER}/", height=1250)
        cdp.wait("document.querySelectorAll('[data-testid=\"stMetric\"]').length >= 5",
                 what="server console")
        cdp.tidy()
        cdp.shot(OUT / "v2_server_console.png")

        # 2. node's review queue with the pending alerts
        open_page(cdp, f"http://localhost:{NODE}/?node={args.node}", height=1350)
        cdp.wait("[...document.querySelectorAll('button')].some(b => "
                 "b.innerText.includes('Confirm'))", what="pending alerts")
        cdp.tidy()
        cdp.shot(OUT / "v2_node_review.png")

        # 3. the operator's decisions, one alert at a time
        for alert_id, track_id in pending:
            key = f"no-{alert_id}" if track_id in args.reject else f"ok-{alert_id}"
            cdp.wait(f"!!document.querySelector('.st-key-{key} button')",
                     what=f"button {key}")
            cdp.js(f"document.querySelector('.st-key-{key} button').click(); true")
            cdp.wait(f"!document.querySelector('.st-key-{key} button')",
                     what=f"decision on alert {alert_id}")
            log(f"alert {alert_id} (track {track_id}): {key.split('-')[0]}")

        # Keep the audit rows of this case for the text.
        con = sqlite3.connect(DB)
        rows = con.execute("SELECT ts_utc, actor, action, detail FROM audit ORDER BY id").fetchall()
        con.close()
        (OUT / "v2_audit_rows.json").write_text(json.dumps(rows, indent=1))

        # 4. server console: confirmed sightings, then the audit log
        for tab, name, height in (("Confirmed sightings", "v2_confirmed_sighting.png", 1150),
                                  ("Audit log", "v2_audit_log.png", 1150)):
            open_page(cdp, f"http://localhost:{SERVER}/", height=height)
            cdp.wait("document.querySelectorAll('[data-testid=\"stMetric\"]').length >= 5",
                     what="server console")
            cdp.js(f"[...document.querySelectorAll('button[role=tab]')].find(b => "
                   f"b.innerText.includes('{tab}')).click(); true")
            if tab == "Audit log":
                cdp.wait("[...document.querySelectorAll('[data-testid=\"stDataFrame\"]')]"
                         ".some(d => d.offsetHeight > 50 && d.querySelector('canvas'))",
                         what="audit table")
                time.sleep(6)
            else:
                time.sleep(3)
            cdp.tidy()
            cdp.js("window.scrollTo(0, 0); "
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
