"""Node console: the human gate, and the label buffer behind it.

    python -m streamlit run application/node_app.py

This screen does three jobs at once, which is why the design report argues it
is the most important component in the system rather than an optional extra:

  It is the false-positive control. Nothing reaches the coordinating server
  until a named person here has looked at the crop and said yes.

  It is the legal safeguard. Every disclosure has a human attached to it, and
  every decision is written to an audit log that both sides can read.

  It is the label source. Confirm and reject are binary labels from this
  camera's own footage. They are counted and audited here; feeding them into
  federated training rounds is future work.

Note what the watchlist tab shows: case references and key fingerprints. Not
photographs. This node cannot see the faces it is searching for, and does not
need to.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st                                          # noqa: E402

from fedmps.config import default_config                        # noqa: E402
from fedmps.store import (                                       # noqa: E402
    STATUS_CONFIRMED, STATUS_PENDING, STATUS_REJECTED, Store,
)

st.set_page_config(page_title="FedMPS · Node console",
                   page_icon="🎥", layout="wide")


@st.cache_resource
def _cfg():
    return default_config()


@st.cache_resource
def _store(_path: str) -> Store:
    return Store(Path(_path))


cfg = _cfg()
store = _store(str(cfg.paths.artifacts / "fedmps.sqlite"))
ROOT = Path(__file__).resolve().parents[1]

SCAN_MODES = {
    "first": "find the first match, then stop",
    "full": "scan the whole video for every match",
}


def _fmt_secs(s) -> str:
    if s is None:
        return "estimating…"
    s = int(round(s))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def _read_progress(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def start_scan(node: int, video: Path, first_match_only: bool) -> None:
    """Run the pipeline in a separate process, so the console stays live and
    can draw a progress bar from the file the scan writes after every frame."""
    progress_file = cfg.paths.artifacts / f"scan_node{node}.json"
    log_file = cfg.paths.artifacts / f"scan_node{node}.log"
    progress_file.unlink(missing_ok=True)
    cmd = [
        sys.executable, "application/run_video_scan.py",
        "--video", str(video),
        "--node", str(node),
        "--camera", f"cam-{'ABCDE'[node % 5]} (site {node})",
        "--fps-cap", "0",
        "--sample-fps", "4",
        "--progress-file", str(progress_file),
    ]
    if first_match_only:
        cmd.append("--stop-on-alert")
    log = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    st.session_state["scan"] = {
        "proc": proc, "log": log, "log_file": log_file,
        "progress_file": progress_file, "node": node,
        "mode": "first" if first_match_only else "full",
    }
    st.session_state.pop("last_scan", None)


def finish_scan(stopped_by_user: bool = False) -> None:
    scan = st.session_state.pop("scan")
    scan["log"].close()
    state = _read_progress(scan["progress_file"])
    if stopped_by_user:
        state["phase"] = "stopped"
    elif state.get("phase") != "done":
        tail = scan["log_file"].read_text(encoding="utf-8", errors="replace")
        state = {"phase": "error",
                 "error": state.get("error") or tail.strip()[-800:]}
    state["mode"] = scan["mode"]
    state["node"] = scan["node"]
    st.session_state["last_scan"] = state

with st.sidebar:
    st.header("This node")
    # Open a given node directly with http://localhost:8502/?node=2
    try:
        start_node = min(max(int(st.query_params.get("node", 0)), 0), 31)
    except ValueError:
        start_node = 0
    node_id = st.number_input("Node id", min_value=0, max_value=31, value=start_node)
    operator = st.text_input("Reviewing as", value=f"operator@node{node_id}",
                             help="Recorded against every decision you make.")
    st.divider()
    st.caption(
        "Raw video, face crops of non-matches, embeddings and the label "
        "buffer never leave this device. Only a confirmed alert does, and "
        "only when you say so.")
    st.divider()
    st.subheader("Camera Feed Simulation")
    video_file = cfg.paths.media / f"node{int(node_id)}.mp4"
    scanning = "scan" in st.session_state
    if video_file.exists():
        st.caption(f"Video: `media/node{int(node_id)}.mp4`")
        if st.button("🔍 Find first match and stop", type="primary",
                     disabled=scanning, width="stretch",
                     help="Stops as soon as one person is matched and raised "
                          "as an alert: the fastest way to answer whether "
                          "this person was seen here."):
            start_scan(int(node_id), video_file, first_match_only=True)
            st.rerun()
        if st.button("🎞 Scan full video, all matches",
                     disabled=scanning, width="stretch",
                     help="Scans every sampled frame to the end and raises an "
                          "alert for every matching track."):
            start_scan(int(node_id), video_file, first_match_only=False)
            st.rerun()
    else:
        st.caption(f"No video file found at {video_file}")

node_id = int(node_id)
st.title(f"Node {node_id} review console")

pending = store.list_alerts(node_id=node_id, status=STATUS_PENDING)
confirmed = store.list_alerts(node_id=node_id, status=STATUS_CONFIRMED)
rejected = store.list_alerts(node_id=node_id, status=STATUS_REJECTED)
watchlist = store.templates_for_node(node_id)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Awaiting review", len(pending))
c2.metric("Confirmed", len(confirmed))
c3.metric("Rejected", len(rejected))
c4.metric("Cases watched", len(watchlist))

# --------------------------------------------------------------------------
# Scan status
# --------------------------------------------------------------------------

# Filled at the end of the script, so the tabs below render first.
status_box = st.container()

if "last_scan" in st.session_state and "scan" not in st.session_state:
    last = st.session_state["last_scan"]
    alerts = last.get("alerts", [])
    with status_box.container(border=True):
        if last["phase"] == "error":
            st.error("The scan failed.")
            st.code(last.get("error", ""), language=None)
        elif last["phase"] == "stopped":
            st.warning("Scan stopped. Any alerts raised before stopping are "
                       "in the review queue.")
        elif last["mode"] == "first" and alerts:
            total = last.get("total_frames") or 0
            where = (f" at frame {last['frame']} of {total} "
                     f"({last['frame'] / total:.0%} of the video)"
                     if total else f" at frame {last['frame']}")
            st.success(f"Match found{where}; scan stopped. Review it below.")
        elif last["mode"] == "first":
            st.info("Whole video scanned: no match found.")
        else:
            st.success(f"Full video scanned in "
                       f"{_fmt_secs(last.get('elapsed_s'))}: "
                       f"{len(alerts)} match(es) raised for review.")
        if alerts:
            names = {w["case_id"]: w["name"] for w in store.templates_for_node(
                last["node"], include_inactive=True)}
            st.dataframe([{
                "alert": a["alert_id"],
                "person": names.get(a["case_id"], a["case_id"]),
                "track": a["track_id"],
                "score": round(a["score"], 3),
            } for a in alerts], width="stretch", hide_index=True)
        if st.button("Dismiss"):
            del st.session_state["last_scan"]
            st.rerun()

tab_queue, tab_watchlist, tab_labels = st.tabs(
    ["Review queue", "Watchlist", "Training labels"])


# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------

with tab_queue:
    if not pending:
        st.success("Nothing awaiting review.")
    else:
        st.caption(
            f"{len(pending)} candidate(s). Compare the reported photograph "
            f"against what the camera saw. Confirming releases this one crop "
            f"upstream; rejecting keeps it here and discards it. Opening an "
            f"alert pulls the reference photograph from the server and is "
            f"recorded in the audit log.")

    for a in pending:
        with st.container(border=True):
            cols = st.columns([1.1, 1.1, 2.4, 1.6, 1.8])

            # Who we are looking for. Pulled from the server only now, when a
            # human is actually reviewing, and the pull is written to the
            # audit log. Without this the operator is asked to confirm a face
            # they have never seen, which makes the human gate meaningless.
            ref = store.reference_thumb(a["case_id"], actor=operator)
            if ref:
                cols[0].image(ref, width=130, caption="Reported missing")
            else:
                cols[0].info("no reference photograph")

            crop = Path(a["crop_path"] or "")
            if crop.exists():
                cols[1].image(str(crop), width=130, caption="Seen by camera")
            else:
                cols[1].warning("crop missing")

            cols[2].markdown(
                f"**{a['case_name'] or a['case_id']}**  \n"
                f"`{a['case_id']}`  \n"
                f"camera **{a['camera']}** · track **{a['track_id']}**  \n"
                f"{a['created_utc'][:19].replace('T', ' ')} UTC")
            cols[3].metric("Match score", f"{a['score']:.3f}")
            cols[3].caption(f"{a['frames_voted']} of {a['frames_seen']} "
                            f"frames in this track agreed")

            if cols[4].button("Confirm — send upstream",
                              key=f"ok-{a['alert_id']}", type="primary"):
                store.review_alert(a["alert_id"], STATUS_CONFIRMED, operator)
                st.rerun()
            if cols[4].button("Reject — not this person",
                              key=f"no-{a['alert_id']}"):
                store.review_alert(a["alert_id"], STATUS_REJECTED, operator)
                st.rerun()

    if confirmed or rejected:
        with st.expander(f"Reviewed here ({len(confirmed) + len(rejected)})"):
            rows = [{
                "alert": a["alert_id"], "case": a["case_id"],
                "score": round(a["score"], 3), "status": a["status"],
                "by": a["reviewed_by"], "at": (a["reviewed_utc"] or "")[:19],
            } for a in (confirmed + rejected)]
            st.dataframe(rows, width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------

with tab_watchlist:
    st.subheader("What this node is searching for")
    st.caption(
        f"{cfg.privacy.template_bits}-bit keyed codes. No photographs are "
        f"held here, and the same person enrolled at another node has an "
        f"entirely different pattern.")
    if cfg.pipeline.match_mode != "template":
        st.caption(
            "Cosine match mode: this node also holds each case's enrolled "
            "512-d embedding and matches against it. Template mode matches "
            "on the keyed codes alone.")

    if not watchlist:
        st.info("No active cases. The server distributes them; closed and "
                "expired cases drop off automatically.")
    else:
        rows = [{
            "case": w["case_id"],
            "name": w["name"],
            "template (hex)": bytes(w["template"]).hex(),
            "key": w["key_fp"],
            "issued": w["issued_utc"][:19],
            "expires": (w["expires_utc"] or "—")[:10],
        } for w in watchlist]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(f"{len(watchlist)} case(s) × "
                   f"{cfg.privacy.template_bits // 8} bytes = "
                   f"{len(watchlist) * cfg.privacy.template_bits // 8} bytes "
                   f"of watchlist in total.")


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

with tab_labels:
    st.subheader("Training labels produced here")
    reviewed = len(confirmed) + len(rejected)
    st.write(
        "Every decision on the review queue is a labelled example. Confirms "
        "are positives; rejects are hard negatives — near-misses that this "
        "camera actually produced. They stay on this device. Feeding them "
        "into the next federated training round is future work; today they "
        "are counted here and recorded in the audit log."
    )

    l1, l2, l3 = st.columns(3)
    l1.metric("Labelled examples", reviewed)
    l2.metric("Positives", len(confirmed))
    l3.metric("Hard negatives", len(rejected))

    if reviewed:
        precision = len(confirmed) / reviewed
        st.metric("Local precision", f"{precision:.0%}",
                  help="Of the alerts you reviewed, how many were real. This "
                       "is the number that decides whether the system is "
                       "usable, far more than any benchmark accuracy.")
        if precision < 0.3:
            st.warning(
                "Most alerts are false. Raise `pipeline.match_threshold` or "
                "`track_vote_k` in the config — an operator drowning in false "
                "positives stops reading them, and the human gate stops "
                "working.")

    st.divider()
    st.caption("Audit trail for this node")
    entries = [e for e in store.list_audit(400)
               if f"node_{node_id}" in e["actor"] or operator in e["actor"]]
    if entries:
        st.dataframe(entries[:100], width="stretch", hide_index=True)
    else:
        st.info("Nothing logged for this node yet.")


# --------------------------------------------------------------------------
# Scan in progress: poll the scan's progress file and redraw the bar. This
# runs last so the review queue above stays usable while the video is scanned.
# --------------------------------------------------------------------------

if "scan" in st.session_state:
    scan = st.session_state["scan"]
    with status_box.container(border=True):
        head, stop_col = st.columns([5, 1])
        head.markdown(f"**Scanning node {scan['node']}'s video**: "
                      f"{SCAN_MODES[scan['mode']]}")
        if stop_col.button("■ Stop scan", width="stretch"):
            scan["proc"].terminate()
            scan["proc"].wait()
            finish_scan(stopped_by_user=True)
            st.rerun()
        bar = st.progress(0.0, text="Loading the face models…")
        detail = st.empty()
        while scan["proc"].poll() is None:
            p = _read_progress(scan["progress_file"])
            if p.get("phase") == "scanning":
                frac = p.get("fraction")
                found = len(p.get("alerts", []))
                if frac is None:
                    bar.progress(0.0, text=f"Scanning… frame {p['frame']}")
                else:
                    bar.progress(frac, text=(
                        f"Scanning… {frac:.0%} · frame {p['frame']} of "
                        f"{p['total_frames']} · about "
                        f"{_fmt_secs(p.get('eta_s'))} left"))
                detail.caption(
                    f"{found} match(es) so far · {p['processed']} frames "
                    f"analysed in {_fmt_secs(p.get('elapsed_s'))}")
                # A new alert is in the store: redraw the page so it shows up
                # in the review queue while the scan carries on.
                if found > scan.get("shown", 0):
                    scan["shown"] = found
                    st.rerun()
            time.sleep(0.5)
        finish_scan()
        st.rerun()
