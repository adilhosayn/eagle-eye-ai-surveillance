from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.api.storage import SQLiteAlertStore


def main() -> None:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit dashboard dependencies are not installed. Install `streamlit` and `pandas`."
        ) from exc

    store = SQLiteAlertStore(
        os.getenv("SENTINEL_ALERTS_DB_PATH", "data/outputs/alerts.db")
    )

    st.set_page_config(page_title="Eagle Eye AI Dashboard", layout="wide", page_icon="🦅")
    st.title("🦅 Eagle Eye AI — Intelligent Security Surveillance Dashboard")

    with st.sidebar:
        st.header("Security Filters")
        camera_options = ["All"] + store.distinct_values("camera_id")
        event_options = ["All"] + store.distinct_values("event_type")
        zone_options = ["All"] + store.distinct_values("zone")
        severity_options = ["All", "critical", "high", "medium", "low"]

        camera_id = st.selectbox("Camera", camera_options)
        event_type = st.selectbox("Event Type", event_options)
        severity = st.selectbox("Alert Severity", severity_options)
        zone = st.selectbox("Security Zone", zone_options)

        default_start = datetime.now(timezone.utc) - timedelta(days=7)
        start_time = st.text_input(
            "Start Time (UTC ISO-8601)",
            default_start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        end_time = st.text_input(
            "End Time (UTC ISO-8601)",
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        limit = st.slider("Max Events Displayed", min_value=10, max_value=500, value=100, step=10)

    filters = {
        "camera_id": None if camera_id == "All" else camera_id,
        "event_type": None if event_type == "All" else event_type,
        "zone": None if zone == "All" else zone,
        "start_time": start_time or None,
        "end_time": end_time or None,
        "severity": None if severity == "All" else severity,
    }

    stats = store.stats(**filters)
    alerts = store.list_alerts(limit=limit, offset=0, **filters)
    frame = pd.DataFrame(alerts)

    # Compute key KPI metrics
    weapon_count = stats["by_event_type"].get("weapon_detected", 0)
    critical_or_high = 0
    for alert in alerts:
        sev = str(alert.get("severity", "medium")).lower()
        if sev in ("critical", "high"):
            critical_or_high += 1

    # Metric Cards
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Total Incidents", stats["total_alerts"])
    metric_col_2.metric("Weapon Detections", weapon_count, delta=f"{weapon_count} Critical" if weapon_count else None)
    metric_col_3.metric("High / Critical Alerts", critical_or_high)
    metric_col_4.metric("Latest Incident", stats["latest_timestamp"] or "None")

    # Visual Charts
    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.subheader("Incidents by Event Type")
        if stats["by_event_type"]:
            event_df = pd.DataFrame(
                [
                    {"event_type": key, "count": value}
                    for key, value in stats["by_event_type"].items()
                ]
            ).set_index("event_type")
            st.bar_chart(event_df)
        else:
            st.info("No security incidents recorded in the selected window.")

    with chart_col_2:
        st.subheader("Incidents by Camera Source")
        if stats["by_camera_id"]:
            camera_df = pd.DataFrame(
                [
                    {"camera_id": key, "count": value}
                    for key, value in stats["by_camera_id"].items()
                ]
            ).set_index("camera_id")
            st.bar_chart(camera_df)
        else:
            st.info("No camera telemetry for the selected window.")

    # Event Table
    st.subheader("Security Event Log")
    if frame.empty:
        st.info("No security events match the current filter criteria.")
        return

    # Add severity emoji indicator for high-visibility dashboards
    if "severity" in frame.columns:
        sev_icons = {"critical": "🔴 CRITICAL", "high": "🟠 HIGH", "medium": "🟡 MEDIUM", "low": "🔵 LOW"}
        frame["severity_indicator"] = frame["severity"].map(lambda s: sev_icons.get(str(s).lower(), "⚪ UNKNOWN"))

    preferred_columns = [
        "timestamp",
        "severity_indicator",
        "event_type",
        "class",
        "confidence",
        "camera_id",
        "zone",
        "track_id",
        "snapshot_path",
        "clip_path",
        "event_id",
    ]
    visible_columns = [
        column for column in preferred_columns if column in frame.columns
    ]
    st.dataframe(frame[visible_columns], use_container_width=True)

    # Detailed Incident Inspection & Evidence Preview
    st.subheader("Incident Evidence Inspector")
    selected_event_id = st.selectbox("Select Incident ID for Evidence", frame["event_id"].tolist())
    selected = next(alert for alert in alerts if alert["event_id"] == selected_event_id)

    detail_col_1, detail_col_2 = st.columns([1, 1])

    with detail_col_1:
        st.markdown("**Incident Telemetry:**")
        st.json(selected)

    with detail_col_2:
        st.markdown("**Captured Visual Evidence:**")
        snapshot_file = selected.get("snapshot_path")
        if snapshot_file and Path(snapshot_file).exists():
            st.image(str(snapshot_file), caption=f"Snapshot: {Path(snapshot_file).name}", use_container_width=True)
        else:
            st.info(f"Snapshot not found or recorded: {snapshot_file}")

        clip_file = selected.get("clip_path")
        if clip_file and Path(clip_file).exists():
            st.video(str(clip_file))
        elif clip_file:
            st.caption(f"Video clip reference: `{clip_file}`")


if __name__ == "__main__":
    main()
