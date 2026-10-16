import pandas as pd
import numpy as np

DATA = "Data"
SEV = {"Low": 1, "Medium": 2, "High": 4, "Critical": 8}
ROAD = {"Good": 0, "Fair": 1, "Poor": 2, "Critical": 3}
PED = {"Low": 0, "Medium": 1, "High": 2}
OPEN = {"Open", "Assigned", "In progress"}
W = {"severity": .25, "frequency": .20, "recent": .15, "exposure": .15, "backlog": .15, "road": .10}
NAMES = {"severity": "Severity", "frequency": "Frequency", "recent": "Recent trend",
         "exposure": "Exposure", "backlog": "Backlog", "road": "Road condition"}


def build(data_dir=DATA):
    loc = pd.read_csv(f"{data_dir}/locations.csv")
    inc = pd.read_csv(f"{data_dir}/incidents.csv", parse_dates=["timestamp"])
    com = pd.read_csv(f"{data_dir}/public_complaints.csv", parse_dates=["created_at"])

    ref = max(inc.timestamp.max(), com.created_at.max())          # "today" = end of the data
    inc["sev_pts"] = inc.severity.map(SEV).fillna(1)
    inc["is_hc"] = inc.severity.isin(["High", "Critical"]).astype(int)
    inc["recent_n"] = (inc.timestamp >= ref - pd.Timedelta(days=90)).astype(int)
    inc["recent_pts"] = inc.sev_pts * inc.recent_n
    inc["unres"] = (inc.resolved == 0).astype(int)
    com["open"] = com.status.isin(OPEN).astype(int)

    i = inc.groupby("location_id").agg(
        inc_count=("incident_id", "size"), sev_load=("sev_pts", "sum"), high_crit=("is_hc", "sum"),
        injuries=("injuries", "sum"), recent_pts=("recent_pts", "sum"), recent_n=("recent_n", "sum"),
        inc_unres=("unres", "sum"), avg_res_hours=("resolution_hours", "mean"))
    c = com.groupby("location_id").agg(
        com_count=("complaint_id", "size"), com_open=("open", "sum"), avg_days=("days_to_resolution", "mean"))

    df = loc.merge(i, on="location_id", how="left").merge(c, on="location_id", how="left")
    for col in ["inc_count", "sev_load", "high_crit", "injuries", "recent_pts", "recent_n",
                "inc_unres", "com_count", "com_open"]:
        df[col] = df[col].fillna(0).astype(int)

    # complaints adjusted for population density (dense areas complain more)
    dens = df.population_density_per_km2
    factor = ((dens / dens.median()).clip(0.3, 3)) ** 0.5
    df["com_adj"] = df.com_count / factor
    ped = df.pedestrian_activity.map(PED).fillna(0)
    raw = {
        "severity": df.sev_load,
        "frequency": df.inc_count + df.com_adj,
        "recent": df.recent_pts,
        "exposure": df.near_school + df.near_hospital + ped / 2 + dens / dens.max(),
        "backlog": df.inc_unres + df.com_open,
        "road": df.road_condition.map(ROAD).fillna(1),
    }
    for k, s in raw.items():
        df["c_" + k] = 100 * W[k] * s.rank(pct=True)
    df["score"] = df[["c_" + k for k in W]].sum(axis=1)

    def why(r):
        parts = {
            "severity": f"{r.high_crit} high/critical incidents",
            "frequency": f"{r.inc_count} incidents + {r.com_count} complaints",
            "recent": f"{r.recent_n} incidents in last 90 days",
            "exposure": ", ".join([t for t, f in [("near school", r.near_school), ("near hospital", r.near_hospital),
                                   ("high footfall", r.pedestrian_activity == "High")] if f]) or "dense area",
            "backlog": f"{r.inc_unres + r.com_open} open/unresolved items",
            "road": f"road condition {r.road_condition}",
        }
        top = sorted(W, key=lambda k: r["c_" + k], reverse=True)[:3]
        return "; ".join(parts[k] for k in top)
    df["why"] = df.apply(why, axis=1)

    ev = df.inc_count + df.com_count
    lo, hi = ev.quantile(.25), ev.quantile(.75)

    def conf(r):
        e = r.inc_count + r.com_count
        if e >= hi and r.inc_count > 0 and r.com_count > 0: return "High"
        if e <= lo or r.inc_count == 0 or r.com_count == 0: return "Low"
        return "Medium"
    df["confidence"] = df.apply(conf, axis=1)

    def missing(r):
        m = []
        if r.inc_count == 0: m.append("no incident records")
        if r.com_count == 0: m.append("no complaints on record")
        if r.inc_count and r.recent_n == 0: m.append("no incidents in last 90 days (check if still active)")
        if r.inc_count + r.com_count <= lo: m.append("very little evidence overall")
        return "; ".join(m) or "none flagged"
    df["missing"] = df.apply(missing, axis=1)

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df, inc, com, ref
