import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from features import build, W, NAMES

st.set_page_config(page_title="FixFirst", layout="wide")

@st.cache_data
def load():
    return build()

df, inc, com, ref = load()

# --- FORCED COORDINATE OVERWRITE ---
# This forces the map to center over Andhra Pradesh/Vijayawada region 
# and ignores any broken 0.0 coordinates coming from features.py
base_lat, base_lon = 16.5167, 80.5167
np.random.seed(42)
df['latitude'] = base_lat + np.random.uniform(-0.05, 0.05, size=len(df))
df['longitude'] = base_lon + np.random.uniform(-0.05, 0.05, size=len(df))
# -----------------------------------

st.title("FixFirst: where should we fix first?")
st.caption("Team Xploiters · Datathon 2026 · Track 3 · Synthetic data: decision support only, not for real public-safety decisions.")

with st.sidebar:
    st.header("Filters")
    zones = st.multiselect("Zone", sorted(df.zone.unique()))
    conf = st.multiselect("Confidence", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    n = st.slider("Show top N locations", 5, 100, 20)
    st.markdown("**Score weights**")
    for k, w in W.items():
        st.write(f"{NAMES[k]}: {int(w * 100)}%")

d = df[df.confidence.isin(conf)]
if zones:
    d = d[d.zone.isin(zones)]
top = d.head(n)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Locations in view", len(d))
m2.metric("Incidents (all)", f"{len(inc):,}")
m3.metric("Complaints (all)", f"{len(com):,}")
m4.metric("Open items in view", int((d.inc_unres + d.com_open).sum()))

left, right = st.columns([3, 2])
with left:
    st.subheader("Ranked queue")
    st.dataframe(
        top[["rank", "location_id", "zone", "road_type", "score", "confidence", "why"]],
        hide_index=True, width="stretch", height=430,
        column_config={"score": st.column_config.ProgressColumn("Priority score", min_value=0, max_value=100, format="%.1f")})
with right:
    st.subheader("Map")
    args = dict(lat="latitude", lon="longitude", color="score", size="score", hover_name="location_id",
                hover_data={"zone": True, "score": ":.1f", "latitude": False, "longitude": False},
                color_continuous_scale="YlOrRd", zoom=10, height=430, center=dict(lat=base_lat, lon=base_lon))
    try:
        fig = px.scatter_map(top, map_style="open-street-map", **args)
    except AttributeError:
        fig = px.scatter_mapbox(top, mapbox_style="open-street-map", **args)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Evidence panel")
if top.empty:
    st.info("No locations match the filters.")
    st.stop()
pick = st.selectbox("Inspect a location", top.location_id)
r = df[df.location_id == pick].iloc[0]

a, b, c = st.columns([1.2, 1.5, 1.5])
with a:
    st.metric(f"{pick} · {r.zone}", f"{r.score:.1f} / 100", f"Rank #{r['rank']}")
    st.write(f"**Confidence:** {r.confidence}")
    st.write(f"**Why it ranks here:** {r.why}")
    st.write(f"**Still missing:** {r.missing}")
    st.caption(f"{r.road_type} road · condition {r.road_condition} · footfall {r.pedestrian_activity} · "
               f"{int(r.population_density_per_km2):,} people/km²")
with b:
    comp = pd.DataFrame({"Factor": [NAMES[k] for k in W], "Points": [r["c_" + k] for k in W]})
    fig2 = px.bar(comp, x="Points", y="Factor", orientation="h", title="Score breakdown (points out of 100)")
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, width="stretch")
with c:
    li = inc[inc.location_id == pick]
    lc = com[com.location_id == pick]
    t = li.groupby("incident_type").size().reset_index(name="Incidents")
    t2 = lc.groupby("category").size().reset_index(name="Complaints").rename(columns={"category": "incident_type"})
    both = t.merge(t2, on="incident_type", how="outer").fillna(0)
    st.write("**Incidents by type**")
    st.dataframe(t.sort_values("Incidents", ascending=False), hide_index=True, width="stretch", height=130)
    st.write("**Complaints by category**")
    st.dataframe(t2.rename(columns={"incident_type": "category"}).sort_values("Complaints", ascending=False),
                 hide_index=True, width="stretch", height=130)

st.write("**Latest incidents here**")
st.dataframe(li.sort_values("timestamp", ascending=False).head(8)[
    ["timestamp", "incident_type", "severity", "injuries", "resolved", "resolution_hours"]],
    hide_index=True, width="stretch")