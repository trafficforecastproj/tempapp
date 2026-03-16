#.\venv\Scripts\Activate.ps1
#python -m uvicorn app:app --reload

from compute_engine import plan_routes
from fastapi import FastAPI, Header, HTTPException, Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import requests
import json
from dotenv import load_dotenv
import os

import traceback
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import math
import asyncio
from collections import defaultdict
import xgboost as xgb
import httpx
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from typing import Any
from fastapi.middleware.cors import CORSMiddleware




load_dotenv()

LTA_API_KEY = os.getenv("LTA_API_KEY")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ONEMAP_TOKEN = os.getenv("authToken")
ONEMAP_SEARCH_URL = "https://www.onemap.gov.sg/api/common/elastic/search"

# Load ML Model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("model/traffic_xgb_model.json")


# Load road link data and precompute the mid_lat and mid_lon
road_links_df = pd.read_parquet("data/road_links.parquet").copy()

for col in ["start_lat", "start_lon", "end_lat", "end_lon"]:
    road_links_df[col] = road_links_df[col].astype(float)

road_links_df["mid_lat"] = (road_links_df["start_lat"] + road_links_df["end_lat"]) / 2.0
road_links_df["mid_lon"] = (road_links_df["start_lon"] + road_links_df["end_lon"]) / 2.0

# Load data for Speedbands, to be used as inputs for ML. ---------- 
road_category_dict = road_links_df.set_index("link_id")["road_category"].to_dict()
live_speedbands = defaultdict(list)

# To tidy up and split into separate files in the future if possible
# CLASSES
class HabitRouteIn(BaseModel):
    route_name: str | None = None
    from_label: str | None = None
    to_label: str | None = None
    coords_json: list[list[float]]
    distance_m: float | None = None
    link_ids: list[int] = [] 

class SavedPlaceIn(BaseModel):
    place_name: str
    label: str
    lat: float
    lon: float
    postal: str | None = None 

class RouteSettingsUpdate(BaseModel):
    alert_enabled: bool
    alert_start_time: str
    alert_end_time: str

# Poll LTA for live data and store data needed for inputs in dictionary
# Similar process used for when polling data for SQLite3 database for training
async def lightweight_poller():
    url = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"
    headers = {"AccountKey": LTA_API_KEY, "accept": "application/json"}

    while True:
        try:
            print("Fetching lightweight live speed bands...")
            skip = 0

            async with httpx.AsyncClient(timeout=10) as client:
                while True:
                    res = await client.get(f"{url}?$skip={skip}", headers=headers)

                    if res.status_code == 200:
                        data = res.json().get("value", [])

                        # Reached end 
                        if not data:
                            break

                        for item in data:
                            try:
                                lid = int(item["LinkID"])
                                sb = int(item["SpeedBand"])
                            except Exception:
                                continue

                            live_speedbands[lid].insert(0, sb)

                            if len(live_speedbands[lid]) > 4:
                                live_speedbands[lid].pop()

                        if len(data) < 500:
                            break

                        skip += 500
            
                    else:
                        print(f"Poll failed with status {res.status_code}")
                        break

            print(f"Updated cache for {len(live_speedbands)} roads.")


        except Exception as e:
            print(f"Lightweight poller error: {e}")

        await asyncio.sleep(300)

# Scheduler for habit_route alerts
# Scheduler for habit_route alerts
async def alert_scheduler():
    # Define Timezone locally to prevent hoisting issues
    local_sg_tz = timezone(timedelta(hours=8))
    
    # Use HTTPX so we don't block the FastAPI server!
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            try:
                now_str = datetime.now(local_sg_tz).strftime("%H:%M")
                print(f"--- [Scheduler] Heartbeat at {now_str} ---")
                
                headers = {"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}

                # Fetch active windows
                res = await client.get(f"{SUPABASE_URL}/rest/v1/habit_routes?alert_enabled=eq.true", headers=headers)
                if res.status_code != 200:
                    await asyncio.sleep(60)
                    continue
                
                for route in res.json():
                    # TIME WINDOW LOGIC
                    start = route.get("alert_start_time", "07:30")
                    end = route.get("alert_end_time", "09:00")
                    
                    in_window = False
                    if start <= end:
                        in_window = start <= now_str <= end
                    else: 
                        in_window = now_str >= start or now_str <= end
                    
                    if not in_window: 
                        continue

                    # CONGESTION THRESHOLD LOGIC
                    all_links = route.get("link_ids", [])
                    if not all_links:
                        continue
                        
                    # Find all links that are currently Band 1, 2, or 3
                    jammed_links = [lid for lid in all_links if live_speedbands.get(lid, [9])[0] < 4]
                    
                    # Force threshold to 0 for testing
                    threshold_limit = 0
                    
                    if len(jammed_links) >= threshold_limit:
                        
                        # --- SPAM PREVENTION LOGIC ---
                        time_limit = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")                        
                        check_url = f"{SUPABASE_URL}/rest/v1/traffic_alerts?route_id=eq.{route['id']}&created_at=gt.{time_limit}"
                        
                        check_res = await client.get(check_url, headers=headers)
                        print(f"DEBUG DB CHECK: Status {check_res.status_code} | Body: {check_res.text}")
                        
                        if check_res.status_code == 200:
                            alerts_found = check_res.json()
                            if len(alerts_found) == 0:
                                # ISSUE ALERT & LOG IT
                                print(f"FIRING NEW ALERT: {route['route_name']}!")
                                
                                log_body = {
                                    "user_id": route["user_id"],
                                    "route_id": route["id"],
                                    "affected_link_ids": jammed_links,
                                    "is_dismissed": False
                                }
                                post_res = await client.post(f"{SUPABASE_URL}/rest/v1/traffic_alerts", headers=headers, json=log_body)
                                print(f"DEBUG DB POST: Status {post_res.status_code} | Body: {post_res.text}")
                            else:
                                print(f"Cooldown active. Found {len(alerts_found)} recent alerts in DB.")
                        else:
                            print(f"DB Check Failed!")

            except Exception as e:
                print(f"Scheduler Error: {e}")

            # Sleep for 60 seconds
            await asyncio.sleep(60)

app = FastAPI()
print("### LOADED THIS APP.PY WITH /api/route ###")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # This allows the Node site to talk to you
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# On startup, start lightweight_poller to start collecting ML inputs
@app.on_event("startup")
async def startup_event():
    if LTA_API_KEY:
        asyncio.create_task(lightweight_poller())
        asyncio.create_task(alert_scheduler())


# # force numeric node coords
# for _, data in G.nodes(data=True):
#     if "x" in data:
#         data["x"] = float(data["x"])
#     if "y" in data:
#         data["y"] = float(data["y"])

# force numeric edge lengths
# for u, v, k, data in G.edges(keys=True, data=True):
#     if "length" in data:
#         data["length"] = float(data["length"])

# For now, create a dev bypass so we don't have to log in everytime
DEV_BYPASS_AUTH = os.getenv("DEV_BYPASS_AUTH", "0") == "1"

# Implement this for API endpoints we want to protect, 
# to ensure user is authenticated before being able to access our services
def require_user(authorization: str | None):

    # Dev bypass
    if DEV_BYPASS_AUTH:
        return {"id": "dev-user"}
    
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not Logged In")
    r = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": authorization,
            "apikey": SUPABASE_API_KEY
        },
        timeout=5,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return r.json()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/success")
def success():
    return FileResponse("static/index.html")

@app.get("/api/incidents")
def get_incidents(authorization: str | None = Header(default=None)):

    num_incidents = 30

    if not LTA_API_KEY:
        return load_placeholder_incidents(num_incidents, reason="missing_lta_api_key")


    user = require_user(authorization)
    url_incidents = "https://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents"
    headers = {
        "AccountKey": LTA_API_KEY,
        "accept": "application/json"
    }

    try:
        response = requests.get(url_incidents, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        raw_incidents = data.get("value", [])

        incidents = []
        for inc in raw_incidents:
            lat = inc.get("Latitude")
            lon = inc.get("Longitude")

            # Get closest link_id, to approximate affected road links. Store
            # additional info to display in backend
            matched_link_id, road_name = find_nearest_link_id(lat, lon)
            prediction_tp15 = "N/A"
            current_sb = "N/A"
            if matched_link_id:
                history = live_speedbands.get(matched_link_id, [])
                if history:
                    current_sb = history[0]
                prediction_tp15 = predict_for_link(matched_link_id)
            inc["matched_link_id"] = matched_link_id
            inc["matched_road_name"] = road_name
            inc["current_speed_band"] = current_sb 
            inc["predicted_speed_band_tp15"] = prediction_tp15
            incidents.append(inc)
        return {"incidents": incidents[:num_incidents], "user_id": user["id"]}
    except Exception:
        return load_placeholder_incidents(num_incidents, reason="lta_request_failed")

@app.post("/api/habit-routes")
def save_habit_route(
    payload: HabitRouteIn,
    authorization: str | None = Header(default=None)
):
    user = require_user(authorization)

    body = {
        "user_id": user["id"],
        "route_name": payload.route_name,
        "from_label": payload.from_label,
        "to_label": payload.to_label,
        "coords_json": payload.coords_json,
        "distance_m": payload.distance_m,
        "link_ids": payload.link_ids,     
    }

    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/habit_routes",
        headers=supabase_headers(authorization),
        json=body,
        timeout=10
    )

    if r.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Save failed: {r.text}")

    return {"saved": True}

# Habit routes get endpoint
@app.get("/api/habit-routes")
def get_habit_routes(authorization: str | None = Header(default=None)):
    user = require_user(authorization)

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/habit_routes",
        headers=supabase_headers(authorization),
        params={
            "user_id": f"eq.{user['id']}",
            "select": "*",
            "order": "created_at.desc"
        },
        timeout=10
    )

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Load failed: {r.text}")

    return {"routes": r.json()}

# Habit Routes update endpoint
@app.patch("/api/habit-routes/{route_id}")
def update_route_settings(
    route_id: int,
    payload: RouteSettingsUpdate,
    authorization: str | None = Header(default=None)
):
    user = require_user(authorization)

    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/habit_routes",
        headers = supabase_headers(authorization),
        params={
            "id": f"eq.{route_id}",
            "user_id": f"eq.{user['id']}"
        },
        json=payload.dict(),
        timeout=10
    )

    if r.status_code not in (200, 204):
        raise HTTPException(status_code=500, detail=f"Update failed: {r.text}")
    
    return {"updated": True}

# Habit Routes delete endpoint
@app.delete("/api/habit-routes/{route_id}")
def delete_habit_route(
    route_id: int,
    authorization: str | None = Header(default=None)
):
    user = require_user(authorization)

    r = requests.delete(
        f"{SUPABASE_URL}/rest/v1/habit_routes",
        headers=supabase_headers(authorization),
        params={
            "id": f"eq.{route_id}",
            "user_id": f"eq.{user['id']}"
        },
        timeout=10
    )

    if r.status_code not in (200, 204):
        raise HTTPException(status_code=500, detail=f"Delete failed: {r.text}")

    return {"deleted": True}

# Insert Saved Places to Database
@app.post("/api/saved-places")
def create_saved_place(payload: SavedPlaceIn, authorization: str | None = Header(default=None)):
    user = require_user(authorization)

    body = {
        "user_id": user["id"],
        "place_name": payload.place_name,
        "label": payload.label,
        "lat": payload.lat,
        "lon": payload.lon
    }
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/saved_places",
        headers=supabase_headers(authorization),
        json=body,
        timeout=10
    )
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Save failed: {r.text}")

    return {"saved": True}

# Retrieve Saved Places from Database
@app.get("/api/saved-places")
def get_saved_places(authorization: str | None = Header(default=None)):
    user = require_user(authorization)

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/saved_places",
        headers=supabase_headers(authorization),
        params={
            "user_id": f"eq.{user['id']}",
            "select": "*",
            "order": "created_at.desc"
        },
        timeout=10
    )

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Load failed: {r.text}")

    return {"places": r.json()}

@app.delete("/api/saved-places/{place_id}")
def delete_saved_place(
    place_id: int,
    authorization: str | None = Header(default=None)
):
    user = require_user(authorization)

    r = requests.delete(
        f"{SUPABASE_URL}/rest/v1/saved_places",
        headers=supabase_headers(authorization),
        params={
            "id": f"eq.{place_id}",
            "user_id": f"eq.{user['id']}"
        },
        timeout=10
    )

    if r.status_code not in (200, 204):
        raise HTTPException(status_code=500, detail=f"Delete failed: {r.text}")

    return {"deleted": True}

# If unable to access LTA API for some reason
def load_placeholder_incidents(num_incidents: int, reason: str):
    try:
        with open("static/placeholder_incidents.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            incidents = data
        else:
            incidents = data.get("value", [])
        return {
            "source": "placeholder",
            "incidents": incidents[:num_incidents],
            "reason": reason
        }
    except Exception as e:
        return {
            "incidents": [],
            "reason": "Failed to load.."
        }

# Routing function from OneMap API. To obtain the coordinates from street names or postal code.
@app.get("/api/geocode")
def geocode(q: str = Query(..., min_length=2)):
    r = requests.get(
        ONEMAP_SEARCH_URL,
        params={
            "searchVal": q,
            "returnGeom": "Y",
            "getAddrDetails": "Y",
            "pageNum": 1,
        },
        headers={
            "Authorization": ONEMAP_TOKEN
        },
        timeout = 5,
    )

    data = r.json()
    results = []
    for it in data.get("results", []):
        lat = it.get("LATITUDE")
        lon = it.get("LONGITUDE")
        if lat and lon:
            results.append({
                "label": it.get("ADDRESS") or it.get("SEARCHVAL"),
                "postal": it.get("POSTAL"),
                "lat": float(lat),
                "lon": float(lon),
            })

    return JSONResponse({"results": results[:8]})

# Perform actual routing logic using A*
@app.get("/api/route")
def api_route(fromLat: float, fromLon: float, toLat: float, toLon: float):
    try:
        # Calculate bounding box with a small padding
        padding = 0.02
        s = min(fromLat, toLat) - padding
        n = max(fromLat, toLat) + padding
        w = min(fromLon, toLon) - padding
        e = max(fromLon, toLon) + padding

        # The exact Overpass query used in Node server
        overpass_query = f"""
        [out:json][timeout:25];
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link)$"]({s},{w},{n},{e});
        );
        out body geom;
        """
        
        # Fetch the roads
        res = requests.post("https://overpass-api.de/api/interpreter", data={"data": overpass_query}, timeout=30)
        roads_json = res.json()

        print("2. Feeding map to teammate's A* Engine...")
        payload = {
            "roads": roads_json,
            "start": {"lat": fromLat, "lon": fromLon},
            "end": {"lat": toLat, "lon": toLon},
            "signalPoints": [] # LTA traffic light 
        }

        # Run the routing
        py_result = plan_routes(payload)
        
        if not py_result.get("routes"):
            return {"routes": []}

        print("3. Mapping Overpass routes to LTA Links...")
        routes_out = []
        for r in py_result["routes"]:
            coords = r["coords"]
            
            match_info = match_route_to_lta_links(coords)
            
            routes_out.append({
                "id": r.get("id", "custom"),
                "label": r.get("label", "Route"),
                "coords": coords,
                "match_info": match_info, 
                "estMinutes": r.get("estMinutes", 0),
                "totalDist": r.get("totalDist", 0)
            })

        return {"routes": routes_out}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    

# HELPER FUNCTION, APPROXIMATE DISTANCE IN METERS
def approx_meters(lat1, lon1, lat2, lon2):
    """
    Fast local distance approximation in meters, good enough for Singapore-scale matching.
    """
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111320.0 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110540.0
    return math.sqrt(dx * dx + dy * dy)

# Bearing deg to check the route direction
def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Bearing of a segment in degrees 0..360.
    """
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * math.cos(mean_lat)
    dy = (lat2 - lat1)
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0

# Calculate absolute smallest angle difference to compare if roads are 
# in the same direction
def bearing_diff_deg(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

# For the generated route, approximate routes that exists in LTA road_links
# Because the system is only able to retrieve ML input data from LTA road links
# it approximates based on distance, and segment direction
def match_route_to_lta_links(route_coords, max_midpoint_dist_m=35.0, max_bearing_diff_deg=35.0):

    matched = []
    segment_matches = []
    total_route_len_m = 0.0
    covered_len_m = 0.0

    # Loop through road segments
    # Break each road segment into point A -> point B
    for i in range(len(route_coords) - 1):
        lat1, lon1 = route_coords[i]
        lat2, lon2 = route_coords[i + 1]

        # Calculate segment length
        seg_len_m = approx_meters(lat1, lon1, lat2, lon2)
        # If segment is too short, skip it.. (to be tuned)
        if seg_len_m < 5:
            segment_matches.append(None)
            continue

        total_route_len_m += seg_len_m

        seg_mid_lat = (lat1 + lat2) / 2.0
        seg_mid_lon = (lon1 + lon2) / 2.0
        seg_bearing = bearing_deg(lat1, lon1, lat2, lon2)

        # cheap bounding-box
        lat_pad = 0.0005  
        lon_pad = 0.0005   

        # Filter only road link within bounding box to optimize the search
        candidates = road_links_df[
            (road_links_df["mid_lat"].between(seg_mid_lat - lat_pad, seg_mid_lat + lat_pad)) &
            (road_links_df["mid_lon"].between(seg_mid_lon - lon_pad, seg_mid_lon + lon_pad))
        ].copy()

        if candidates.empty:
            segment_matches.append(None)
            continue

        # score candidates by midpoint distance + bearing similarity
        best = None
        best_score = None

        # Evaluate each candidate
        # Check the candidate's midpoint against route segment's midpoint
        for _, row in candidates.iterrows():
            dist_m = approx_meters(seg_mid_lat, seg_mid_lon, row["mid_lat"], row["mid_lon"])
            if dist_m > max_midpoint_dist_m:
                continue

            link_bearing = bearing_deg(
                row["start_lat"], row["start_lon"],
                row["end_lat"], row["end_lon"]
            )

            # allow either direction because LTA link direction may differ from route drawing direction
            diff1 = bearing_diff_deg(seg_bearing, link_bearing)
            diff2 = bearing_diff_deg(seg_bearing, (link_bearing + 180.0) % 360.0)
            bdiff = min(diff1, diff2)

            if bdiff > max_bearing_diff_deg:
                continue

            score = dist_m + 0.7 * bdiff

            if best_score is None or score < best_score:
                best_score = score

                current_sb = live_speedbands.get(int(row["link_id"]), [None])[0]
                best = {
                    "link_id": int(row["link_id"]),
                    "road_name": str(row["road_name"]) if pd.notna(row["road_name"]) else None,
                    "road_category": int(row["road_category"]) if pd.notna(row["road_category"]) else None,
                    "pred_band": predict_for_link(int(row["link_id"])),
                    "current_band": int(current_sb) if current_sb is not None else None,
                    "dist_m": round(dist_m, 2),
                    "bearing_diff_deg": round(bdiff, 2),
                    "segment_len_m": round(seg_len_m, 2)
                }

        if best is not None:
            covered_len_m += seg_len_m
            matched.append(best)
            segment_matches.append(best)
        else:
            segment_matches.append(None)

    # deduplicate consecutive repeats
    deduped = []
    for item in matched:
        if not deduped or deduped[-1]["link_id"] != item["link_id"]:
            deduped.append(item)
        else:
            deduped[-1]["segment_len_m"] += item["segment_len_m"]

    coverage_ratio = covered_len_m / total_route_len_m if total_route_len_m > 0 else 0.0

    return {
        "matched_links": deduped,
        "segment_matches": segment_matches,
        "total_route_len_m": round(total_route_len_m, 1),
        "covered_len_m": round(covered_len_m, 1),
        "coverage_ratio": round(coverage_ratio, 3)
    }

# Helper function to get input data for road links
SG_TZ = timezone(timedelta(hours=8))

road_spatial_dict = road_links_df.set_index("link_id")[["mid_lat", "mid_lon"]].to_dict(orient="index")
def predict_for_link(link_id: int):
    vals = live_speedbands.get(link_id, [])

    # need 4 readings: t0, t-5, t-10, t-15
    # If there is at least t0, use it as placeholder
    if len(vals) == 0:
        return None
    
    # If app just started, t-x returns null. Map it to current_sb as placeholder
    current_sb = vals[0]
    sb_tm5 = vals[1] if len(vals) > 1 else current_sb
    sb_tm10 = vals[2] if len(vals) > 2 else current_sb
    sb_tm15 = vals[3] if len(vals) > 3 else current_sb

    road_cat = road_category_dict.get(link_id)
    if road_cat is None:
        return None

    now = datetime.now(SG_TZ)
    hour = now.hour
    dow = now.weekday()

    sp = road_spatial_dict.get(link_id)
    if sp is None:
        return None
    
    X = pd.DataFrame([{
        "sb": current_sb,
        "sb_tm5": sb_tm5,
        "sb_tm10": sb_tm10,
        "sb_tm15": sb_tm15,
        "road_category": int(road_cat),
        "dow_sg": dow,
        "hour_sg": hour,
        "mid_lat": sp["mid_lat"],
        "mid_lon": sp["mid_lon"]
    }])

    X = X[["sb", "sb_tm5", "sb_tm10", "sb_tm15", "road_category", "dow_sg", "hour_sg", "mid_lat", "mid_lon"]]

    pred = xgb_model.predict(X)[0]
    return int(pred) + 1

# For Incidents Mapping to Road_link
def find_nearest_link_id(inc_lat, inc_lon):
    # Search within ~110m box
    pad = 0.001 
    candidates = road_links_df[
        (road_links_df["mid_lat"].between(inc_lat - pad, inc_lat + pad)) &
        (road_links_df["mid_lon"].between(inc_lon - pad, inc_lon + pad))
    ]
    
    if candidates.empty:
        return None, "Ummapped Road"
        
    # Pick the absolute closest by distance
    best_id = None
    min_dist = 9999
    for _, row in candidates.iterrows():
        d = approx_meters(inc_lat, inc_lon, row["mid_lat"], row["mid_lon"])
        if d < min_dist:
            min_dist = d
            best_id = int(row["link_id"])
            best_name = str(row["road_name"]) if pd.notna(row["road_name"]) else "LTA Road"
    return best_id, best_name

# Helper to call Supabase REST table
def supabase_headers(user_jwt: str):
    return {
        "apikey": SUPABASE_API_KEY,
        "Authorization": user_jwt,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

# This receives coords and recalculates the match-info
@app.post("/api/habit-routes/analyze")
def analyze_habit_route(payload: dict[str, Any],
    authorization: str | None = Header(default=None)
):
    require_user(authorization)

    coords = payload.get("coords_json")
    if not coords or not isinstance(coords, list) or len(coords) < 2:
        raise HTTPException(status_code=400, detail="coords_json must contain 2 coordinates")
    try:
        match_info = match_route_to_lta_links(coords)
        return {
            "coords": coords,
            "match_info": match_info
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error")
    
# Alerts endpoint
@app.get("/api/my-alerts")
def get_my_alerts(authorization: str | None = Header(default=None)):
    user = require_user(authorization)
    
    # Fetch alerts that belong to this user and haven't been dismissed
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/traffic_alerts?user_id=eq.{user['id']}&is_dismissed=eq.false&order=created_at.desc",
        headers=supabase_headers(authorization)
    )
    
    if r.status_code == 200:
        return r.json()
    return []