#!/usr/bin/env python3
"""
Python 计算引擎（供 Node.js 后端子进程调用）

本文件目前提供两个计算能力：
1) enrich_incidents_with_cameras：
   对事故点匹配最近实时摄像头，并补齐事故影响字段（扩散半径/持续时长）。
2) plan_routes：
   基于 OSM 路网做 A* 寻路，输出 3 条策略路线（时间优先/少红绿灯/均衡）。

调用方式：
- Node.js 会以 `python3 compute_engine.py --op <op_name>` 方式启动此脚本。
- 通过 stdin 输入 JSON payload，通过 stdout 输出 JSON 结果。
"""

import argparse
import heapq
import json
import math
import re
import sys
from typing import Dict, List


# -------------------- 通用地理计算 --------------------
def haversine(lat1, lon1, lat2, lon2):
    """计算两点球面距离（单位：米）。"""
    r = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def to_float(v):
    """将输入安全转为有限浮点数；失败返回 None。"""
    try:
        n = float(v)
        if math.isfinite(n):
            return n
    except Exception:
        return None
    return None


# -------------------- 事故影响估算 --------------------
def infer_impact_by_type(type_text, message=""):
    """
    根据事故类型/文案关键词给出经验估算。

    返回字段：
    - spreadRadiusKm：预计影响扩散半径（公里）
    - minMin/maxMin：预计持续时间区间（分钟）
    """
    t = f"{type_text or ''} {message or ''}".lower()
    if re.search(r"(accident|collision|crash|fire|fatal)", t):
        return {"spreadRadiusKm": 2.2, "minMin": 50, "maxMin": 110}
    if re.search(r"(roadwork|construction|road works|works)", t):
        return {"spreadRadiusKm": 1.5, "minMin": 45, "maxMin": 95}
    if re.search(r"(breakdown|stalled|vehicle breakdown)", t):
        return {"spreadRadiusKm": 1.2, "minMin": 25, "maxMin": 60}
    if re.search(r"(heavy traffic|congestion|jam)", t):
        return {"spreadRadiusKm": 1.0, "minMin": 20, "maxMin": 45}
    return {"spreadRadiusKm": 0.9, "minMin": 15, "maxMin": 35}


def build_impact_meta(raw):
    """
    合并“上游已给值”和“经验估算值”。

    优先级：
    - 若事故自身带 estimatedDurationMin/Max、spreadRadiusKm，则优先用它
    - 否则使用 infer_impact_by_type 的经验值
    """
    inferred = infer_impact_by_type(raw.get("type"), raw.get("message", ""))
    lta_min = to_float(raw.get("estimatedDurationMin"))
    lta_max = to_float(raw.get("estimatedDurationMax"))
    radius = to_float(raw.get("spreadRadiusKm"))

    min_min = lta_min if lta_min is not None else inferred["minMin"]
    max_min = lta_max if lta_max is not None else inferred["maxMin"]
    if max_min < min_min:
        min_min, max_min = max_min, min_min

    return {
        "spreadRadiusKm": round(radius if radius is not None else inferred["spreadRadiusKm"], 1),
        "estimatedDurationMin": max(1, int(round(min_min))),
        "estimatedDurationMax": max(int(round(min_min)), int(round(max_min))),
    }


def derive_incident_area(message, lat, lon):
    """
    尝试从事故描述中提取区域名。

    例如：
    - "PIE - accident near ..." -> 提取 "PIE"
    提取失败时回退为坐标字符串。
    """
    msg = str(message or "").strip()
    if msg:
        parts = [x.strip() for x in re.split(r"\s-\s|,|;", msg) if x.strip()]
        if parts:
            return parts[0]
    if lat is None or lon is None:
        return "(unknown)"
    return f"({lat:.4f}, {lon:.4f})"


# -------------------- 事故与摄像头匹配 --------------------
def enrich_incidents_with_cameras(payload):
    """
    输入 incidents + cameras，输出匹配后的事故列表。

    关键规则：
    - 每条事故找最近实时摄像头
    - 最近距离 超过两公里 视为无可用摄像头
    - 为每条事故补齐 area / spread / duration 等字段
    """
    incidents = payload.get("incidents") or []
    cameras = payload.get("cameras") or []
    output = []

    for inc in incidents:
        inc_lat = to_float(inc.get("lat"))
        inc_lon = to_float(inc.get("lon"))

        nearest = None
        best_dist = float("inf")

        # 仅在事故坐标有效时进行最近点搜索
        if inc_lat is not None and inc_lon is not None:
            for cam in cameras:
                c_lat = to_float(cam.get("Latitude"))
                c_lon = to_float(cam.get("Longitude"))
                if c_lat is None or c_lon is None:
                    continue
                d = haversine(inc_lat, inc_lon, c_lat, c_lon)
                if d < best_dist:
                    best_dist = d
                    nearest = cam

        # 超过阈值则视为无摄像头证据
        if best_dist > 2000:
            nearest = None

        impact = build_impact_meta(inc)

        output.append({
            "id": inc.get("id"),
            "type": inc.get("type"),
            "message": inc.get("message"),
            "area": derive_incident_area(inc.get("message"), inc_lat, inc_lon),
            "lat": inc_lat,
            "lon": inc_lon,
            "createdAt": inc.get("createdAt"),
            "spreadRadiusKm": inc.get("spreadRadiusKm") if inc.get("spreadRadiusKm") is not None else impact["spreadRadiusKm"],
            "estimatedDurationMin": inc.get("estimatedDurationMin") if inc.get("estimatedDurationMin") is not None else impact["estimatedDurationMin"],
            "estimatedDurationMax": inc.get("estimatedDurationMax") if inc.get("estimatedDurationMax") is not None else impact["estimatedDurationMax"],
            "imageLink": nearest.get("ImageLink") if nearest else None,
            "cameraName": nearest.get("Name") if nearest else None,
            "cameraDistanceMeters": int(round(best_dist)) if nearest and math.isfinite(best_dist) else None,
        })

    return {"value": output}


# -------------------- 路网构图与 A* --------------------
def node_key(lat, lon):
    """节点归一化 key：保留 4 位小数，约 10m 级别合并。"""
    return f"{round(lat, 4)},{round(lon, 4)}"


def build_graph(roads):
    """
    从 Overpass 返回的 roads.elements 构建图结构。

    图结构说明：
    - 节点：{key, lat, lon, edges, degree}
    - 边：{to, weight}，其中 weight 为“小时”
    """
    nodes: Dict[str, Dict] = {}

    def ensure(lat, lon):
        k = node_key(lat, lon)
        if k not in nodes:
            nodes[k] = {"key": k, "lat": lat, "lon": lon, "edges": [], "degree": 0}
        return nodes[k]

    for el in (roads or {}).get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue

        for i in range(len(geom) - 1):
            a = geom[i]
            b = geom[i + 1]
            a_lat = to_float(a.get("lat"))
            a_lon = to_float(a.get("lon"))
            b_lat = to_float(b.get("lat"))
            b_lon = to_float(b.get("lon"))
            if None in (a_lat, a_lon, b_lat, b_lon):
                continue

            n1 = ensure(a_lat, a_lon)
            n2 = ensure(b_lat, b_lon)

            dist_m = haversine(a_lat, a_lon, b_lat, b_lon)
            if dist_m < 2:
                continue

            # 假设平均速度 40km/h，权重单位为“小时”
            base_hours = (dist_m / 1000.0) / 40.0

            # 双向建边
            n1["edges"].append({"to": n2["key"], "weight": base_hours})
            n2["edges"].append({"to": n1["key"], "weight": base_hours})

            # 度数用于路口判断（度数>=3 通常视为路口）
            n1["degree"] += 1
            n2["degree"] += 1

    return nodes


def nearest_node(nodes, lat, lon):
    """在图中找距离给定坐标最近的节点，限制 600 米内。"""
    best_key = None
    best_dist = float("inf")
    for k, n in nodes.items():
        d = haversine(lat, lon, n["lat"], n["lon"])
        if d < best_dist and d < 600:
            best_dist = d
            best_key = k
    return best_key


def edge_key(a, b):
    """无向边标准化 key，用于去重与复用惩罚计算。"""
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def reconstruct_path(prev, end_key):
    """根据 prev 映射回溯路径。"""
    out = []
    cur = end_key
    while cur is not None:
        out.append(cur)
        cur = prev.get(cur)
    out.reverse()
    return out


def a_star(nodes, start_key, end_key, cost_fn):
    """
    A* 主过程。

    - g：起点到当前点的已知最小代价
    - h：当前点到终点的启发式估计（直线距离/50kmh）
    - f = g + h
    """
    g = {start_key: 0.0}
    prev = {start_key: None}
    open_heap = [(0.0, start_key)]  # 小根堆，存 (f_score, node_key)
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == end_key:
            break

        closed.add(current)
        cur_node = nodes[current]

        for edge in cur_node["edges"]:
            to_key = edge["to"]
            if to_key in closed:
                continue

            tentative = g[current] + cost_fn(edge, cur_node, nodes[to_key])
            if tentative < g.get(to_key, float("inf")):
                prev[to_key] = current
                g[to_key] = tentative
                h = haversine(nodes[to_key]["lat"], nodes[to_key]["lon"], nodes[end_key]["lat"], nodes[end_key]["lon"]) / 1000.0 / 50.0
                heapq.heappush(open_heap, (tentative + h, to_key))

    if end_key not in prev:
        return []
    return reconstruct_path(prev, end_key)


# -------------------- 红绿灯统计与路线指标 --------------------
def distance_to_route(route_coords, lat, lon):
    """计算点到路线折线点集的最小距离（简化为点到顶点最小值）。"""
    best = float("inf")
    for c in route_coords:
        d = haversine(lat, lon, c[0], c[1])
        if d < best:
            best = d
    return best


def count_lights_by_signals(route_coords, signal_points, match_radius_m=35, dedupe_radius_m=65):
    """
    用真实信号点位统计红绿灯数量。

    步骤：
    1) 先找“离路线足够近”的信号点
    2) 再做半径聚类去重，避免一个路口被多次计数
    """
    if len(route_coords) < 2 or not signal_points:
        return 0

    hits = []
    for sig in signal_points:
        s_lat = to_float(sig.get("lat"))
        s_lon = to_float(sig.get("lon"))
        if s_lat is None or s_lon is None:
            continue
        if distance_to_route(route_coords, s_lat, s_lon) <= match_radius_m:
            hits.append({"lat": s_lat, "lon": s_lon, "count": 1})

    if not hits:
        return 0

    clusters = []
    for sig in hits:
        merged = False
        for c in clusters:
            if haversine(sig["lat"], sig["lon"], c["lat"], c["lon"]) <= dedupe_radius_m:
                c["count"] += 1
                c["lat"] = (c["lat"] * (c["count"] - 1) + sig["lat"]) / c["count"]
                c["lon"] = (c["lon"] * (c["count"] - 1) + sig["lon"]) / c["count"]
                merged = True
                break
        if not merged:
            clusters.append(sig)

    return len(clusters)


def count_lights_by_degree(path_keys, nodes):
    """当真实信号点不足时，用“节点度数>=3”估算红绿灯。"""
    if len(path_keys) < 3:
        return 0
    cnt = 0
    for i in range(1, len(path_keys) - 1):
        if (nodes[path_keys[i]].get("degree") or 0) >= 3:
            cnt += 1
    return cnt


def calc_path_distance(path_keys, nodes, start, end):
    """计算完整路径总长度（米），包含起点接入与终点接出。"""
    total = 0.0
    prev_lat = start["lat"]
    prev_lon = start["lon"]
    for k in path_keys:
        n = nodes[k]
        total += haversine(prev_lat, prev_lon, n["lat"], n["lon"])
        prev_lat = n["lat"]
        prev_lon = n["lon"]
    total += haversine(prev_lat, prev_lon, end["lat"], end["lon"])
    return total


def get_route_coords(path_keys, nodes, start, end):
    """把路径节点序列转为前端可直接绘制的坐标数组。"""
    coords = [[start["lat"], start["lon"]]]
    for k in path_keys:
        n = nodes[k]
        coords.append([n["lat"], n["lon"]])
    coords.append([end["lat"], end["lon"]])
    return coords


# -------------------- 路线规划主入口 --------------------
def plan_routes(payload):
    """
    路线规划主函数。

    输入：
    - roads: Overpass 道路数据
    - start/end: 起终点坐标
    - signalPoints: 真实信号点位

    输出：
    - routes: 3 条策略路线（若可达）
    """
    roads = payload.get("roads") or {}
    start = payload.get("start") or {}
    end = payload.get("end") or {}
    signal_points = payload.get("signalPoints") or []

    start_lat = to_float(start.get("lat"))
    start_lon = to_float(start.get("lon"))
    end_lat = to_float(end.get("lat"))
    end_lon = to_float(end.get("lon"))
    if None in (start_lat, start_lon, end_lat, end_lon):
        return {"routes": []}

    start = {"lat": start_lat, "lon": start_lon}
    end = {"lat": end_lat, "lon": end_lon}

    nodes = build_graph(roads)
    if not nodes:
        return {"routes": []}

    start_key = nearest_node(nodes, start_lat, start_lon)
    end_key = nearest_node(nodes, end_lat, end_lon)
    if not start_key or not end_key:
        return {"routes": []}

    # 三个策略与前端旧版逻辑保持一致
    modes = [
        {"id": "fastest", "label": "FASTEST", "color": "#2563eb", "desc": "Prioritize total time"},
        {"id": "fewerLights", "label": "FEWER LIGHTS", "color": "#16a34a", "desc": "Reduce intersection waiting"},
        {"id": "balanced", "label": "BALANCED", "color": "#ea580c", "desc": "Near-fastest with fewer lights"},
    ]

    plans = []
    used_edge_sets: List[set] = []

    for mode in modes:

        # 不同策略使用不同代价函数，但都基于同一张图和同一 A*
        def cost_fn(edge, from_node, to_node):
            base = edge["weight"]
            intersection_cost = (15 / 3600.0) if (to_node.get("degree") or 0) >= 3 else 0.0

            # 若该边已被前一条路线使用，增加少量复用惩罚，提高路线差异性
            ep = edge_key(from_node["key"], to_node["key"])
            reuse_penalty = 0.025 if any(ep in s for s in used_edge_sets) else 0.0

            if mode["id"] == "fastest":
                return base + reuse_penalty
            if mode["id"] == "fewerLights":
                return base + intersection_cost * 1.8 + reuse_penalty
            return base + intersection_cost * 0.9 + reuse_penalty

        path_keys = a_star(nodes, start_key, end_key, cost_fn)
        if len(path_keys) < 2:
            continue

        # 生成签名，去掉完全重复的路线
        edge_set = set()
        for i in range(len(path_keys) - 1):
            edge_set.add(edge_key(path_keys[i], path_keys[i + 1]))
        signature = ",".join(sorted(edge_set))
        if any(p.get("signature") == signature for p in plans):
            continue

        total_dist = calc_path_distance(path_keys, nodes, start, end)
        est_minutes = (total_dist / 1000.0 / 40.0) * 60.0
        coords = get_route_coords(path_keys, nodes, start, end)

        # 优先用真实信号点统计红绿灯；无命中再用路口度数估算
        signal_lights = count_lights_by_signals(coords, signal_points, 35, 65)
        traffic_lights = signal_lights if signal_lights > 0 else count_lights_by_degree(path_keys, nodes)

        plans.append({
            "id": mode["id"],
            "label": mode["label"],
            "color": mode["color"],
            "desc": mode["desc"],
            "totalDist": total_dist,
            "estMinutes": est_minutes,
            "trafficLights": traffic_lights,
            "coords": coords,
            "signature": signature,
        })

        used_edge_sets.append(edge_set)

    # 返回前按基础 ETA 升序
    plans.sort(key=lambda x: x.get("estMinutes", float("inf")))
    return {"routes": plans}


# -------------------- CLI 入口 --------------------
def main():
    """命令行入口：读取 stdin JSON，根据 --op 执行并输出 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    args = parser.parse_args()

    raw = sys.stdin.read() or "{}"
    payload = json.loads(raw)

    if args.op == "enrich_incidents_with_cameras":
        result = enrich_incidents_with_cameras(payload)
    elif args.op == "plan_routes":
        result = plan_routes(payload)
    else:
        raise RuntimeError(f"Unsupported op: {args.op}")

    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 统一把错误写到 stderr，便于 Node.js 侧读取 details
        sys.stderr.write(str(exc))
        sys.exit(1)
