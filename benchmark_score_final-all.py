
from __future__ import annotations
import json
import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
import re
import os
import numpy as np
import csv
from datetime import datetime

# Utility Functions: Parsing Structure and Numerical Values

def _norm_str(s: str) -> str:
    return (s or "").strip().lower()

# Extract Attributes
def _extract_attr_value(attributes: List[List[Any]], key: str) -> Optional[str]:

    for item in attributes or []:
        if len(item) >= 2:
            k, v = str(item[0]).lower(), str(item[1])
            if k == key.lower():
                return str(v)
    return None

# Extract Quantity
def _extract_quantity(
    numbers: List[List[Any]], target_key: str, unit: Optional[str] = None
) -> List[float]:

    vals = []
    for q in numbers or []:
        if not q:
            continue
        key = str(q[0]).lower()
        if key == target_key.lower():
            if unit is None:
                if len(q) >= 2 and isinstance(q[1], (int, float)):
                    vals.append(float(q[1]))
            else:
                if (
                    len(q) >= 3
                    and str(q[2]).lower() == unit.lower()
                    and isinstance(q[1], (int, float))
                ):
                    vals.append(float(q[1]))
    return vals

# Extract Trajectory and Destination
def _extract_traj_from_item(item: Dict[str, Any]):

    qs = item.get("quantities", []) or []
    traj = []

    for ent in qs:
        if not (isinstance(ent, list) and len(ent) >= 2):
            continue

        val = ent[1]
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            try:
                x = float(val[0])
                y = float(val[1])
                z = float(val[2])
                traj.append((x, y, z))
            except Exception:
                continue

    qtype = str(item.get("question_type", item.get("qid", ""))).upper()

    if qtype in {"Q15", "Q16"}:
        traj = traj[:6]
    elif qtype in {"Q17", "Q18", "Q19"}:
        traj = traj[:1]
    else:
        return []

    # Add Timeline
    return [(x, y, z, (i + 1)* 0.5) for i, (x, y, z) in enumerate(traj)]


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def _find_q12_risk(gt_list: list[dict]) -> str:
    for g in gt_list or []:
        if str(g.get("question_type", "")).upper() == "Q12":
            val = _extract_attr_value(g.get("attributes", []), "risk_level")
            if val:
                return val
    return "unknown"


# 1. Scene Understanding


# q12/q14: risk level
def spice_component_risk_level(
    gt_item: Dict[str, Any], pred_item: Dict[str, Any]
) -> int:
    gt = _extract_attr_value(gt_item.get("attributes", []), "risk_level")
    pr = _extract_attr_value(pred_item.get("attributes", []), "risk_level")
    return int(gt is not None and pr is not None and _norm_str(gt) == _norm_str(pr))

# q13: interaction
def spice_component_interaction(
    gt_item: Dict[str, Any], pred_item: Dict[str, Any]
) -> int:
    gt_vals = _extract_quantity(gt_item.get("quantities", []), "interaction")
    pr_vals = _extract_quantity(pred_item.get("quantities", []), "interaction")
    return int(
        len(gt_vals) == len(pr_vals)
        and all(abs(a - b) < 1e-6 for a, b in zip(gt_vals, pr_vals))
    )

# q14: Coverage of all objects and attributes
def spice_component_objects_attrs(
    gt_item: Dict[str, Any], pred_item: Dict[str, Any]
) -> float:

    gt_objs = set([_norm_str(x) for x in gt_item.get("objects", [])])
    pr_objs = set([_norm_str(x) for x in pred_item.get("objects", [])])

    gt_attrs = set()
    for a in gt_item.get("attributes", []) or []:
        if len(a) >= 2:
            gt_attrs.add((_norm_str(str(a[0])), _norm_str(str(a[1]))))

    pr_attrs = set()
    for a in pred_item.get("attributes", []) or []:
        if len(a) >= 2:
            pr_attrs.add((_norm_str(str(a[0])), _norm_str(str(a[1]))))

    # truth set size
    truth = len(gt_objs) + len(gt_attrs)
    if truth == 0:
        return 1.0

    match = len(gt_objs & pr_objs) + len(gt_attrs & pr_attrs)
    return match / truth

# SPICE = 0.1*SPICE1 + 0.1*SPICE2 + 0.8*SPICE3
def spice_score(gt_item: Dict[str, Any], pred_item: Dict[str, Any]) -> float:

    s1 = spice_component_risk_level(gt_item, pred_item)  # SPICE1
    s2 = spice_component_interaction(gt_item, pred_item)  # SPICE2
    s3 = spice_component_objects_attrs(gt_item, pred_item)  # SPICE3
    return 0.1 * s1 + 0.1 * s2 + 0.8 * s3


def risk_class_acc(gt_item: Dict[str, Any], pred_item: Dict[str, Any]) -> float:

    return spice_component_risk_level(gt_item, pred_item)

# GPT call log file


def gpt_score_api(question, gt_text, pred_text, spice, risk_acc, gt_risk):

    if spice == 'N/A':
        spice_value = 'N/A' 
    else:
        spice_value = float(spice) 

    prompt = f"""
We are evaluating how semantically aligned the model's prediction is with the reference answer.

Below are the details of this sample:
- The question being asked: {question}
- The reference (ground truth) answer: {gt_text}
- The model's predicted answer: {pred_text}
Additional numeric evaluation signals (use only as weak hints):
- SPICE Score: {spice_value}——(0–1, higher = more similar object/attribute semantics)
- Risk Classification Accuracy: {risk_acc}—— (0 or 1: 1 means risk_level matches; 0 means mismatch)
- Ground Truth Risk Level (if available): {gt_risk}

Your task:
Provide a final semantic alignment score between 0 and 1, based primarily on the **meaning** of the reference answer and the prediction.  
Use the numeric signals ONLY as supplementary hints, not as the main criterion.

Scoring:
- 1.0 = semantically identical
- 0.0 = unrelated or contradictory

Return **only** a single numeric score.
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a semantic evaluation assistant that rates semantic alignment based on provided metrics and text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        content = f"0.0 (error: {e})"

    try:
        score = float(content)
    except ValueError:
        nums = re.findall(r"\d*\.?\d+", content)
        score = float(nums[0]) if nums else 0.0

    #Log output simplified to: complete prompt + raw GPT output + parsed results
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "gpt_output_raw": content,
        "parsed_score": score
    }

    try:
        with open(GPT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

    return score



# 2. Perception

# Combining counts and category comparisons for Q4/Q5/Q7/Q11
def per_class_accuracy(
    gt_item: Dict[str, Any], pred_item: Dict[str, Any]
) -> Dict[str, float]:

    CLASSES = ["vehicle", "truck", "pedestrian", "bicycle", "motorcycle"]
    weights = {
        "vehicle": 0.2,
        "truck": 0.2,
        "pedestrian": 0.2,
        "bicycle": 0.2,
        "motorcycle": 0.2,
    }

    gt_classes = [
        _norm_str(c) for c in gt_item.get("objects", []) if _norm_str(c) in CLASSES
    ]
    pr_classes = [
        _norm_str(c) for c in pred_item.get("objects", []) if _norm_str(c) in CLASSES
    ]
    CLASSES_q7 = ["pedestrian", "bicycle"]

    for cls in CLASSES_q7:
        gt_vals = _extract_quantity(gt_item.get("quantities", []), cls)  # List[float]
        pr_vals = _extract_quantity(pred_item.get("quantities", []), cls)  # List[float]

        cnt_gt = len(gt_vals)

        hits = 0
        used = [False] * len(pr_vals)
        for gv in gt_vals:
            for j, pv in enumerate(pr_vals):
                if not used[j] and abs(float(pv) - float(gv)) <= 1e-6:
                    used[j] = True
                    hits += 1
                    break

        if cnt_gt > 0:
            gt_classes += [cls] * cnt_gt
            pr_classes += [cls] * hits

    gt_counter = Counter(gt_classes)
    pr_counter = Counter(pr_classes)

    per_acc = {}
    for c in CLASSES:
        Nc = gt_counter[c]
        TPc = min(Nc, pr_counter[c])  
        per_acc[c] = (
            _safe_div(TPc, Nc) if Nc > 0 else (1.0 if pr_counter[c] == 0 else 0.0)
        )

    weighted = sum(per_acc[c] * weights[c] for c in CLASSES)
    return {"per_class": per_acc, "weighted": weighted}


def _mean_relative_error(pairs: List[Tuple[float, float]]) -> float:
    errs = []
    for gt, pr in pairs:
        if gt == 0:
            continue
        errs.append(abs(pr - gt) / abs(gt))
    return sum(errs) / len(errs) if errs else 0.0

# Average relative error of distance to vehicle (in meters)
def emrde(gt_items: List[Dict[str, Any]], pred_items: List[Dict[str, Any]]) -> float:

    pairs = []
    for g, p in zip(gt_items, pred_items):
        # The distance is calculated based on the target class in the objects list; if there are multiple targets, each class is matched sequentially.
        gq = g.get("quantities", [])
        pq = p.get("quantities", [])
        for cls in set(
            [q[0] for q in gq if len(q) >= 3 and _norm_str(str(q[2])) == "meters"]
        ):
            gt_vals = _extract_quantity(gq, cls, unit="meters")
            pr_vals = _extract_quantity(pq, cls, unit="meters")
            for i, v in enumerate(gt_vals):
                if i < len(pr_vals):
                    pairs.append((v, pr_vals[i]))
    return _mean_relative_error(pairs)

# Average relative error of the vehicle angle (in degrees)
def emrae(gt_items: List[Dict[str, Any]], pred_items: List[Dict[str, Any]]) -> float:

    pairs = []
    for g, p in zip(gt_items, pred_items):
        gq = g.get("quantities", [])
        pq = p.get("quantities", [])
        for cls in set(
            [q[0] for q in gq if len(q) >= 3 and _norm_str(str(q[2])) == "degrees"]
        ):
            gt_vals = _extract_quantity(gq, cls, unit="degrees")
            pr_vals = _extract_quantity(pq, cls, unit="degrees")
            for i, v in enumerate(gt_vals):
                if i < len(pr_vals):
                    pairs.append((v, pr_vals[i]))
    return _mean_relative_error(pairs)

# Relative error of object-to-object distance (in meters)
def omrde(gt_pairs: List[Dict[str, Any]], pred_pairs: List[Dict[str, Any]]) -> float:

    pairs = []
    for g, p in zip(gt_pairs, pred_pairs):
        gq = g.get("quantities", [])
        pq = p.get("quantities", [])
        # Extract GT and all meters values ​​in the prediction
        gvals = [float(q[1]) for q in gq if len(q) >= 3 and str(q[2]).lower() == "meters"]
        pvals = [float(q[1]) for q in pq if len(q) >= 3 and str(q[2]).lower() == "meters"]
        # pair in order
        for i, gv in enumerate(gvals):
            if i < len(pvals):
                pairs.append((gv, pvals[i]))
    return _mean_relative_error(pairs)

# Average relative error of object-to-object angle (in degrees)
def omrae(gt_pairs: List[Dict[str, Any]], pred_pairs: List[Dict[str, Any]]) -> float:

    pairs = []
    for g, p in zip(gt_pairs, pred_pairs):
        gq = g.get("quantities", [])
        pq = p.get("quantities", [])
        for cls in set(
            [q[0] for q in gq if len(q) >= 3 and _norm_str(str(q[2])) == "degrees"]
        ):
            gt_vals = _extract_quantity(gq, cls, unit="degrees")
            pr_vals = _extract_quantity(pq, cls, unit="degrees")
            for i, v in enumerate(gt_vals):
                if i < len(pr_vals):
                    pairs.append((v, pr_vals[i]))
    return _mean_relative_error(pairs)



def perception_clip_score(
    gt_item: Dict[str, Any], pred_item: Dict[str, Any]
) -> Dict[str, float]:

    cls_res = per_class_accuracy(gt_item, pred_item)
    return {
        "PerClassWeighted": cls_res["weighted"],
        "PerClassDetail": cls_res["per_class"],
    }


# 3. Trajectory & Behavior

# Make up points
def _prepend_origin(traj):
    if not traj:
        return traj
    return [(0.0, 0.0, 0.0, 0.0)] + traj

# Calculation speed
def _vel(p1, p2):
    (x1, y1, z1, t1), (x2, y2, z2, t2) = p1, p2
    dt = t2 - t1
    if dt <= 0:
        return (0.0, 0.0, 0.0)
    return (
        (x2 - x1) / dt,
        (y2 - y1) / dt,
        (z2 - z1) / dt,
    )

# Calculate acceleration
def _acc(p1, p2, p3):
    v1 = _vel(p1, p2)
    v2 = _vel(p2, p3)
    dv = (
        v2[0] - v1[0],
        v2[1] - v1[1],
        v2[2] - v1[2],
    )
    dt = p3[3] - p2[3]
    if dt <= 0:
        return 0.0
    return math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2) / dt

# Calculate angle
def _heading(v):
    vx, _, vz = v
    return math.degrees(math.atan2(vx, vz))

# Returning the signed acceleration along the forward direction (z-axis)
def _signed_acceleration(p1, p2, p3):

    (x1, y1, z1, t1), (x2, y2, z2, t2), (x3, y3, z3, t3) = p1, p2, p3
    dt1 = t2 - t1
    dt2 = t3 - t2
    if dt1 <= 0 or dt2 <= 0:
        return 0.0

    v_prev_z = (z2 - z1) / dt1
    v_now_z  = (z3 - z2) / dt2

    return (v_now_z - v_prev_z) / dt2

# Change in speed (along the direction of travel):acceleration,deceleration,constant
def _speed_bucket(a: float) -> str:

    if a >= 0.15:
        return "acceleration"
    if a <= -0.15:
        return "deceleration"
    return "constant"

# Change in angle:left_turn,right_turn,straight
def _turn_bucket(dtheta: float) -> str:

    if dtheta >= 8:
        return "left_turn"
    if dtheta <= -8:
        return "right_turn"
    return "straight"

# Return two action dimensions simultaneously: (speed change, steering).
def _action_buckets(a: float, dtheta: float) -> tuple[str, str]:

    return _speed_bucket(a), _turn_bucket(dtheta)

# Generate action sequences from 4D trajectory points (x, y, z, t).
def _trajectory_actions(
    traj: List[Tuple[float, float, float, float]]
) -> List[Tuple[str, str]]:

    n = len(traj)
    if n < 3:
        return []

    actions: List[Tuple[str, str]] = []

    for i in range(1, n - 1):
        p_prev, p_curr, p_next = traj[i - 1], traj[i], traj[i + 1]

        # velocity vector
        v_prev = _vel(p_prev, p_curr)
        v_now  = _vel(p_curr, p_next)

        # Signed acceleration (along z)
        a_signed = _signed_acceleration(p_prev, p_curr, p_next)

        # Heading angle variation (XZ plane)
        heading_prev = _heading(v_prev)
        heading_now  = _heading(v_now)
        dtheta = heading_now - heading_prev

        # Unified Action Discretization
        actions.append(_action_buckets(a_signed, dtheta))

    return actions

# Action consistency score
def dcs_acc(gt_traj, pred_traj) -> float:
    g = _trajectory_actions(gt_traj)
    p = _trajectory_actions(pred_traj)

    m = 0
    for (gs, gt), (ps, pt) in zip(g, p):
        if gs == ps and gt == pt:
            m += 1

    return m / 5.0


# Mean relative error (acceleration)
def mre_acceleration(gt_traj, pred_traj) -> float:

    def seq_acc(traj):
        if len(traj) < 3:
            return []
        accs = []
        for i in range(1, len(traj) - 1):
            accs.append(_acc(traj[i - 1], traj[i], traj[i + 1]))
        return accs[:5]

    ga = seq_acc(gt_traj)
    pa = seq_acc(pred_traj)

    n = min(len(ga), len(pa), 5)
    if n == 0:
        return 0.0

    err_sum = 0.0
    valid = 0
    for i in range(n):
        gv = ga[i]
        pv = pa[i]
        if gv != 0:

            err_sum += abs(pv - gv) / (abs(gv)+0.002)
            valid += 1

    return err_sum / valid if valid > 0 else 0.0

# Mean relative error (variation of heading angle)
def are_heading(gt_traj, pred_traj) -> float:

    def seq_dtheta(traj):
        if len(traj) < 3:
            return []
        dts = []
        for i in range(1, len(traj) - 1):
            v_prev = _vel(traj[i - 1], traj[i])
            v_now  = _vel(traj[i], traj[i + 1])
            dts.append(_heading(v_now) - _heading(v_prev))
        return dts[:5]

    gd = seq_dtheta(gt_traj)
    pd = seq_dtheta(pred_traj)

    n = min(len(gd), len(pd), 5)
    if n == 0:
        return 0.0

    err_sum = 0.0
    valid = 0
    for i in range(n):
        gv = gd[i]
        pv = pd[i]
        if gv != 0:

            err_sum += abs(pv - gv) / (abs(gv)+0.01)
            valid += 1

    return err_sum / valid if valid > 0 else 0.0

# Average Displacement Error
def ade(gt_traj, pred_traj):
    n = min(len(gt_traj), len(pred_traj))
    if n == 0:
        return 0.0
    d = 0.0
    for i in range(n):
        dx = pred_traj[i][0] - gt_traj[i][0]
        dy = pred_traj[i][1] - gt_traj[i][1]
        dz = pred_traj[i][2] - gt_traj[i][2]
        d += math.sqrt(dx*dx + dy*dy + dz*dz)
    return d / n

# Final Displacement Error evaluated at fixed future indices (Δt = 0.5s)
def fde_at_index(gt_traj, pred_traj, idx: int) -> float:

    if not gt_traj or not pred_traj:
        return 0.0

    n = min(len(gt_traj), len(pred_traj))
    if idx < 0 or idx >= n:
        return 0.0

    gx, gy, gz = gt_traj[idx][:3]
    px, py, pz = pred_traj[idx][:3]

    return math.sqrt(
        (px - gx) ** 2 +
        (py - gy) ** 2 +
        (pz - gz) ** 2
    )

def fde_at_T(gt_traj, pred_traj):
    if not gt_traj or not pred_traj:
        return 0.0

    if len(gt_traj) > 1 and gt_traj[0][:3] == (0.0, 0.0, 0.0):
        gt_traj = gt_traj[1:]
    if len(pred_traj) > 1 and pred_traj[0][:3] == (0.0, 0.0, 0.0):
        pred_traj = pred_traj[1:]

    # Situation 1：Q17~Q19（Single-point FDE）
    if len(gt_traj) == 1 or len(pred_traj) == 1:
        dx = pred_traj[0][0] - gt_traj[0][0]
        dy = pred_traj[0][1] - gt_traj[0][1]
        dz = pred_traj[0][2] - gt_traj[0][2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    # Situation 2：Q15~Q16
    last_idx = min(len(gt_traj), len(pred_traj)) - 1
    dx = pred_traj[last_idx][0] - gt_traj[last_idx][0]
    dy = pred_traj[last_idx][1] - gt_traj[last_idx][1]
    dz = pred_traj[last_idx][2] - gt_traj[last_idx][2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)



# 4. Summary and main process

# Error-related indicators are uniformly normalized
def normalize_piecewise_error(E, params):
    x1 = params["x1"]
    x2 = params["x2"]
    k  = params["k"]

    if E < x1:
        return 100.0

    if E < x2:
        return 100.0 - 40.0 * (E - x1) / (x2 - x1 + 1e-9)

    return 60.0 * math.exp(-k * (E - x2))

# Main scoring process
def _score_from_flat_pairs(gt_list: list, pr_list: list, norm_params: dict, route_by: str = "qid", use_gpt=False) -> dict:

    # Initialize all indicator pools
    global_gpt_scores = []
    scene_spice_pairs, scene_risk_pairs, scene_gpt_pairs = [], [], []
    perclass_pairs = [], []
    emrde_g, emrde_p, emrae_g, emrae_p = [], [], [], []
    omrde_g, omrde_p, omrae_g, omrae_p = [], [], [], []
    traj_dcs, traj_mreA, traj_are, traj_ade, traj_fde1, traj_fde2, traj_fde3 = [], [], [], [], [], [], []

    # core distribution loop
    for g, p in zip(gt_list, pr_list):
        qid = _norm_str(g.get("question_type", g.get("qid", "")))
        cat = _norm_str(g.get("category", ""))

        # Scene Understanding 
        if (route_by == "qid" and qid in {"q12", "q13", "q14"}) or (
            route_by == "category" and cat == "scene_understanding"):
            scene_spice_pairs.append((g, p))
            if qid in {"q12", "q14"}:
                scene_risk_pairs.append((g, p))
            scene_gpt_pairs.append((g, p))

        # Perception 
        if (route_by == "qid" and qid in {"q4", "q5", "q7", "q11"}) or (
            route_by == "category" and cat == "perception_class"):
            perclass_pairs.append((g, p))

        if (route_by == "qid" and qid in {"q1", "q4", "q6", "q11"}) or (
            route_by == "category" and cat == "perception_emre_dist"):
            emrde_g.append(g); emrde_p.append(p)

        if (route_by == "qid" and qid in {"q1","q21"}) or (
            route_by == "category" and cat == "perception_emre_angle"):
            emrae_g.append(g); emrae_p.append(p)

        if (route_by == "qid" and qid in {"q20"}) or (
            route_by == "category" and cat == "perception_omre_dist"
        ):
            omrde_g.append(g); omrde_p.append(p)

        if (route_by == "qid" and qid in {"q22"}) or (
            route_by == "category" and cat == "perception_omre_angle"
        ):
            omrae_g.append(g); omrae_p.append(p)

        # Trajectory & Behavior
        if (route_by == "qid" and qid in {"q15", "q16"}) or (
            route_by == "category" and cat in {"trajectory_behavior", "trajectory"}
        ):
            g_traj = _extract_traj_from_item(g)
            p_traj = _extract_traj_from_item(p)
            g_mod = dict(g, trajectory_gt=_prepend_origin(g_traj))
            p_mod = dict(p, trajectory_pred=_prepend_origin(p_traj))
            traj_dcs.append((g_mod, p_mod))
            traj_mreA.append((g_mod, p_mod))
            traj_are.append((g_mod, p_mod))
            traj_ade.append((g_mod, p_mod))
            traj_fde3.append((g_mod, p_mod))

        if (route_by == "qid" and qid in {"q17", "q18", "q19"}) or (
            route_by == "category" and cat in {"trajectory_fde1", "trajectory_fde2", "trajectory_fde3"}
        ):
            g_traj = _extract_traj_from_item(g, is_gt=True)
            p_traj = _extract_traj_from_item(p, is_gt=False)
            g_mod = dict(g, trajectory_gt=g_traj)
            p_mod = dict(p, trajectory_pred=p_traj)

            if qid == "q17" or cat == "trajectory_fde1":
                traj_fde1.append((g_mod, p_mod))
            elif qid == "q18" or cat == "trajectory_fde2":
                traj_fde2.append((g_mod, p_mod))
            elif qid == "q19" or cat == "trajectory_fde3":
                traj_fde3.append((g_mod, p_mod))

    # ---------- Scene Understanding ----------
    scene_spice_vals = []

    clip_gt_pool = [gg for gg, _ in scene_spice_pairs]
    clip_q12_risk = _find_q12_risk(clip_gt_pool)  # If Q12 is not available, return "unknown"

    # 1) Calculate SPICE on scene_spice_pairs 
    for g, p in scene_spice_pairs:
        spice_val = spice_score(g, p)
        scene_spice_vals.append(spice_val)


    # 2) Risk-Class-Acc statistics are calculated separately for Q12/Q14.
    scene_risk_vals = []
    for g, p in scene_risk_pairs:
        risk_val = risk_class_acc(g, p)
        scene_risk_vals.append(risk_val)
        

    # Average summary
    scene_spice = float(np.mean(scene_spice_vals) if scene_spice_vals else 0.0)
    scene_risk  = float(np.mean(scene_risk_vals)  if scene_risk_vals  else 0.0)

    # Scene total score
    scene_spice *= 100.0
    scene_risk  *= 100.0

    scene_total = 0.7 * scene_spice + 0.3 * scene_risk 


    # ---------- Perception ----------
    perclass_weighted = _safe_div(sum(per_class_accuracy(g,p)["weighted"] for g,p in perclass_pairs), len(perclass_pairs)) if perclass_pairs else 0

    val_emrde = emrde(emrde_g, emrde_p)
    val_emrae = emrae(emrae_g, emrae_p)
    val_omrde = omrde(omrde_g, omrde_p)
    val_omrae = omrae(omrae_g, omrae_p)
    score_emrde = normalize_piecewise_error(val_emrde, norm_params["EMRDE"])
    score_emrae = normalize_piecewise_error(val_emrae, norm_params["EMRAE"])
    score_omrde = normalize_piecewise_error(val_omrde, norm_params["OMRDE"])
    score_omrae = normalize_piecewise_error(val_omrae, norm_params["OMRAE"])

    # Perception total score
    perclass_weighted *= 100.0

    percep_total = (
        0.2 * perclass_weighted
        + 0.3 * score_emrde
        + 0.2 * score_emrae
        + 0.2 * score_omrde
        + 0.1 * score_omrae
    )

    # ---------- Motion Planning ----------
    dcs_score  = np.mean([dcs_acc(g.get("trajectory_gt",[]), p.get("trajectory_pred",[])) for g,p in traj_dcs]) if traj_dcs else 0
    mreA_val = np.mean([mre_acceleration(g.get("trajectory_gt",[]), p.get("trajectory_pred",[])) for g,p in traj_mreA]) if traj_mreA else 0
    are_val  = np.mean([are_heading(g.get("trajectory_gt",[]), p.get("trajectory_pred",[])) for g,p in traj_are]) if traj_are else 0
    ade_val  = np.mean([ade(g.get("trajectory_gt",[]), p.get("trajectory_pred",[])) for g,p in traj_ade]) if traj_ade else 0
    fde1_val = np.mean([fde_at_index(g["trajectory_gt"], p["trajectory_pred"], idx=1) for g, p in traj_fde3]) if traj_fde3 else 0.0
    fde2_val = np.mean([fde_at_index(g["trajectory_gt"], p["trajectory_pred"], idx=3) for g, p in traj_fde3]) if traj_fde3 else 0.0
    fde3_val = np.mean([fde_at_index(g["trajectory_gt"], p["trajectory_pred"], idx=5) for g, p in traj_fde3]) if traj_fde3 else 0.0

    mreA_score  = normalize_piecewise_error(mreA_val, norm_params["MRE_A"])
    are_score   = normalize_piecewise_error(are_val,   norm_params["ARE"])
    ade_score   = normalize_piecewise_error(ade_val,   norm_params["ADE"])
    fde1_score  = normalize_piecewise_error(fde1_val,  norm_params["FDE1"])
    fde2_score  = normalize_piecewise_error(fde2_val,  norm_params["FDE2"])
    fde3_score  = normalize_piecewise_error(fde3_val,  norm_params["FDE3"])

    # Motion Planning Total Score
    dcs_score *= 100.0

    traj_total = (
        0.2 * dcs_score
        + 0.1 * mreA_score
        + 0.1 * are_score
        + 0.1 * fde1_score
        + 0.1 * fde2_score
        + 0.2 * fde3_score
        + 0.2 * ade_score
    )

    # Global GPT-based semantic scoring (optional)
    if not use_gpt:
        avg_gpt_global = 0.0
    else:
        global_gpt_scores = []
        for g, p in zip(gt_list, pr_list):
            qid = str(g.get("question_type", g.get("qid", ""))).lower()
            question = g.get("question", "")
            gt_text = g.get("sentence") or g.get("answer") or str(g)
            pred_text = p.get("sentence") or p.get("answer") or str(p)

            if qid in {"q12", "q13", "q14"}:
                # Scene-based questions: with SPICE and Risk information
                spice_val = spice_score(g, p)
                risk_val = risk_class_acc(g, p) if qid in {"q12", "q14"} else "N/A"
                gt_risk = _extract_attr_value(g.get("attributes", []), "risk_level") or "unknown"
                gpt_val = gpt_score_api(
                    question=question,
                    gt_text=gt_text,
                    pred_text=pred_text,
                    spice=spice_val,
                    risk_acc=risk_val,
                    gt_risk=gt_risk
                )
            else:
                # Non-Scene Questions: Only Question, Truth Value, and Answer
                gpt_val = gpt_score_api(
                    question=question,
                    gt_text=gt_text,
                    pred_text=pred_text,
                    spice="N/A",
                    risk_acc="N/A",
                    gt_risk="N/A"
                )

            global_gpt_scores.append(gpt_val)

        avg_gpt_global = float(np.mean(global_gpt_scores) if global_gpt_scores else 0.0)

    avg_gpt_global *= 100.0


    # ---------- Overall score ----------
    scene_weight, percep_weight, traj_weight,gpt_weight = 0.15, 0.35, 0.40 , 0.1
    overall = scene_weight*scene_total + percep_weight*percep_total + traj_weight*traj_total+gpt_weight*avg_gpt_global

    results = {
        "Scene Understanding": {
            "average_SPICE": scene_spice,
            "average_Risk-Class-Acc": scene_risk,
            "Scene-Understanding(avg)": scene_total,
        },
        "Perception": {
            "Class-Acc": perclass_weighted,
            "EMRDE(raw)": val_emrde, "EMRDE(score)": score_emrde,
            "EMRAE(raw)": val_emrae, "EMRAE(score)": score_emrae,
            "OMRDE(raw)": val_omrde, "OMRDE(score)": score_omrde,
            "OMRAE(raw)": val_omrae, "OMRAE(score)": score_omrae,
            "Perception(total)": percep_total,
        },
        "Trajectory & Behavior": {
                "DCS-Acc": dcs_score,
                "MRE-Acceleration(raw)": mreA_val,
                "MRE-Acceleration(score)": mreA_score,
                "ARE(raw)": are_val,
                "ARE(score)": are_score,
                "ADE(raw)": ade_val,
                "ADE(score)": ade_score,
                "FDE@1(raw)": fde1_val,
                "FDE@1(score)": fde1_score,
                "FDE@2(raw)": fde2_val,
                "FDE@2(score)": fde2_score,
                "FDE@3(raw)": fde3_val,
                "FDE@3(score)": fde3_score,
                "Trajectory(total)": traj_total,
        },
        "GPT-Score": {
                "average_GPT_Score": avg_gpt_global,
        },
        "Overall": {
            "Scene-Understanding(avg)": scene_total,
            "Perception(total)": percep_total,
            "Trajectory(total)": traj_total,
            "GPT-Score": avg_gpt_global,
            "Overall Score": overall,
        }
    }
    return results



def _accumulate_numeric_metrics(acc: dict, d: dict, path: tuple = ()):

    for k, v in d.items():
        if isinstance(v, dict):
            _accumulate_numeric_metrics(acc, v, path + (k,))
        else:
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                acc.setdefault(path + (k,), []).append(float(v))

def _build_avg_from_acc(acc: dict) -> dict:

    out = {}
    for path, vals in acc.items():
        cursor = out
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = float(np.mean(vals)) if vals else 0.0
    return out


def main(
    path: str,
    route_by: str = "qid",
    save_json: bool = True,
    out_path: str =  None,
    norm_params_path: str = None,
    max_clips: int = 10000,
    use_gpt: bool = True,
) -> dict:

    # Read norm_params.json
    norm_params = None
    if norm_params_path and os.path.exists(norm_params_path):
        with open(norm_params_path, "r", encoding="utf-8") as f:
            norm_params = json.load(f)
        print(f"The norm_params file has been read：{norm_params_path}")
    else:
        print("If the path to the norm_params file is not provided, only the raw error will be calculated.")

    # Read input file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # clip format: score clip by clip
    if isinstance(data, list) and data and isinstance(data[0], dict) \
       and "ref_results" in data[0] and "model_results" in data[0]:
        
        clips_to_process = data[0:0+max_clips]  

        clip_scores = []
        overall_list = []
        scene_list, percep_list, traj_list, gpt_list = [], [], [], []

        # Used to record the cumulative value of all indicators.
        metric_accumulator = {}

        for i, clip in enumerate(clips_to_process, 1):  
            print(f"Processing {i}/{len(clips_to_process)}...")  

            clip_index = clip.get("clip_prefix")
            ref_list = clip.get("ref_results", [])
            model_list = clip.get("model_results", [])
            res = _score_from_flat_pairs(ref_list, model_list, norm_params=norm_params, route_by=route_by, use_gpt=use_gpt)

            # Save the complete result for each clip
            gpt_score = res.get("GPT-Score", {}).get("average_GPT_Score", None)
            if gpt_score is not None:
                res["GPT-Score"] = {"average_GPT_Score": gpt_score}
            clip_scores.append({"clip_index": clip_index, "scores": res})

            overall_list.append(res["Overall"]["Overall Score"])
            scene_list.append(res["Overall"]["Scene-Understanding(avg)"])
            percep_list.append(res["Overall"]["Perception(total)"])
            traj_list.append(res["Overall"]["Trajectory(total)"])

            if "GPT-Score" in res:
                gpt_list.append(res["GPT-Score"]["average_GPT_Score"])

            # Recursively collect all numerical indicators
            _accumulate_numeric_metrics(metric_accumulator, res)

        # Calculate global average
        avg_by_metric = _build_avg_from_acc(metric_accumulator)

        def _filter_metric_tree(tree: dict) -> dict:
            REMOVE_KEYS = {
                "Scene-Understanding(avg)",
                "Perception(total)",
                "Trajectory(total)",
                "average_GPT_Score",
                "Overall Score"
            }
            out = {}
            for k, v in tree.items():
                if k in REMOVE_KEYS:
                    continue
                if isinstance(v, dict):
                    filtered_sub = _filter_metric_tree(v)
                    if filtered_sub:
                        out[k] = filtered_sub
                else:
                    out[k] = v
            return out

        avg_by_metric = _filter_metric_tree(avg_by_metric)


        final = {
            "clip_scores": clip_scores,
            "overall_avg": float(np.mean(overall_list) if overall_list else 0.0),
            "scene_avg": float(np.mean(scene_list) if scene_list else 0.0),
            "perception_avg": float(np.mean(percep_list) if percep_list else 0.0),
            "trajectory_avg": float(np.mean(traj_list) if traj_list else 0.0),
            "gpt_avg": float(np.mean(gpt_list) if gpt_list else 0.0),
            "metrics_avg_detail": avg_by_metric
        }

        if save_json:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final, f, indent=4, ensure_ascii=False)
            print(f"✅ 按 clip 打分完成，已保存到 {out_path}")

        # All metric paths (normalized) that need to be written into the table.
        TARGET_KEYS = [
            ("Scene Understanding", "average_SPICE"),
            ("Scene Understanding", "average_Risk-Class-Acc"),
            ("Scene Understanding", "Scene-Understanding(avg)"),

            ("Perception", "Class-Acc"),
            ("Perception", "EMRDE(score)"),
            ("Perception", "EMRAE(score)"),
            ("Perception", "OMRDE(score)"),
            ("Perception", "OMRAE(score)"),
            ("Perception", "Perception(total)"),

            ("Trajectory & Behavior", "DCS-Acc"),
            ("Trajectory & Behavior", "MRE-Acceleration(score)"),
            ("Trajectory & Behavior", "ARE(score)"),
            ("Trajectory & Behavior", "ADE(score)"),
            ("Trajectory & Behavior", "FDE@1(score)"),
            ("Trajectory & Behavior", "FDE@2(score)"),
            ("Trajectory & Behavior", "FDE@3(score)"),
            ("Trajectory & Behavior", "Trajectory(total)"),

            ("GPT-Score", "average_GPT_Score"),

            ("Overall", "Overall Score"),
        ]

        # CSV file name
        csv_path = out_path.replace(".json", ".csv")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # Indicator name
            header = ["clip"] + [" / ".join(k) for k in TARGET_KEYS]
            writer.writerow(header)

            # Score for each clip
            for clip, res in zip(clip_scores, [c["scores"] for c in clip_scores]):
                row = [f"clip_{clip['clip_index']}"]
                for a, b in TARGET_KEYS:
                    row.append(res[a][b])
                writer.writerow(row)

            # === Overall Model Performance===
            avg_row = ["model"]
            avg_vals = []

            for a, b in TARGET_KEYS:
                avg_vals.append(avg_by_metric[a][b] if a in avg_by_metric and b in avg_by_metric[a] else 0.0)
            avg_row.extend(avg_vals)
            writer.writerow(avg_row)

        print(f"The table has saved as {csv_path}")

        return final


    if data and isinstance(data, list) and isinstance(data[0], (list, tuple)) and len(data[0]) == 2:
        gt_data = [x[0] for x in data]
        pr_data = [x[1] for x in data]
    elif isinstance(data, dict) and "gt" in data and "pred" in data:
        gt_data, pr_data = data["gt"], data["pred"]
    else:
        raise ValueError("Structural error")

    results = _score_from_flat_pairs(gt_data, pr_data, norm_params=norm_params, route_by=route_by, use_gpt=use_gpt)
    if save_json:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"The score has saved as {out_path}")
    return results


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# The initialization of gpt4o
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your api"),
    base_url="your base url"
)
GPT_LOG_PATH = r'...'


if __name__ == "__main__":


    data_path = r"your data-path after parser"
    data = load_json(data_path)
    OUTPUT_PATH = r"..."
    NORM_PATH = r"your norm params"
    main(path=data_path, route_by="qid", save_json=True, out_path=OUTPUT_PATH, norm_params_path=NORM_PATH)