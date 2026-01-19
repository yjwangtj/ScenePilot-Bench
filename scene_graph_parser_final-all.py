"""
Scene semantic graph parser
"""

import re
import json
from pathlib import Path
import argparse
import os
import ast
import operator

# Path configuration
INPUT_JSON_PATH = r'...'
OUTPUT_JSON_PATH = r'...'


# Basic Configuration & General Tools
OBJECT_ALIASES = {
    "vehicle": ["vehicle", "car", "van", "auto", "sedan", "suv"],
    "truck": ["truck", "lorry"],
    "pedestrian": ["pedestrian", "person", "people", "walker"],
    "bicycle": ["bicycle", "bike", "cyclist"],
    "ego_vehicle": ["ego vehicle", "our car", "ego", "ego car"],
    "motorcycle": ["motorcycle", "motorbike"],
    "animal": ["animal"],
}

SEMANTIC_NOUN_WHITELIST = {
    "risk",
    "intersection",
    "weather",
    "road",
    "road type",
    "lane",
    "road_type",
    "time",
}

NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


DIST_PATTERN = re.compile(r"(\[?[-+]?\d+(?:\.\d+)?\]?)\s*(?:[-\s]?)?(m|meter|meters)\b", re.I)
ANGLE_PATTERN = re.compile(r"(\[?-?\d+(?:\.\d+)?\]?)\s*(degrees?|°)\b", re.I)
BBOX_PARENS_PATTERN = re.compile(r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]")
TRIPLE_PATTERN = re.compile(r"\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]")
BEARING_PATTERN = re.compile(r"([\d.]+)\s*degrees?\s*\(\s*(left|right|front|behind|rear)\s*\)",re.I,)

# Unify some symbols
def normalize_text(t: str) -> str:

    return (
        t.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("\u3000", " ")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
    )

# Deduplication while maintaining order
def dedup(seq):

    out, seen = [], set()

    def _key(x):
        if isinstance(x, list):
            return tuple(_key(e) for e in x)
        if isinstance(x, dict):
            return tuple(sorted((k, _key(v)) for k, v in x.items()))
        return x

    for i in seq:
        k = _key(i)
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out

# clean numbers
def safe_float(x: str):

    return float(x.strip().rstrip("."))

# Map words to standard category names
def canonical_class(word: str) -> str | None:

    w = word.lower().strip()
    for cls, aliases in OBJECT_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", w):
                return cls
    return None

# Robust identification of all object categories appearing in text
def detect_classes_in_text(text: str):

    t = text.lower()

    # Constructing alias→class mapping
    alias_map = []
    for cls, aliases in OBJECT_ALIASES.items():
        for alias in aliases:
            # Supports plural forms: car → cars, person → people
            alias_map.append((alias, cls))
            if alias.endswith("y"):  # bicycle → bicycles
                alias_map.append((alias[:-1] + "ies", cls))
            elif not alias.endswith("s"):  # car → cars, truck → trucks
                alias_map.append((alias + "s", cls))

    alias_map.sort(key=lambda x: -len(x[0]))

    found = []
    spans = []

    for alias, cls in alias_map:
        # Find all matching positions
        for m in re.finditer(rf"\b{re.escape(alias)}\b", t):
            spans.append((m.start(), cls))

    # Sort by actual order of appearance
    spans.sort(key=lambda x: x[0])

    for _, cls in spans:
        if cls not in found:
            found.append(cls)

    return found

# Find "the main object"
def main_object(text: str, prefer_non_ego=True, default="object") -> str:
 
    classes = detect_classes_in_text(text)
    if not classes:
        return default
    if prefer_non_ego and len(classes) > 1:
        # The last non-ego_vehicle
        non_ego = [c for c in classes if c != "ego_vehicle"]
        if non_ego:
            return non_ego[-1]
    return classes[-1]

# Extract distance
def extract_distance(text: str):
    t = normalize_text(text)
    m = DIST_PATTERN.search(t)
    if m:
        dist_str = m.group(1).strip("[]")
        try: 
            dist_val = float(dist_str)
            return dist_val
        except ValueError:
            return None  
    return float(m.group(1)) if m else None

# Extract angle
def extract_angle(text: str):
    t = normalize_text(text)
    m = ANGLE_PATTERN.search(t)
    if m:
        angle_str = m.group(1).strip("[]") 
        try:
            angle_val = float(angle_str)
            return angle_val
        except ValueError:
            return None   
    return float(m.group(1)) if m else None

# Parse all 2D boxes,unified return format
def extract_bboxes(text: str):

    t = normalize_text(text)
    boxes = []

    # Complete Quadruple (cx, cy, w, h)
    BBOX4 = re.compile(
        r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)"
    )
    for m in BBOX4.finditer(t):
        vals = [safe_float(v) for v in m.groups()]
        cx, cy, w, h = vals
        boxes.append(["bounding_box", [cx, cy, w, h], "normalized"])

    # square bracket form
    BBOX4_SQ = re.compile(
        r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]"
    )
    for m in BBOX4_SQ.finditer(t):
        vals = [safe_float(v) for v in m.groups()]
        cx, cy, w, h = vals
        boxes.append(["bounding_box", [cx, cy, w, h], "normalized"])

    if boxes:
        return boxes

    # Find the center point
    CENTER2 = re.compile(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")
    m_center = CENTER2.search(t)
    if not m_center:
        return [] 

    cx, cy = map(float, m_center.groups())

    # Identify height and width
    WH_SPANS = re.compile(r"spans\s*([-\d.]+)\s*(?:by|×|x)\s*([-\d.]+)")
    WH_MEAS = re.compile(
        r"measures\s*([-\d.]+)\s*(?:in\s+width)?\s*(?:and|,)\s*([-\d.]+)\s*(?:in\s+height)?"
    )
    WH_NORM = re.compile(
        r"width\s+of\s+([-\d.]+)\s*(?:and\s+height\s+of\s+([-\d.]+))"
    )
    WH_UNITS = re.compile(
        r"box\s+is\s*([-\d.]+)\s*(?:units\s+)?wide\s*(?:and\s+([-\d.]+)\s*(?:units\s+)?tall)"
    )
    WH_BY = re.compile(r"area\s+of\s*([-\d.]+)\s*(?:by|×|x)\s*([-\d.]+)")

    w = h = None

    for pat in [WH_SPANS, WH_MEAS, WH_NORM, WH_UNITS, WH_BY]:
        m = pat.search(t)
        if m:
            gx, gy = m.groups()
            w = safe_float(gx)
            h = safe_float(gy)
            break

    if w is None or h is None:
        return []

    return [["bounding_box", [cx, cy, w, h], "normalized"]]





# safe operator
SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def safe_eval_expr(expr):

    def _eval(node):
        if isinstance(node, ast.Num):  
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("Illegal expression")

    node = ast.parse(expr, mode='eval').body
    return _eval(node)

# Extract track points
TRIPLE_RAW_PAT = re.compile(r"\[\s*([^\[\]]+?)\s*\]")

def extract_traj_points(text: str):

    text = normalize_text(text)

    matches = TRIPLE_RAW_PAT.findall(text)
    triples = []

    for m in matches:
        parts = [p.strip() for p in m.split(",")]
        if len(parts) not in (2, 3):
            continue

        values = []
        for p in parts:
            try:
                values.append(safe_eval_expr(p))
            except Exception:
                values.append(0.0)

        if len(values) == 2:
            t = 0.5 * (len(triples) + 1)
            values.append(t)

        triples.append(values)

        if len(triples) == 6:
            break

    if len(triples) > 6:
        triples = triples[:6]

    if len(triples) == 0:
        return [[0.0, 0.0, 0.0] for _ in range(6)]

    while len(triples) < 6:
        triples.append([0.0, 0.0, 0.0])

    return triples


# Extract single point
def extract_single_coords(text: str):

    pts = extract_traj_points(text)
    return pts[0] if pts else [0, 0, 0]


BEARING_PATTERN_Q22 = re.compile(
    r"([-\d.]+)\s*degrees?\s*\(\s*(left|right|front|behind|rear|straight\s+ahead)\s*\)"
)

# Extracting the angle between the car and the object
def extract_bearing_and_dir(text: str):

    t = normalize_text(text)

    m = BEARING_PATTERN_Q22.search(t)
    if m:
        bearing = float(m.group(1))  # Extract angle
        direction = m.group(2).lower()  # Extraction direction

        # Processing special directions
        if direction == "straight ahead":
            direction = "straight_ahead"
        
        return bearing, direction

    if "ahead"  in t:
        return 0.0, "ahead"

    return None, None



# True or False Questions
def yes_no_from_text(text: str):

    t = text.strip().lower()
    if t.startswith("yes") or " yes," in t or " yep" in t:
        return True
    if t.startswith("no") or " no," in t or " nope" in t:
        return False
    return None


# Question-by-question analysis for Q1 – Q22

# Q1 ：meters+degrees
def parse_Q1(text: str):

    obj = main_object(text)
    dist = extract_distance(text)
    ang = extract_angle(text)
    quantities = []
    if dist is not None:
        quantities.append([obj, dist, "meters"])
    if ang is not None:
        quantities.append([obj, ang, "degrees"])
    return {
        "sentence": text,
        "objects": [obj] if obj != "object" else [],
        "attributes": [],
        "quantities": quantities,
        "bounding_boxes": [],
    }


# Q3: Bounding box parsing
def parse_Q3(text: str):

    boxes = extract_bboxes(text)
    return {
        "sentence": text,
        "objects": [],
        "attributes": [],
        "quantities": [],
        "bounding_boxes": boxes,
    }

# Q4: Closest goal
def parse_Q4(text: str):

    obj = main_object(text)
    dist = extract_distance(text)
    attrs = [[obj, "closest"]]
    quants = []
    if dist is not None:
        quants.append([obj, dist, "meters"])
    return {
        "sentence": text,
        "objects": dedup(["ego_vehicle", obj]) if obj != "object" else ["ego_vehicle"],
        "attributes": attrs,
        "quantities": quants,
        "bounding_boxes": [],
    }

# Q5: Count the total number + all object categories
def parse_Q5(text: str):

    t = normalize_text(text.lower())

    count = None

    for word, num in NUM_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            count = num
            break

    if count is None:
        m = re.search(
             r"\b(\d+)\s+(?:notable\s+|visible\s+)?(?:objects?|targets?|participants?)\b",   
            t
        )
        if m:
            count = int(m.group(1))

    objects = detect_classes_in_text(text)

    quantities = []
    if count is not None:
        quantities.append(["objects", count])

    return {
        "sentence": text,
        "objects": objects,
        "attributes": [],
        "quantities": quantities,
        "bounding_boxes": [],
    }

# Q6: Compare distance differences (meters difference)
def parse_Q6(text: str):

    t = normalize_text(text.lower())
    quantities = []
    objects = []

    # Utility function
    def cls(x):
        return canonical_class(x) or x

    def key(c, s):
        return f"{c}_{s}" if s else c

    # allowed directions
    SIDE = r"(ahead|behind|left|right|left-front|right-front|rear)"

    # Template 1 ： A_on_sideA is DIFF meters closer/farther than B_on_sideB
    pat1 = re.compile(
        rf"the\s+([a-z_]+)\s+on\s+the\s+{SIDE}\s+is\s+([\d\.]+)\s*meters?\s+"
        rf"(closer|farther)\s+than\s+the\s+([a-z_]+)\s+on\s+the\s+{SIDE}",
        re.I,
    )
    m = pat1.search(t)
    if m:
        clsA_raw, sideA, diff, rel, clsB_raw, sideB = m.groups()
        clsA = cls(clsA_raw)
        clsB = cls(clsB_raw)
        objects = dedup([clsA, clsB])
        quantities.append([key(clsA, sideA), float(diff), "meters"])
        return {
            "sentence": text,
            "objects": objects,
            "attributes": [],
            "quantities": quantities,
            "bounding_boxes": [],
        }


    # Template 2： B_on_sideB is DIFF meters farther/closer than A_on_sideA
    pat4 = re.compile(
        rf"the\s+([a-z_]+)\s+on\s+the\s+{SIDE}\s+is\s+([\d\.]+)\s*meters?\s+"
        rf"(closer|farther)\s+than\s+the\s+([a-z_]+)\s+on\s+the\s+{SIDE}",
        re.I,
    )
    m = pat4.search(t)
    if m:
        clsB_raw, sideB, diff, rel, clsA_raw, sideA = m.groups()  # Note that the order is reversed
        clsA = cls(clsA_raw)
        clsB = cls(clsB_raw)
        objects = dedup([clsA, clsB])

        quantities.append([key(clsA, sideA), float(diff), "meters"])
        return {
            "sentence": text,
            "objects": objects,
            "attributes": [],
            "quantities": quantities,
            "bounding_boxes": [],
        }

    # Template 3：There is a DIFF-meter distance difference: A(sideA) is closer/farther than B(sideB)
    pat2 = re.compile(
        rf"there\s+is\s+a\s+([\d\.]+)-?meter\s+(?:distance\s+)?difference[:\s]+"
        rf"the\s+([a-z_]+)\s*\({SIDE}\)\s+is\s+(closer|farther)\s+than\s+"
        rf"the\s+([a-z_]+)\s*\({SIDE}\)",
        re.I,
    )
    m = pat2.search(t)
    if m:
        diff, clsA_raw, sideA, rel, clsB_raw, sideB = m.groups()
        clsA = cls(clsA_raw)
        clsB = cls(clsB_raw)
        objects = dedup([clsA, clsB])
        quantities.append([key(clsA, sideA), float(diff), "meters"])
        return {
            "sentence": text,
            "objects": objects,
            "attributes": [],
            "quantities": quantities,
            "bounding_boxes": [],
        }

    # Template 4：ego-distance differs from A → B by DIFF meters
    pat3 = re.compile(
        rf"ego[-\s]*distance\s+to\s+the\s+([a-z_]+)\s*\({SIDE}\)\s+"
        rf"differs\s+from\s+that\s+of\s+the\s+([a-z_]+)\s*\({SIDE}\)\s+"
        rf"by\s+([\d\.]+)\s*meters?",
        re.I,
    )
    m = pat3.search(t)
    if m:
        clsA_raw, sideA, clsB_raw, sideB, diff = m.groups()
        clsA = cls(clsA_raw)
        clsB = cls(clsB_raw)
        objects = dedup([clsA, clsB])

        quantities.append([key(clsB, sideB), float(diff), "meters"])
        return {
            "sentence": text,
            "objects": objects,
            "attributes": [],
            "quantities": quantities,
            "bounding_boxes": [],
        }


    # Template 5：A(sideA) and B(sideB) are separated by DIFF meters → distance_between
    pat5 = re.compile(
        rf"in\s+terms\s+of\s+range,\s+the\s+([a-z_]+)\s*\({SIDE}\)\s+and\s+"
        rf"([a-z_]+)\s*\({SIDE}\)\s+are\s+separated\s+by\s+([\d\.]+)\s*meters?",
        re.I,
    )
    m = pat5.search(t)
    if m:
        clsA_raw, sideA, clsB_raw, sideB, diff = m.groups()
        clsA = cls(clsA_raw)
        clsB = cls(clsB_raw)
        objects = dedup([clsA, clsB])
        quantities.append(["distance_between", float(diff), "meters"])
        return {
            "sentence": text,
            "objects": objects,
            "attributes": [],
            "quantities": quantities,
            "bounding_boxes": [],
        }

    objs = detect_classes_in_text(text)

    if len(objs) >= 2:
        main_cls = objs[0]
    elif len(objs) == 1:
        main_cls = objs[0]
    else:
        main_cls = "object"

    diff_val = extract_distance(text)   
    if diff_val is None:
        diff_val = 0.0  

    quantities.append([main_cls, diff_val, "meters"])

    return {
        "sentence": text,
        "objects": objs,
        "attributes": [],
        "quantities": quantities,
        "bounding_boxes": [],
    }

# Q7: Are there any pedestrians or bicycles?
def parse_Q7(text: str):

    t = normalize_text(text.lower())

    # 1. Detecting negation semantics
    neg_words = [
        r"\bno\b",
        r"\bnot\b",
        r"\bnone\b",
        r"\bnothing\b",
        r"\bnowhere\b",
        r"don['’]?t\s+see",
        r"doesn['’]?t\s+see",
        r"without\s+any",
        r"\black(?:ing)?\b",
        r"\babsence\s+of\b",
        r"\bno\s+sign\s+of\b",
    ]
    is_negative = any(re.search(p, t) for p in neg_words)

    # 2. Detect whether the target category appears in the text.
    classes = detect_classes_in_text(t)

    ped_present = "pedestrian" in classes
    bike_present = "bicycle" in classes

    # 3. Final Logic
    ped = 1 if ped_present and not is_negative else 0
    bike = 1 if bike_present and not is_negative else 0

    return {
        "sentence": text,
        "objects": ["pedestrian", "bicycle"],
        "attributes": [],
        "quantities": [
            ["pedestrian", ped],
            ["bicycle", bike],
        ],
        "bounding_boxes": [],
    }

# Q11: Most dangerous object + bbox + distance
def parse_Q11(text: str):

    obj = main_object(text)
    dist = extract_distance(text)
    boxes = extract_bboxes(text)
    attrs = [[obj, "hazardous"]]
    quants = []
    if dist is not None:
        quants.append([obj, dist, "meters"])
    return {
        "sentence": text,
        "objects": [obj],
        "attributes": attrs,
        "quantities": quants,
        "bounding_boxes": boxes,
    }

# Q12: overall risk level (high / medium / low)
def parse_Q12(text: str):

    t = normalize_text(text.lower())



    m = re.search(r"\b(high|medium|low)\b", t)
    if m:
        level = m.group(1)
    else:

        level = "medium"

    return {
        "sentence": text,
        "objects": ["risk_level"],
        "attributes": [["risk_level", level]],
        "quantities": [],
        "bounding_boxes": [],
    }


# Q13: Is it in an intersection scenario?
def parse_Q13(text: str):

    t = normalize_text(text.lower())
    yes = yes_no_from_text(text)
    if "intersection" in t and yes is None:
        if "not" in t or "isn’t" in t or "isn't" in t:
            yes = False
        else:
            yes = True
    val = 1 if yes else 0
    return {
        "sentence": text,
        "objects": ["intersection"],
        "attributes": [],
        "quantities": [["intersection", val]],
        "bounding_boxes": [],
    }

# Q14: Scene description (weather / road_type / lane / time / intersection / risk)
def parse_Q14(text: str):

    t = normalize_text(text.lower())
    objects = []
    attrs = []
    quants = []

    # weather
    m_weather = re.search(r"weather\s+is\s+([a-z]+)", t)
    if m_weather:
        w = m_weather.group(1)
        objects.append("weather")
        attrs.append(["weather", w])

    # time
    m_time = re.search(r"\bit\s+is\s+(daytime|nighttime|night|day|dawn|dusk)\b", t)
    if m_time:
        tm = m_time.group(1)
        objects.append("time")
        attrs.append(["time", tm])

    # road_type
    m_rt = re.search(r"road\s+type\s+is\s+([a-z_]+)", t)
    if not m_rt:
        m_rt = re.search(r"the\s+road\s+is\s+([a-z_]+)", t)
    if m_rt:
        rt = m_rt.group(1)
        objects.append("road_type")
        attrs.append(["road_type", rt])


    # lane count
    lanes = None

    # 1) road has 3 lanes / road has 1 lane
    m_lane = re.search(r"road\s+has\s+(\d+)\s+lanes?", t)
    if m_lane:
        lanes = int(m_lane.group(1))

    # 2) road has one lane / road has two lanes
    if lanes is None:
        for w, num in NUM_WORDS.items():
            if re.search(rf"road\s+has\s+{w}\s+lanes?", t):
                lanes = num
                break
            if re.search(rf"road\s+has\s+{w}\s+lane\b", t):   # one lane
                lanes = num
                break

    # 3) road has a lane / road has a single lane
    if lanes is None:
        if re.search(r"road\s+has\s+(a|single)\s+lane\b", t):
            lanes = 1

    # 4) it's a 3-lane road / a one-lane road
    if lanes is None:
        m_lane2 = re.search(r"(\d+)[- ]*lane\s+road", t)
        if m_lane2:
            lanes = int(m_lane2.group(1))

    if lanes is None:
        for w, num in NUM_WORDS.items():
            if re.search(rf"{w}[- ]*lane\s+road", t):
                lanes = num
                break

    if lanes is not None:
        objects.append("lane")
        quants.append(["lane", lanes])

    # intersection
    inter_val = None
    if "intersection" in t:
        if (
            "not an intersection" in t
            or "isn’t an intersection" in t
            or "isn't an intersection" in t
        ):
            inter_val = 0
        else:
            inter_val = 1
    if inter_val is not None:
        objects.append("intersection")
        quants.append(["intersection", inter_val])

    # risk level
    m_risk = re.search(r"(?:risk\s+level|the\s+risk|risk)\s+is\s+(high|medium|low)", t)
    if m_risk:
        lvl = m_risk.group(1)
        objects.append("risk_level")
        attrs.append(["risk_level", lvl])

    return {
        "sentence": text,
        "objects": dedup(objects),
        "attributes": attrs,
        "quantities": quants,
        "bounding_boxes": [],
    }


DEFAULT_TRAJ = [[0, 0, 0] for _ in range(6)]

# Q15: The future trajectory of ego
def parse_Q15(text: str):

    traj = extract_traj_points(text)

    if len(traj) != 6:
        traj = DEFAULT_TRAJ

    quants = [["trajectory_point", p, "position"] for p in traj]

    return {
        "sentence": text,
        "objects": [],
        "attributes": [],
        "quantities": quants,
        "bounding_boxes": [],
    }

# Q16: Same logic as Q15
def parse_Q16(text: str):

    return parse_Q15(text)

# Q17: Ego's position after 1 second
def parse_Q17(text: str):

    coord = extract_single_coords(text)
    quants = []
    if coord is not None:
        quants.append(["ego_vehicle", coord, "position"])
    return {
        "sentence": text,
        "objects": ["ego_vehicle"],
        "attributes": [],
        "quantities": quants,
        "bounding_boxes": [],
    }

# Q18: Ego's position after 2 seconds
def parse_Q18(text: str):

    coord = extract_single_coords(text)
    quants = []
    if coord is not None:
        quants.append(["ego_vehicle", coord, "position"])
    return {
        "sentence": text,
        "objects": ["ego_vehicle"],
        "attributes": [],
        "quantities": quants,
        "bounding_boxes": [],
    }

# Q19: Ego's position after 3 seconds
def parse_Q19(text: str):

    coord = extract_single_coords(text)
    quants = []
    if coord is not None:
        quants.append(["ego_vehicle", coord, "position"])
    return {
        "sentence": text,
        "objects": ["ego_vehicle"],
        "attributes": [],
        "quantities": quants,
        "bounding_boxes": [],
    }

# Q20: The distance between two objects
def parse_Q20(text: str):

    dist = extract_distance(text)
    quants = []
    if dist is not None:
        quants.append(["object_pair", dist, "meters"])
    return {
        "sentence": text,
        "objects": [],
        "attributes": [],
        "quantities": quants,
        "bounding_boxes": [],
    }

# Q21: Difference in azimuth between two objects
def parse_Q21(text: str):

    ang = extract_angle(text)
    quants = []
    if ang is not None:
        quants.append(["object_pair", ang, "degrees"])
    return {
        "sentence": text,
        "objects": [],
        "attributes": [],
        "quantities": quants,
        "bounding_boxes": [],
    }

# Q22: Absolute location + direction from the first to the second
def parse_Q22(text: str):

    bearing, direction = extract_bearing_and_dir(text)
    quants = []
    attrs = []
    if bearing is not None:
        quants.append(["object_pair", bearing, "degrees"])
    if direction:
        attrs.append(["object_pair", direction])
    return {
        "sentence": text,
        "objects": [],
        "attributes": attrs,
        "quantities": quants,
        "bounding_boxes": [],
    }


# Sub-posts & Main Portal
Q_HANDLERS = {
    "Q1": parse_Q1,
    "Q3": parse_Q3,
    "Q4": parse_Q4,
    "Q5": parse_Q5,
    "Q6": parse_Q6,
    "Q7": parse_Q7,
    "Q11": parse_Q11,
    "Q12": parse_Q12,
    "Q13": parse_Q13,
    "Q14": parse_Q14,
    "Q15": parse_Q15,
    "Q16": parse_Q16,
    "Q17": parse_Q17,
    "Q18": parse_Q18,
    "Q19": parse_Q19,
    "Q20": parse_Q20,
    "Q21": parse_Q21,
    "Q22": parse_Q22,
}


def parse_by_qtype(qtype: str, text: str):
    handler = Q_HANDLERS.get(qtype)
    if handler is None:
        return {
            "sentence": text,
            "objects": [],
            "attributes": [],
            "quantities": [],
            "bounding_boxes": [],
        }
    return handler(text)


# Clip-level processing & main program
def process_clip(clip_json: dict):

    # Multiple entries (entries with the same path prefix) are grouped into one clip based on the first two levels of the image_path
    ref_results = []
    model_results = []


    for item in clip_json.get("results", []):
        image_path = item.get("image_path", "").strip() 

        if not image_path:
            print(f"⚠️ Skipped entry due to missing or invalid image_path: {item}")
            continue  

        image_path = image_path.replace("/", os.sep).replace("\\", os.sep)

        path_parts = image_path.split(os.sep)

        file_name = os.path.splitext(path_parts[-1])[0]  # such as "00_00007_000031" -> "00_00007_00003"
        clip_prefix = os.path.join(path_parts[-3], path_parts[-2], file_name[:-1])  

        # ref_answer and model_answer
        q = item.get("question", "")
        qtype = item.get("question_type", "")
        ref_sentence = item.get("ref_answer", "") or ""
        model_sentence = item.get("model_answer", "") or ""

        if ref_sentence:
            parsed_ref = parse_by_qtype(qtype, ref_sentence)
            ref_results.append(
                {
                    "question": q,
                    "question_type": qtype,
                    **parsed_ref,
                }
            )

        if model_sentence:
            parsed_model = parse_by_qtype(qtype, model_sentence)
            model_results.append(
                {
                    "question": q,
                    "question_type": qtype,
                    **parsed_model,
                }
            )

    # Categorize all results by clip_prefix
    return {
        "clip_prefix": clip_prefix, 
        "ref_results": ref_results,
        "model_results": model_results,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Scene Semantic Graph Parser"
    )
    ap.add_argument("--input", default=INPUT_JSON_PATH, help="Path to input JSON file")
    ap.add_argument(
        "--output", default=OUTPUT_JSON_PATH, help="Path to save output JSON"
    )
    args = ap.parse_args()

    print(f"Read input file: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data if isinstance(data, list) else [data]

    print(f"A total of {len(clips)} VQA instances were found; parsing begins....")
    aggregated = {}

    for clip in clips:
        processed_clip = process_clip(clip)
        
        if processed_clip is None:
            continue

        clip_prefix = processed_clip["clip_prefix"]

        if clip_prefix not in aggregated:
            aggregated[clip_prefix] = {
                "ref_results": [],
                "model_results": []
            }

        aggregated[clip_prefix]["ref_results"].extend(processed_clip["ref_results"])
        aggregated[clip_prefix]["model_results"].extend(processed_clip["model_results"])

    print(f"Parsing complete: {len(aggregated)} clips")


    with open(args.output, "w", encoding="utf-8") as f:
        f.write("[\n")

        for i, (clip_prefix, clip_data) in enumerate(aggregated.items()):

            # Merge all data
            clip = {
                "clip_prefix": clip_prefix,
                "ref_results": clip_data["ref_results"],
                "model_results": clip_data["model_results"]
            }
        
            json_str = json.dumps(clip, ensure_ascii=False, indent=2)

            json_str = re.sub(
                r'\[\s*((?:[-+]?[\d\.Ee]+(?:\s*,\s*[-+]?[\d\.Ee]+){1,10}))\s*\]',
                lambda m: "[" + ", ".join(
                    x.strip() for x in re.split(
                        r'\s*,\s*',
                        re.sub(r'\s+', ' ', m.group(1)).strip()
                    )
                ) + "]",
                json_str
            )

            json_str = re.sub(
                r'\[\s+("[^"\n]+?"(?:\s*,\s*[^,\]\n]+){0,3})\s+\]',
                lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]",
                json_str
            )

            # compression ["bounding_box", [ ... ], "normalized"]
            json_str = re.sub(
                r'\[\s+("[^"\n]+?"\s*,\s*\[[^\]]+\]\s*,\s*"[^"]+"\s*)\]',
                lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]",
                json_str
            )

            f.write("    " + json_str.replace("\n", "\n    "))
            if i < len(aggregated) - 1:
                f.write(",\n")
            else:
                f.write("\n")

        f.write("]")



if __name__ == "__main__":
    main()
