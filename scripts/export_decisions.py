"""R7: export a sanitized decision-level dataset from the API cache so the live-vs-extraction analyses can be rerun
without the cache. Columns: source (live_v1/live_v2/extraction), pair (A,B), option_order (1 = B listed first),
state (base-4 code of the 5-record memory, oldest first; record = 2*myB + partnerB), choice_B (0/1),
cm, seed, round, agent (live) or k (extraction). No prompts, responses or keys are included."""
import csv, gzip, json, re, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.envs.naming_game import parse_value
WP = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]
REC = re.compile(r"- You chose (\w), partner chose (\w)"); OPT = re.compile(r"from: \[(\w), (\w)\]")
con = sqlite3.connect("data/cache/llm_cache.sqlite"); out = Path("results/decisions"); out.mkdir(parents=True, exist_ok=True)
def row(q, text, A, B):
    recs = REC.findall(q["input"]); o = OPT.search(q["instructions"] or "")
    v = parse_value(text, (A, B))
    if len(recs) != 5 or not o or v is None: return None
    return int(o.group(1) == B), int("".join(str(2 * (m == B) + (p == B)) for m, p in recs), 4), int(v == B)
n = 0
with gzip.open(out / "deepseek_decisions.csv.gz", "wt", newline="") as fh:
    w = csv.writer(fh); w.writerow(["source", "pair", "option_order", "state", "choice_B", "cm", "seed", "round", "agent", "k"])
    for req, text in con.execute("select request, text from cache where request like '%\"tag\": \"live_quench/%' or request like '%\"tag\": \"live_token/%' or request like '%\"tag\": \"policy/deepseek-v4-flash/%'"):
        q = json.loads(req); tag = q["tag"]
        if tag.startswith("policy/"):
            m = re.match(r"policy/deepseek-v4-flash/(\d{5})/k(\d+)", tag); A, B = WP[int(m.group(2)) % 6]; r = row(q, text, A, B)
            if r: w.writerow(["extraction", A + B, *r, "", "", "", "", m.group(2)]); n += 1
        else:
            if "/try0" not in tag: continue
            m = re.search(r"/cm(\d+)/(?:w(\w)(\w)/)?s(\d+)/t(\d+)/a(\d+)/try0", tag)
            if not m: continue
            A, B = (m.group(2), m.group(3)) if m.group(2) else WP[int(m.group(4)) % 6]; r = row(q, text, A, B)
            if r: w.writerow(["live_v2" if "live_token" in tag else "live_v1", A + B, *r, m.group(1), m.group(4), m.group(5), m.group(6), ""]); n += 1
print("rows", n)
