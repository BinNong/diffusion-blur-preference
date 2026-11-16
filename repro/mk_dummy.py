"""构造一个"空格子"数据文件：把基线（bdense+blow）合并成单个 JSON。

用途：analyze_model.py 的**零检验**。令 --cellB 和 --cellC 都指向它，
则 B−C、C−A、B−A 必须全部精确为 0，M2/M3/M4 必须全部读"测不出"。
若读不出 0，说明裁判自己有 bug（配对/bootstrap/网格对齐），此时任何真实结论都不可信。

用法： venv2/bin/python mk_dummy.py
"""

import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

rows = []
cfg = None
for fn in ["sweep_bdense.json", "sweep_blow.json"]:
    d = json.load(open(os.path.join(OUT, fn)))
    if cfg is None:
        cfg = dict(d["cfg"])
    rows.extend(d["rows"])

# 去重：同一个 (fid, kunif) 只留**最后**一条。
# 必须与裁判的内存合并语义一致：analyze_bdense/analyze_bt/analyze_model 都是
# `pd.setdefault(fid,[]).extend(blow_rows)` 然后用 dict 按键覆盖，
# 所以** blow 赢**。mk_dummy 若留第一条（bdense 赢），零检验就会给出
# b∞=-0.681 vs -0.666 的假差异——那是两轮独立 Hutchinson 对同一算子的测量差。
seen = set()
uniq = []
for r in reversed(rows):
    k = (r["meta"]["fid"], round(r["kunif"], 6))
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)
uniq = list(reversed(uniq))

p = os.path.join(OUT, "sweep_zdummy.json")
json.dump(dict(cfg=cfg, rows=uniq), open(p, "w"))
print(f"[saved] {p}: {len(uniq)} 算子（去重前 {len(rows)}）")
print(" 期望 analyze_model.py --cellB sweep_zdummy.json --cellC sweep_zdummy.json")
print("   给出 B−C = 0、C−A = 0、B−A = 0，且 M2/M3/M4 全部'测不出'。")
