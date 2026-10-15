# -*- coding: utf-8 -*-
"""
评测.py —— 作业A 第4步：10 道题逐题跑，记录召回了哪些块、答对没有、错在哪
- 至少 2 道跨公司全景题（作业硬要求）
- 自动判定：答案里是否出现标准数字/结论关键词；判不了的标"人工复核"
输出: 04_评测\\评测结果.json + 评测记录.md
"""
import json
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "03_问答"))
import 服务 as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT_JSON = Path(__file__).resolve().parent / "评测结果.json"
OUT_MD = Path(__file__).resolve().parent / "评测记录.md"

# (题号, 问题, 类型, 判定要点: {关键词/数字}, 标准答案说明)
QUESTIONS = [
    ("Q1", "贵州茅台2026上半年营业收入是多少元？", "单公司·主要会计数据",
     ["90,703,260,964.48"], "90,703,260,964.48元（+1.47%）"),
    ("Q2", "五粮液2026上半年归属于上市公司股东的净利润是多少元？", "单公司·主要会计数据",
     ["8,752,942,991.31"], "8,752,942,991.31元（+89.30%）"),
    ("Q3", "泸州老窖2026上半年营业收入是多少？同比变化如何？", "单公司·表格+变化",
     ["10,472,224,575.78", "36.35"], "104.72亿元，同比-36.35%"),
    ("Q4", "茅台2026上半年直销渠道收入是多少？", "单公司·渠道表",
     ["52,006,683,587.88"], "520.07亿元（直销渠道表）"),
    ("Q5", "山西汾酒2026上半年营业收入是多少？", "单公司·主要会计数据",
     ["21,044,387,808.34"], "210.44亿元（-12.18%）"),
    ("Q6", "洋河股份2026上半年营业收入是多少？", "单公司·主要会计数据",
     ["10,540,431,576.04"], "105.40亿元"),
    ("Q7", "迎驾贡酒2026上半年稀释每股收益是多少？", "单公司·主要财务指标",
     ["1.45"], "1.45元/股"),
    ("Q8", "五粮液和泸州老窖2026上半年营业收入哪家更高？", "跨公司·两方比较",
     ["五粮液"], "五粮液284.17亿 > 泸州老窖104.72亿"),
    ("Q9", "茅台、五粮液、汾酒2026上半年营业收入从高到低怎么排？", "跨公司·三方排序",
     ["茅台", "五粮液", "汾酒"], "茅台907亿 > 五粮液284亿 > 汾酒210亿"),
    ("Q10", "10家白酒公司2026上半年营业收入最高的是哪家？", "全景·跨10家",
     ["茅台"], "贵州茅台907亿（10家最高）"),
]


def normalize(s):
    return s.replace(",", "").replace("，", "").replace(" ", "")


def judge(answer_text, keys):
    ans = normalize(answer_text)
    hits = []
    for k in keys:
        k2 = normalize(k)
        if k2 in ans:
            hits.append(k)
            continue
        # 数字按"亿"换算再试：90,703,260,964.48 -> 907.03亿
        m = re.fullmatch(r"([\d\.]+)", k2)
        if m and ("亿" in ans):
            try:
                yi = float(m.group(1)) / 1e8
                if f"{yi:.2f}亿" in ans or f"{yi:.1f}亿" in ans or f"{yi:.0f}亿" in ans:
                    hits.append(k)
            except ValueError:
                pass
    if not hits:
        return "✗ 未命中标准答案（需人工复核是答案错还是判定过严）"
    if len(hits) == len(keys):
        return "✓ 对"
    return "△ 部分命中"


def main():
    env = S.load_env()
    retriever = S.Retriever()
    results = []
    for qid, q, typ, keys, truth in QUESTIONS:
        t0 = time.time()
        r = S.answer_question(q, retriever, env)
        verdict = judge(r["answer"], keys)
        used = [s for s in r["sources"] if s["被引用"]]
        results.append({"题号": qid, "问题": q, "类型": typ, "答案": r["answer"],
                        "判定": verdict, "标准答案": truth,
                        "召回块": [{k: s[k] for k in ("id", "公司", "章节", "页码", "类型", "命中", "被引用")}
                                  for s in r["sources"]],
                        "引用块": [s["id"] for s in used],
                        "公司覆盖": sorted({s["公司"] for s in r["sources"]}),
                        "用时": r["耗时"]})
        print(f"{qid} [{typ}] {verdict}  ({r['耗时']}s)")
        print(f"   召回覆盖公司: {sorted({s['公司'] for s in r['sources']})}")
        print(f"   答案: {r['answer'][:80]}...")
        time.sleep(1)

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # 汇总 markdown
    lines = ["# 评测记录（10 题）\n",
             f"评测时间：{time.strftime('%Y-%m-%d %H:%M')} · 3,321 块 · 向量+BM25 双路 · deepseek-chat\n",
             "| 题号 | 类型 | 判定 | 用时 | 引用块 |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['题号']} | {r['类型']} | {r['判定']} | {r['用时']}s | {','.join(r['引用块']) or '—'} |")
    lines.append("\n## 逐题详情\n")
    for r in results:
        lines.append(f"### {r['题号']} {r['问题']}\n")
        lines.append(f"- **类型**：{r['类型']}　**判定**：{r['判定']}")
        lines.append(f"- **标准答案**：{r['标准答案']}")
        lines.append(f"- **系统回答**：{r['答案']}")
        lines.append(f"- **召回块**（公司覆盖：{'、'.join(r['公司覆盖'])}）：")
        for b in r["召回块"]:
            mark = "●引用" if b["被引用"] else "○"
            lines.append(f"  - {mark} [{b['id']}] {b['公司']} {b['章节']} p{b['页码']} {b['类型']} ({b['命中']})")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    ok = sum(1 for r in results if r["判定"].startswith("✓"))
    part = sum(1 for r in results if r["判定"].startswith("△"))
    print(f"\n===== 汇总：✓对 {ok} / △部分 {part} / ✗错 {len(results)-ok-part} =====")
    print(f"明细 -> {OUT_MD}")


if __name__ == "__main__":
    main()
