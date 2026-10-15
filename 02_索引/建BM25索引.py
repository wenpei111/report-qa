# -*- coding: utf-8 -*-
"""
建BM25索引.py —— 作业A 第2步（本地半边）：对 3364 块建 BM25 索引
纯 Python 实现（中文用二字元切词），零第三方依赖。
输出: 02_索引\\bm25_index.json
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
CHUNKS = BASE / "01_解析切块" / "文本块.json"
OUT = Path(__file__).resolve().parent / "bm25_index.json"

K1, B = 1.5, 0.75


def tokenize(text):
    """中文按二字元（bigram），连续字母/数字按词，兼顾公司名、代码、金额"""
    tokens = []
    for word in re.findall(r"[A-Za-z0-9\.%]+|[\u4e00-\u9fff]", text):
        if re.match(r"[\u4e00-\u9fff]", word):
            tokens.append(word)  # 单字
        else:
            tokens.append(word.lower())
    # 中文 bigram：把连续汉字串拆开
    han_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for run in han_runs:
        for i in range(len(run) - 1):
            tokens.append(run[i:i + 2])
    return tokens


def main():
    t0 = time.time()
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    ids = [c["id"] for c in chunks]
    docs = [tokenize(f"{c['公司']} {c['章节']} {c['文本']}") for c in chunks]
    doclen = [len(d) for d in docs]
    N = len(docs)
    avgdl = sum(doclen) / N

    # postings: term -> {doc_idx: tf}
    postings = {}
    for i, d in enumerate(docs):
        for term, tf in Counter(d).items():
            postings.setdefault(term, {})[i] = tf

    index = {
        "k1": K1, "b": B, "N": N, "avgdl": avgdl,
        "ids": ids, "doclen": doclen,
        "postings": {t: list(p.items()) for t, p in postings.items()},
    }
    OUT.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"文档 {N} 篇 | 词项 {len(postings)} 个 | avgdl {avgdl:.0f} | "
          f"{size_mb:.1f} MB | {time.time()-t0:.1f}s -> {OUT.name}")

    # 自检：跑两个查询看召回是否合理
    for q in ["茅台 直销 收入", "经销商 数量 变化"]:
        qt = tokenize(q)
        score = Counter()
        for term in qt:
            if term not in postings:
                continue
            idf = __import__("math").log(1 + (N - len(postings[term]) + 0.5) / (len(postings[term]) + 0.5))
            for i, tf in postings[term].items():
                s = idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doclen[i] / avgdl))
                score[i] += s
        top = score.most_common(3)
        print(f"\n自检查询「{q}」:")
        for i, s in top:
            c = chunks[i]
            print(f"  {s:6.2f} {c['id']} {c['公司']} {c['章节']} p{c['页码']} | {c['文本'][:50]}...")


if __name__ == "__main__":
    main()
