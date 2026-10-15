# -*- coding: utf-8 -*-
"""
解析切块.py —— 作业A 第1步：半年报 PDF → 文本块 JSON
- 文字：pdfplumber 提取（表格区域的对象先剔除，避免重复）
- 表格：按行列还原（每个表格 = 若干行，行内用 " | " 连接）
- 切块：带 公司 / 章节 / 页码 三元数据（课件硬要求）
输出: 01_解析切块\\文本块.json
"""
import json
import re
import sys
from pathlib import Path

import pdfplumber

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "数据源"
OUT = Path(__file__).resolve().parent / "文本块.json"

# 章节标题：第X节，或 上一节遗留下的"第X节：xxx"式标题
HEADING_RE = re.compile(r"^第[一二三四五六七八九十]+节[\s：:].{0,40}$")
# 目录页整段跳过没有意义，正文里的目录行形如 "第一节 释义 ...... 4"，不当作章节标题
TOC_LINE_RE = re.compile(r"第[一二三四五六七八九十]+节.{2,40}\.{4,}\s*\d+")

CHUNK_TARGET = 600   # 目标块长（字符）
CHUNK_MAX = 900      # 硬上限
ROW_GROUP = 20       # 大表格按行分组，每组行数


def strip_headers_footers(text, company):
    """剥掉每页重复的页眉/页脚：公司名+报告标题行、页码行"""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if company in s and "半年度报告" in s and len(s) < 45:
            continue  # 页眉：如 "宜宾五粮液股份有限公司2026年半年度报告全文"
        if re.fullmatch(r"第\s*\d+\s*页(\s*共\s*\d+\s*页)?", s):
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue  # 纯页码
        out.append(s)
    return "\n".join(out)


def page_objects_outside_tables(page, table_bboxes):
    """剔除落在表格 bbox 内的对象，再用 extract_text 提纯文字"""
    def not_in_table(obj, bboxes=table_bboxes):
        x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
        cx, cy = (x0 + x1) / 2, (top + bottom) / 2
        return not any(bx0 <= cx <= bx1 and btop <= cy <= bbottom
                       for bx0, btop, bx1, bbottom in bboxes)
    filtered = page.filter(not_in_table)
    return filtered.extract_text() or ""


def table_to_rows(table):
    """表格按行列还原：去空行、每行单元格 ' | ' 连接"""
    rows = []
    for row in table.extract():
        cells = [re.sub(r"\s+", " ", (c or "")).strip() for c in row]
        if not any(cells):
            continue
        rows.append(" | ".join(cells))
    return rows


def make_chunks(company, code, pages):
    """pages: [(页码, 类型, 文本)] -> 切块"""
    chunks = []
    buf, buf_page, buf_sec = [], None, "封面/目录"
    cur_sec = "封面/目录"

    def flush():
        nonlocal buf, buf_page, buf_sec
        if not buf:
            return
        text = "\n".join(buf).strip()
        if len(text) >= 30:  # 太碎的丢弃
            chunks.append({"公司": company, "代码": code, "章节": buf_sec,
                           "页码": buf_page, "类型": "文本", "文本": text})
        buf, buf_page, buf_sec = [], None, cur_sec

    for page_no, kind, content in pages:
        if kind == "表格":
            flush()
            # 大表格按行分组成多块
            rows = content.split("\n")
            for i in range(0, len(rows), ROW_GROUP):
                part = "\n".join(rows[i:i + ROW_GROUP])
                if len(part) >= 30:
                    chunks.append({"公司": company, "代码": code, "章节": cur_sec,
                                   "页码": page_no, "类型": "表格", "文本": part})
            continue
        # 文本：按段落累积
        for para in content.split("\n"):
            para = para.strip()
            if not para or TOC_LINE_RE.search(para):
                continue
            if HEADING_RE.match(para):
                flush()
                cur_sec = re.sub(r"^第([一二三四五六七八九十]+)节[\s：:]*", r"第\1节 ", para)[:40]
                buf.append(para)
                buf_page, buf_sec = page_no, cur_sec
                continue
            buf.append(para)
            if buf_page is None:
                buf_page = page_no
            if sum(len(p) for p in buf) >= CHUNK_TARGET:
                flush()
    flush()
    # 超长块硬切（罕见，保底）
    final = []
    for c in chunks:
        if len(c["文本"]) <= CHUNK_MAX:
            final.append(c)
            continue
        text = c["文本"]
        for i in range(0, len(text), CHUNK_MAX):
            seg = text[i:i + CHUNK_MAX]
            if len(seg) >= 30:
                final.append({**c, "文本": seg})
    return final


def main():
    all_chunks = []
    stats = []
    pdfs = sorted(SRC.glob("*.pdf"))
    print(f"输入: {len(pdfs)} 份 PDF")
    for pdf in pdfs:
        m = re.match(r"(\d+)_(.+?)_半年报", pdf.stem)
        code, company = (m.group(1), m.group(2)) if m else ("", pdf.stem)
        pages = []
        n_tables = 0
        with pdfplumber.open(pdf) as doc:
            for i, page in enumerate(doc.pages, start=1):
                tables = page.find_tables()
                bboxes = [t.bbox for t in tables]
                # 先处理文字（让章节标题先更新），再处理表格：表格的章节归属才准确
                text = page_objects_outside_tables(page, bboxes)
                text = strip_headers_footers(text, company)
                if text.strip():
                    pages.append((i, "文本", text))
                if bboxes:
                    rows = []
                    for t in tables:
                        rows.extend(table_to_rows(t))
                    if rows:
                        pages.append((i, "表格", "\n".join(rows)))
                        n_tables += 1
        chunks = make_chunks(company, code, pages)
        all_chunks.extend(chunks)
        n_tab = sum(1 for c in chunks if c["类型"] == "表格")
        stats.append((company, len(doc.pages) if False else "ok", len(chunks), n_tab))
        print(f"  {company}: {len(chunks)} 块（其中表格块 {n_tab}，含表页 {n_tables}）")

    # 统一编号
    for idx, c in enumerate(all_chunks):
        c["id"] = f"F{idx:05d}"

    OUT.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n共 {len(all_chunks)} 块 -> {OUT}")
    lens = sorted(len(c["文本"]) for c in all_chunks)
    print(f"块长: 中位 {lens[len(lens)//2]} / 最大 {lens[-1]} / 最小 {lens[0]}")


if __name__ == "__main__":
    main()
