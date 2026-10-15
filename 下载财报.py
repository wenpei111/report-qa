# -*- coding: utf-8 -*-
"""
下载财报.py —— 作业A 第0步：从巨潮资讯下载 10 家白酒公司最近半年报全文 PDF
用法: python 下载财报.py
输出: D:\\财报问答作业\\数据源\\<代码>_<公司>_<报告期>.pdf
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "数据源"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 家白酒公司
COMPANIES = [
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("000568", "泸州老窖"),
    ("600809", "山西汾酒"),
    ("002304", "洋河股份"),
    ("000596", "古井贡酒"),
    ("603369", "今世缘"),
    ("603198", "迎驾贡酒"),
    ("600702", "舍得酒业"),
    ("600779", "水井坊"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "http://www.cninfo.com.cn/",
}

# 报告期偏好：优先最新半年报，取不到顺延上一年
YEAR_PREFS = ["2026", "2025"]


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_post(url, data, timeout=30):
    body = urllib.parse.urlencode(data).encode("utf-8")
    headers = dict(HEADERS, **{"Content-Type": "application/x-www-form-urlencoded"})
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def search_announcements(name, code):
    """巨潮全文检索接口，搜公司名，返回公告列表"""
    url = ("http://www.cninfo.com.cn/new/fulltextSearch/full?searchkey="
           + urllib.parse.quote(name)
           + "&sdate=&edate=&isfulltext=false&sortName=pubdate&sortType=desc&pageNum=1&pageSize=30")
    raw = http_get(url)
    return json.loads(raw.decode("utf-8")).get("announcements") or []


def pick_halfyear_report(anns, code):
    """从公告列表里挑出半年报全文（排除摘要/英文/取消/更正前等），优先最新年份"""
    candidates = []
    for a in anns:
        if a.get("secCode") != code:
            continue
        title = a.get("announcementTitle") or ""
        if "半年度报告" not in title and "半年报" not in title:
            continue
        # 排除摘要、英文版、取消、更正
        if any(k in title for k in ("摘要", "英文", "取消", "更正", "提示")):
            continue
        if not a.get("adjunctUrl"):
            continue
        m = re.search(r"(20\d{2})年?半年度", title)
        year = m.group(1) if m else ""
        candidates.append((year, a.get("announcementTime", 0), title, a["adjunctUrl"]))
    if not candidates:
        return None
    # 按 YEAR_PREFS 顺序，再按时间倒序
    for y in YEAR_PREFS:
        pool = [c for c in candidates if c[0] == y]
        if pool:
            pool.sort(key=lambda c: c[1], reverse=True)
            return pool[0]
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates[0]


def download_pdf(adjunct_url, save_path):
    url = "http://static.cninfo.com.cn/" + adjunct_url
    data = http_get(url, timeout=120)
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"返回内容不是 PDF（前 8 字节: {data[:8]!r}）")
    save_path.write_bytes(data)
    return len(data)


def main():
    print(f"输出目录: {OUT_DIR}")
    ok, fail = [], []
    for code, name in COMPANIES:
        save_path = OUT_DIR / f"{code}_{name}_半年报.pdf"
        if save_path.exists() and save_path.stat().st_size > 100_000:
            print(f"[跳过] {name} 已存在: {save_path.name}")
            ok.append((code, name, save_path.name))
            continue
        print(f"[查询] {name}({code}) ...", flush=True)
        try:
            anns = search_announcements(name, code)
            pick = pick_halfyear_report(anns, code)
            if not pick:
                fail.append((code, name, "未检索到半年报"))
                print(f"  !! 未检索到半年报")
                continue
            year, _, title, adjunct = pick
            size = download_pdf(adjunct, save_path)
            print(f"  OK {title} -> {save_path.name} ({size/1024/1024:.1f} MB)")
            ok.append((code, name, save_path.name))
            time.sleep(1.5)  # 温和限速
        except Exception as e:
            fail.append((code, name, str(e)))
            print(f"  !! 失败: {e}")

    print("\n===== 结果 =====")
    print(f"成功 {len(ok)} / {len(COMPANIES)}")
    for c in ok:
        print(f"  {c[0]} {c[1]}: {c[2]}")
    if fail:
        print(f"失败 {len(fail)}:")
        for c in fail:
            print(f"  {c[0]} {c[1]}: {c[2]}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
