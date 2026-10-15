# -*- coding: utf-8 -*-
"""
建向量索引.py —— 作业A 第2步（云端半边）：调用阿里云百炼 embedding 向量化 3364 块
- 模型: text-embedding-v3（1024 维），OpenAI 兼容接口
- key 从 .env 读（绝不硬编码）
输出: 02_索引\\向量.bin（float32 顺序存放） + 02_索引\\向量_meta.json
"""
import json
import sys
import time
import urllib.request
from array import array
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
CHUNKS = BASE / "01_解析切块" / "文本块.json"
OUT_BIN = Path(__file__).resolve().parent / "向量.bin"
OUT_META = Path(__file__).resolve().parent / "向量_meta.json"
ENV = BASE / ".env"

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
MODEL = "text-embedding-v3"
DIM = 1024
BATCH = 10          # 接口单次上限 10 条
MAX_RETRY = 3


def load_env():
    if not ENV.exists():
        sys.exit("[停止] 找不到 .env —— 请复制 .env.example 为 .env 并填入 DASHSCOPE_API_KEY")
    env = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    if not env.get("DASHSCOPE_API_KEY") or "你的" in env.get("DASHSCOPE_API_KEY", ""):
        sys.exit("[停止] .env 里 DASHSCOPE_API_KEY 还没填好")
    return env


def embed_batch(key, texts):
    body = json.dumps({"model": MODEL, "input": texts, "dimensions": DIM,
                       "encoding_format": "float"}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = sorted(data["data"], key=lambda x: x["index"])
            return [it["embedding"] for it in items]
        except Exception as e:  # 网络抖动/限流退避重试
            last_err = e
            wait = 5 * attempt
            print(f"    第{attempt}次失败: {e}，{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"embedding 请求连续 {MAX_RETRY} 次失败: {last_err}")


def main():
    t0 = time.time()
    key = load_env()["DASHSCOPE_API_KEY"]
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    vec_path = OUT_BIN
    done = 0
    if vec_path.exists() and OUT_META.exists():
        meta = json.loads(OUT_META.read_text(encoding="utf-8"))
        done = meta["done"]
        print(f"断点续跑: 已完成 {done}/{len(chunks)}")
    elif vec_path.exists():
        vec_path.unlink()

    f = open(vec_path, "ab" if done else "wb")
    try:
        for start in range(done, len(chunks), BATCH):
            batch = chunks[start:start + BATCH]
            texts = [f"{c['公司']} {c['章节']} {c['文本'][:1000]}" for c in batch]
            embs = embed_batch(key, texts)
            for e in embs:
                if len(e) != DIM:
                    raise RuntimeError(f"维度异常: {len(e)} != {DIM}")
                f.write(array("f", e).tobytes())
            f.flush()
            n = start + len(batch)
            if n % 100 < BATCH or n == len(chunks):
                print(f"  {n}/{len(chunks)}  用时 {time.time()-t0:.0f}s")
    finally:
        f.close()
    OUT_META.write_text(json.dumps({"done": len(chunks), "model": MODEL,
                                    "dim": DIM}, ensure_ascii=False), encoding="utf-8")
    print(f"完成 {len(chunks)} 块 x {DIM} 维 -> {vec_path.name} "
          f"({vec_path.stat().st_size/1024/1024:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
