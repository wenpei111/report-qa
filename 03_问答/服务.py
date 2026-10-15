# -*- coding: utf-8 -*-
"""
服务.py —— 作业A 第3步：财报问答（双路检索 + DeepSeek 生成 + 出处角标）
用法:
  python 服务.py                 # 启动网页服务 http://127.0.0.1:8787
  python 服务.py --ask "问题"    # 命令行问答（自检用）
key 全部从 .env 读取，代码零硬编码。
"""
import json
import math
import re
import sys
import time
import urllib.request
from array import array
from collections import Counter
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE = Path(__file__).resolve().parent.parent
CHUNKS = BASE / "01_解析切块" / "文本块.json"
BM25 = BASE / "02_索引" / "bm25_index.json"
VECBIN = BASE / "02_索引" / "向量.bin"
VECMETA = BASE / "02_索引" / "向量_meta.json"
ENV = BASE / ".env"

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
EMBED_MODEL, DIM = "text-embedding-v3", 1024
CHAT_MODEL = "deepseek-chat"
TOP_VEC, TOP_BM25, TOP_FINAL = 12, 12, 8
PER_COMPANY_MIN = 3   # 问题里点名的公司，每家至少保证进结果的块数
W_VEC, W_BM25 = 0.6, 0.4

# 简称→公司：问题里写"茅台/汾酒/洋河"也能对上公司（用于跨公司保障）
ALIASES = {
    "贵州茅台": ["茅台", "i茅台"],
    "山西汾酒": ["汾酒", "青花瓷", "杏花村"],
    "洋河股份": ["洋河", "梦之蓝", "海之蓝"],
    "古井贡酒": ["古井贡", "古井", "年份原浆"],
    "迎驾贡酒": ["迎驾"],
    "舍得酒业": ["舍得"],
}

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_env():
    env = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


# ---------- 检索 ----------
class Retriever:
    def __init__(self):
        self.chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
        idx = json.loads(BM25.read_text(encoding="utf-8"))
        self.k1, self.b, self.N, self.avgdl = idx["k1"], idx["b"], idx["N"], idx["avgdl"]
        self.doclen = idx["doclen"]
        self.postings = {t: {int(i): tf for i, tf in lst} for t, lst in idx["postings"].items()}
        self.vecs = array("f")
        self.vecs.frombytes(VECBIN.read_bytes())
        assert len(self.vecs) == self.N * DIM, "向量文件与文本块数不一致"

    def tokenize(self, text):
        tokens = []
        for word in re.findall(r"[A-Za-z0-9\.%]+|[\u4e00-\u9fff]", text):
            tokens.append(word if not re.match(r"[\u4e00-\u9fff]", word) else word)
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        return tokens

    def bm25_scores(self, q_tokens):
        score = Counter()
        for term in q_tokens:
            p = self.postings.get(term)
            if not p:
                continue
            idf = math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
            for i, tf in p.items():
                score[i] += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.doclen[i] / self.avgdl))
        return score

    def vector_scores(self, q_vec):
        # 归一化查询向量
        norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0
        qv = [x / norm for x in q_vec]
        score = Counter()
        for i in range(self.N):
            base = i * DIM
            s = 0.0
            for j in range(DIM):
                s += qv[j] * self.vecs[base + j]
            score[i] = s  # 块向量建库时未归一化，先存点积
        # 用每篇自身的模长修正成余弦
        for i in score:
            m = math.sqrt(sum(x * x for x in self.vecs[i * DIM:(i + 1) * DIM])) or 1.0
            score[i] /= m
        return score

    def search(self, question, env):
        q_tokens = self.tokenize(question)
        q_vec = self._embed(question, env["DASHSCOPE_API_KEY"])
        s_vec = self.vector_scores(q_vec)
        s_bm = self.bm25_scores(q_tokens)
        def norm_scores(counter, topk):
            if not counter:
                return {}
            top = counter.most_common(topk)
            mx = max((v for _, v in top), default=1.0) or 1.0
            return {i: v / mx for i, v in top}
        nv, nb = norm_scores(s_vec, TOP_VEC), norm_scores(s_bm, TOP_BM25)
        merged = Counter()
        for i, v in nv.items():
            merged[i] += W_VEC * v
        for i, v in nb.items():
            merged[i] += W_BM25 * v
        top = merged.most_common()
        # 过滤掉封面/目录块（几乎无信息，还常被误引用）
        top = [(i, s) for i, s in top if self.chunks[i]["章节"] != "封面/目录"]
        score_of = dict(top)
        # 跨公司保障：问题里点名的公司，各取"该公司内部 BM25 最优"的块，
        # 再用全局最优补位（全局榜单一家公司会垄断，捞不到别家的关键表）
        comp_idx = {}
        for i in range(self.N):
            comp_idx.setdefault(self.chunks[i]["公司"], []).append(i)
        mentioned = [comp for comp in comp_idx
                     if comp in question or any(a in question for a in ALIASES.get(comp, []))]
        selected = []
        for comp in mentioned:
            comp_set = {i for i in comp_idx.get(comp, [])
                        if self.chunks[i]["章节"] != "封面/目录"}
            # 公司内部双路混合：BM25 + 向量各归一化后加权，避免单路被注释表格的高词频压过
            comp_bm = Counter({i: v for i, v in s_bm.items() if i in comp_set})
            comp_ve = Counter({i: v for i, v in s_vec.items() if i in comp_set})
            comp_scores = Counter()
            if comp_bm:
                mx = max(comp_bm.values())
                for i, v in comp_bm.items():
                    comp_scores[i] += 0.5 * v / mx
            if comp_ve:
                mx = max(comp_ve.values())
                for i, v in comp_ve.items():
                    comp_scores[i] += 0.5 * v / mx
            if not comp_scores:
                comp_scores = Counter({i: v for i, v in merged.items()
                                       if self.chunks[i]["公司"] == comp})
            for i, _ in comp_scores.most_common(PER_COMPANY_MIN):
                if i not in selected:
                    selected.append(i)
        for i, s in top:
            if len(selected) >= TOP_FINAL:
                break
            if i not in selected:
                selected.append(i)
        results = []
        for i in selected:
            c = self.chunks[i]
            results.append({**c, "得分": round(score_of.get(i, 0.0), 4),
                            "命中": ("向量" if i in nv else "") + ("+BM25" if i in nb else "")})
        return results

    def _embed(self, text, key):
        body = json.dumps({"model": EMBED_MODEL, "input": [text], "dimensions": DIM,
                           "encoding_format": "float"}).encode("utf-8")
        req = urllib.request.Request(DASHSCOPE_URL, data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["data"][0]["embedding"]


# ---------- 生成 ----------
SYSTEM_PROMPT = """你是财报问答助手。规则：
1. 只依据【材料】回答，禁止使用材料之外的任何知识；
2. 每个关键信息句末尾必须带角标 [F0xxxx]，编号只能取自材料给出的块编号，一句可带多个角标；
3. 材料不足以回答的部分，明确写"材料未覆盖"，不要编造；
4. 数字必须与材料原文一致，不要改写单位或四舍五入；
5. 回答用简体中文，简洁分点。"""


def answer_question(question, retriever, env):
    t0 = time.time()
    sources = retriever.search(question, env)
    ctx = "\n\n".join(
        f"[{s['id']}] {s['公司']} {s['章节']} 第{s['页码']}页\n{s['文本']}" for s in sources)
    body = json.dumps({
        "model": CHAT_MODEL, "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【材料】\n{ctx}\n\n【问题】{question}"},
        ]}).encode("utf-8")
    req = urllib.request.Request(DEEPSEEK_URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answer = data["choices"][0]["message"]["content"]
    used = set(re.findall(r"\[(F\d{5})\]", answer))
    for s in sources:
        s["被引用"] = s["id"] in used
    return {"question": question, "answer": answer, "sources": sources,
            "耗时": round(time.time() - t0, 1)}


# ---------- 网页 ----------
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>财报问答知识库 · 作业A</title>
<style>
  :root { --blue:#1a66cc; --bg:#f5f7fa; --card:#fff; --line:#e3e8ef; }
  * { box-sizing:border-box; margin:0; }
  body { font-family:"Microsoft YaHei",sans-serif; background:var(--bg); color:#24292f;
         max-width:960px; margin:0 auto; padding:24px 16px; }
  h1 { font-size:20px; margin-bottom:4px; }
  .sub { color:#656d76; font-size:13px; margin-bottom:18px; }
  .askbox { display:flex; gap:8px; margin-bottom:20px; }
  input { flex:1; padding:11px 14px; border:1px solid var(--line); border-radius:8px; font-size:15px; }
  button { padding:11px 22px; background:var(--blue); color:#fff; border:0; border-radius:8px;
           font-size:15px; cursor:pointer; }
  button:disabled { background:#9bb8dd; cursor:wait; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-bottom:14px; }
  .meta { font-size:12px; color:#656d76; margin-bottom:8px; }
  .ans { line-height:1.8; font-size:15px; white-space:pre-wrap; }
  .cite { display:inline-block; background:#eaf1fb; color:var(--blue); border-radius:4px;
          padding:0 6px; font-size:12px; cursor:pointer; margin:0 2px; vertical-align:2px; }
  .cite:hover { background:var(--blue); color:#fff; }
  .src { border-left:3px solid var(--line); padding:8px 12px; margin:8px 0;
         font-size:13px; background:#fafbfc; border-radius:0 6px 6px 0; line-height:1.6; }
  .src.used { border-left-color:var(--blue); background:#f3f8ff; }
  .src .tag { color:var(--blue); font-weight:bold; }
  .src.dim { opacity:.45; }
  .status { color:#656d76; font-size:13px; padding:8px 0; }
</style>
</head>
<body>
<h1>财报问答知识库</h1>
<div class="sub">10 家白酒公司 2026 半年报 · 3,364 块 · 向量 + BM25 双路检索 · 答案带出处，点角标看原文块</div>
<div class="askbox">
  <input id="q" placeholder="例：茅台上半年直销渠道收入是多少？" onkeydown="if(event.key==='Enter')ask()">
  <button id="btn" onclick="ask()">提问</button>
</div>
<div id="out"></div>
<script>
const CIT = /\[(F\d{5})\]/g;
function renderAns(t){
  return t.replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))
          .replace(CIT, '<span class="cite" onclick="jump(\'$1\')">$1</span>');
}
function jump(id){ const el=document.getElementById('src-'+id); if(el){ el.scrollIntoView({behavior:'smooth',block:'center'});
  el.style.outline='2px solid #1a66cc'; setTimeout(()=>el.style.outline='',1500); } }
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q) return;
  const btn=document.getElementById('btn'); btn.disabled=true; btn.textContent='检索中…';
  document.getElementById('out').innerHTML='<div class="status">检索 + 生成中，约 5~15 秒…</div>';
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const d=await r.json(); if(d.error) throw new Error(d.error);
    let h='<div class="card"><div class="meta">问题：'+d.question+' · 用时 '+d.耗时+'s</div><div class="ans">'+renderAns(d.answer)+'</div></div>';
    h+='<div class="card"><div class="meta">召回的 '+d.sources.length+' 块（蓝边 = 被回答引用，点击角标定位）</div>';
    for(const s of d.sources){
      h+='<div class="src '+(s.被引用?'used':'dim')+'" id="src-'+s.id+'"><span class="tag">['+s.id+']</span> '
        +s.公司+' · '+s.章节+' · 第'+s.页码+'页 · '+s.类型+' · 命中:'+s.命中+' · 分'+s.得分
        +'<br>'+s.文本.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])).slice(0,300)+'…</div>';
    }
    h+='</div>';
    document.getElementById('out').innerHTML=h;
  }catch(e){ document.getElementById('out').innerHTML='<div class="status">出错：'+e.message+'</div>'; }
  btn.disabled=false; btn.textContent='提问';
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    retriever = None
    env = None

    def log_message(self, fmt, *args):
        pass  # 静默访问日志

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            data = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/ask":
            try:
                length = int(self.headers.get("Content-Length", 0))
                q = json.loads(self.rfile.read(length).decode("utf-8")).get("question", "").strip()
                if not q:
                    return self._json({"error": "问题为空"}, 400)
                result = answer_question(q, self.retriever, self.env)
                print(f"[问答] {q} -> 用时 {result['耗时']}s", flush=True)
                self._json(result)
            except Exception as e:
                self._json({"error": str(e)}, 500)
        else:
            self._json({"error": "not found"}, 404)


def main():
    env = load_env()
    if "你的" in env.get("DASHSCOPE_API_KEY", "") + env.get("DEEPSEEK_API_KEY", ""):
        sys.exit("[停止] .env 里 key 还没填好")
    retriever = Retriever()
    print(f"索引就绪: {len(retriever.chunks)} 块 | 模型 {CHAT_MODEL} / {EMBED_MODEL}")
    if "--ask" in sys.argv:
        q = sys.argv[sys.argv.index("--ask") + 1]
        r = answer_question(q, retriever, env)
        print(f"\n问题: {r['question']}（{r['耗时']}s）\n\n{r['answer']}\n\n来源:")
        for s in r["sources"]:
            mark = "●" if s["被引用"] else "○"
            print(f"  {mark} [{s['id']}] {s['公司']} {s['章节']} p{s['页码']} ({s['命中']})")
        return
    Handler.retriever, Handler.env = retriever, env
    print("问答页面: http://127.0.0.1:8787  （Ctrl+C 退出）")
    HTTPServer(("127.0.0.1", 8787), Handler).serve_forever()


if __name__ == "__main__":
    main()
