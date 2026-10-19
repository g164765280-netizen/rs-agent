# -*- coding: utf-8 -*-
"""rsagent_web.py — RS-Agent 网页聊天界面（flask 单文件）。

用法（服务器上）:
  /media/sdb1/gzj/support/llm/envs/all-in-rag/bin/python rsagent_web.py
  → 监听 127.0.0.1:7861（仅本机，外部经 ssh -L 隧道访问）

设计:
  - 启动即后台加载 ToolCallAgent（6 工具 function calling 完整版），页面先可用、显示加载状态
  - 推理全局串行（单锁）——GPU 单并发，多人访问自动排队
  - 零新依赖：flask（all-in-rag 环境自带 3.1.2）
"""
import threading
import time
import traceback

from flask import Flask, jsonify, request
from flask import render_template_string

app = Flask(__name__)

# ---------------- Agent 后台加载 ----------------
_state = {"status": "loading", "error": None, "ready_at": None}
_agent = None
_lock = threading.Lock()          # 推理串行锁
_INFER = threading.Lock()


def _load():
    global _agent
    try:
        from agent_toolcall import ToolCallAgent
        _agent = ToolCallAgent()
        _state["status"] = "ready"
        _state["ready_at"] = time.strftime("%H:%M:%S")
        print("[rsagent_web] agent ready", flush=True)
    except Exception:
        _state["status"] = "error"
        _state["error"] = traceback.format_exc(limit=3)
        print("[rsagent_web] load FAILED\n" + _state["error"], flush=True)


threading.Thread(target=_load, daemon=True).start()

# ---------------- 页面 ----------------
PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RS-Agent · 遥感变化检测智能问答</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #0f1420; color: #e6e9ef; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 14px 22px; border-bottom: 1px solid #232b3d; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 17px; font-weight: 600; }
  header .sub { font-size: 12px; color: #7c869e; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #888; flex: none; }
  .dot.ready { background: #2ecc71; box-shadow: 0 0 6px #2ecc71; }
  .dot.loading { background: #f39c12; animation: blink 1s infinite; }
  .dot.error { background: #e74c3c; }
  @keyframes blink { 50% { opacity: .3; } }
  #status-text { font-size: 12px; color: #9aa4bb; margin-left: auto; }
  #chat { flex: 1; overflow-y: auto; padding: 22px; display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 76%; padding: 11px 15px; border-radius: 12px; font-size: 14px;
         line-height: 1.65; white-space: pre-wrap; word-break: break-word; }
  .msg.user { align-self: flex-end; background: #2563eb; border-bottom-right-radius: 3px; }
  .msg.bot  { align-self: flex-start; background: #1c2434; border: 1px solid #2a3450; border-bottom-left-radius: 3px; }
  .msg.bot.thinking { color: #7c869e; font-style: italic; }
  .chips { padding: 0 22px 10px; display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { font-size: 12px; padding: 6px 12px; border-radius: 14px; background: #1c2434;
          border: 1px solid #2a3450; color: #9aa4bb; cursor: pointer; }
  .chip:hover { border-color: #2563eb; color: #e6e9ef; }
  footer { padding: 14px 22px 18px; border-top: 1px solid #232b3d; display: flex; gap: 10px; }
  #input { flex: 1; background: #161d2d; border: 1px solid #2a3450; border-radius: 10px;
           padding: 12px 14px; color: #e6e9ef; font-size: 14px; outline: none; resize: none; height: 46px; }
  #input:focus { border-color: #2563eb; }
  #send { background: #2563eb; border: none; border-radius: 10px; padding: 0 26px;
          color: #fff; font-size: 14px; cursor: pointer; }
  #send:disabled { background: #33405e; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <span class="dot loading" id="dot"></span>
  <div><h1>RS-Agent · 遥感变化检测智能问答</h1>
       <div class="sub">Qwen3-8B 编排 · 6 工具 function calling · 双路检索+查表+VLM</div></div>
  <span id="status-text">模型加载中…</span>
</header>
<div id="chat"></div>
<div class="chips">
  <span class="chip">变化检测中常用的评价指标有哪些？</span>
  <span class="chip">LEVIR-CD 数据集的训练样本数是多少？</span>
  <span class="chip">BIT 方法的核心思想是什么？</span>
</div>
<footer>
  <textarea id="input" placeholder="输入问题，Enter 发送…（首次提问前请等状态变绿）"></textarea>
  <button id="send" disabled>发送</button>
</footer>
<script>
const chat = document.getElementById('chat'), input = document.getElementById('input'),
      send = document.getElementById('send'), dot = document.getElementById('dot'),
      stext = document.getElementById('status-text');
let busy = false, ready = false;

function addMsg(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls; d.textContent = text;
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
  return d;
}
async function pollHealth() {
  try {
    const r = await (await fetch('/api/health')).json();
    dot.className = 'dot ' + r.status;
    if (r.status === 'ready') {
      ready = true; if (!busy) send.disabled = false;
      stext.textContent = '就绪（' + r.ready_at + ' 加载完成）';
    } else if (r.status === 'error') {
      stext.textContent = '加载失败：' + (r.error || '').split('\\n').slice(-2)[0];
    } else { stext.textContent = '模型加载中…（8B 首次加载约 1-3 分钟）'; }
  } catch (e) { dot.className = 'dot error'; stext.textContent = '服务连接断开'; }
  if (!ready) setTimeout(pollHealth, 3000);
}
async function ask() {
  const q = input.value.trim();
  if (!q || busy || !ready) return;
  busy = true; send.disabled = true; input.value = '';
  addMsg('user', q);
  const tip = addMsg('bot thinking', '思考中（检索 + 工具调用可能需要半分钟）…');
  try {
    const r = await (await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    })).json();
    tip.remove();
    addMsg('bot', r.ok ? r.answer : '❌ ' + r.error);
  } catch (e) { tip.remove(); addMsg('bot', '❌ 请求失败：' + e); }
  busy = false; if (ready) send.disabled = false; input.focus();
}
send.onclick = ask;
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } });
document.querySelectorAll('.chip').forEach(c => c.onclick = () => { input.value = c.textContent; input.focus(); });
pollHealth();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/health")
def health():
    return jsonify(_state)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if _state["status"] != "ready":
        return jsonify(ok=False, error=f"agent 未就绪（{_state['status']}）"), 503
    q = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not q:
        return jsonify(ok=False, error="问题为空"), 400
    with _INFER:                       # GPU 推理串行，多人访问排队
        t0 = time.time()
        try:
            ans = _agent.chat(q)
            print(f"[chat] {time.time()-t0:.1f}s | {q[:40]}", flush=True)
            return jsonify(ok=True, answer=ans, secs=round(time.time() - t0, 1))
        except Exception as e:
            return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    print("[rsagent_web] starting on 127.0.0.1:7861 …", flush=True)
    app.run(host="127.0.0.1", port=7861, threaded=True)
