#!/usr/bin/env python3.12
"""Simple web UI for dots.tts.cpp pipeline."""
import subprocess, os, tempfile, time
from flask import Flask, request, render_template_string, send_file, jsonify

app = Flask(__name__)
PIPELINE_BIN = os.path.join(os.path.dirname(__file__), "build", "e2e_pipeline")
WORKDIR = os.path.dirname(__file__)

HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>dots.tts.cpp - Web Demo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;max-width:700px;margin:40px auto;padding:20px}
h1{color:#58a6ff;margin-bottom:8px}
.sub{color:#8b949e;margin-bottom:24px}
textarea{width:100%;height:80px;background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:12px;font-size:16px;resize:vertical}
.row{display:flex;gap:12px;margin:12px 0;align-items:center}
button{background:#238636;color:white;border:none;padding:10px 24px;border-radius:6px;font-size:15px;cursor:pointer}
button:hover{background:#2ea043}
button:disabled{background:#30363d;cursor:not-allowed}
select{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:8px 12px;font-size:14px}
.result{margin-top:20px;padding:16px;background:#161b22;border:1px solid #30363d;border-radius:6px;display:none}
.result audio{width:100%;margin-top:8px}
.status{color:#8b949e;font-size:13px;margin-top:8px}
.stats{display:flex;gap:20px;margin-top:8px;font-size:13px;color:#8b949e}
.stats span{color:#58a6ff}
.error{color:#f85149}
.log{background:#0d1117;padding:12px;border-radius:4px;margin-top:8px;font-family:monospace;font-size:12px;max-height:200px;overflow-y:auto;white-space:pre-wrap;color:#8b949e}
</style></head><body>
<h1>dots.tts.cpp</h1>
<p class="sub">First C++ implementation of dots.tts — 2B multilingual TTS. Text in, WAV out.</p>
<form id="f">
<textarea id="text" placeholder="Enter text to synthesize (any language supported by Qwen2.5: Russian, English, German, Chinese...)">Привет, как дела?</textarea>
<div class="row">
<button type="submit" id="btn">Generate Speech</button>
<span id="status" class="status"></span>
</div>
</form>
<div class="result" id="result">
<audio id="audio" controls></audio>
<div class="stats">
<span id="samples">-</span> samples &middot;
<span id="duration">-</span> sec &middot;
<span id="rate">48 kHz</span>
</div>
<div class="log" id="log"></div>
</div>
<script>
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  const result = document.getElementById('result');
  const audio = document.getElementById('audio');
  const log = document.getElementById('log');
  btn.disabled = true;
  status.textContent = 'Loading LLM + DiT weights...';
  result.style.display = 'block';
  log.textContent = '';

  try {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: document.getElementById('text').value})
    });
    const data = await resp.json();
    if (data.error) {
      status.innerHTML = '<span class="error">' + data.error + '</span>';
      log.textContent = data.log || '';
    } else {
      status.textContent = 'Done!';
      audio.src = '/audio/' + data.file;
      audio.load();
      document.getElementById('samples').textContent = data.samples;
      document.getElementById('duration').textContent = data.duration;
      log.textContent = data.log || '';
    }
  } catch(err) {
    status.innerHTML = '<span class="error">' + err.message + '</span>';
  }
  btn.disabled = false;
};
</script></body></html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate", methods=["POST"])
def generate():
    text = request.json.get("text", "Hello world")[:500]
    out_wav = os.path.join(WORKDIR, "build", "output.wav")
    
    # Remove old WAV
    if os.path.exists(out_wav):
        os.unlink(out_wav)
    
    start = time.time()
    try:
        result = subprocess.run(
            [PIPELINE_BIN, text],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.join(WORKDIR, "build")
        )
        elapsed = time.time() - start
        
        # Filter llama debug output
        lines = [l for l in result.stdout.split('\n') + result.stderr.split('\n')
                 if l and not l.startswith('llama') and 'print_info' not in l
                 and 'load_tensors' not in l and 'create_tensor' not in l
                 and 'graph_reserve' not in l and 'sched_reserve' not in l
                 and 'kv_cache' not in l and 'done_get' not in l]
        log = '\n'.join(lines)
        
        if os.path.exists(out_wav):
            size = os.path.getsize(out_wav)
            import wave
            with wave.open(out_wav) as w:
                samples = w.getnframes()
                duration = samples / w.getframerate()
            return jsonify({
                "file": "output.wav",
                "samples": samples,
                "duration": f"{duration:.2f}",
                "log": log + f"\n\nGenerated in {elapsed:.1f}s"
            })
        else:
            return jsonify({"error": "WAV not generated", "log": log})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timeout (300s)", "log": ""})
    except Exception as e:
        return jsonify({"error": str(e), "log": ""})

@app.route("/audio/<path:name>")
def audio(name):
    return send_file(os.path.join(WORKDIR, "build", name), mimetype="audio/wav")

if __name__ == "__main__":
    print(f"dots.tts.cpp web demo: http://localhost:8090")
    print(f"Pipeline binary: {PIPELINE_BIN}")
    app.run(host="0.0.0.0", port=8090, debug=False)
