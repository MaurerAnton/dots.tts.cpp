// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026  Anton Maurer

// dots.tts.cpp - Minimal C++ HTTP web demo server
// Replaces Flask web_demo.py. Serves HTML UI and calls e2e_pipeline.
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <mutex>
#include <csignal>

static const int PORT = 8090;
static const char * PIPELINE_BIN = "./build/e2e_pipeline";
static const char * AUDIO_DIR = "./build";
static const char * WORKDIR = "/home/bym/dots.tts.cpp";
static std::mutex gen_mutex;

// Minimal HTML UI
static const char * HTML_PAGE = R"HTML(<!DOCTYPE html>
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
<p class="sub">Pure C++ implementation of dots.tts — 2B multilingual TTS. Text in, WAV out.</p>
<form id="f">
<textarea id="text" placeholder="Enter text to synthesize...">Hello world</textarea>
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
</script></body></html>)HTML";

// Simple URL decode
static std::string url_decode(const std::string & s) {
    std::string r;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            char hex[3] = {s[i+1], s[i+2], 0};
            r += (char)strtol(hex, nullptr, 16);
            i += 2;
        } else if (s[i] == '+') {
            r += ' ';
        } else {
            r += s[i];
        }
    }
    return r;
}

// Very basic HTTP request parser
struct HttpRequest {
    std::string method, path, body;
    std::string header(const char * name) { return ""; }
};

static HttpRequest parse_request(const char * data, int len) {
    HttpRequest req;
    std::string s(data, len);
    size_t pos = s.find(' ');
    if (pos != std::string::npos) {
        req.method = s.substr(0, pos);
        size_t pos2 = s.find(' ', pos + 1);
        if (pos2 != std::string::npos) {
            req.path = s.substr(pos + 1, pos2 - pos - 1);
        }
    }
    // Find body (after double CRLF)
    size_t body_start = s.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        req.body = s.substr(body_start + 4);
    }
    return req;
}

static void send_response(int fd, int code, const char * content_type, 
                          const std::string & body) {
    char header[4096];
    snprintf(header, sizeof(header),
        "HTTP/1.1 %d OK\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        code, content_type, body.size());
    send(fd, header, strlen(header), 0);
    send(fd, body.data(), body.size(), 0);
}

static void send_file(int fd, const char * path, const char * content_type) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        send_response(fd, 404, "text/plain", "File not found");
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char header[4096];
    snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %ld\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        content_type, sz);
    send(fd, header, strlen(header), 0);
    
    char buf[65536];
    while (sz > 0) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        if (n <= 0) break;
        send(fd, buf, n, 0);
        sz -= n;
    }
    fclose(f);
}

// Run pipeline and return JSON
static std::string run_pipeline(const std::string & text) {
    std::lock_guard<std::mutex> lock(gen_mutex);
    
    // Sanitize text (remove newlines, limit length)
    std::string safe = text;
    for (auto & c : safe) if (c == '"' || c == '\n' || c == '\r') c = ' ';
    if (safe.size() > 200) safe = safe.substr(0, 200);
    
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "cd %s && %s \"%s\" 2>&1", WORKDIR, PIPELINE_BIN, safe.c_str());
    
    FILE * p = popen(cmd, "r");
    if (!p) return R"({"error":"Failed to start pipeline"})";
    
    std::string output;
    char buf[4096];
    while (fgets(buf, sizeof(buf), p)) output += buf;
    int rc = pclose(p);
    
    // Check if output.wav was created
    std::string wav_path = std::string(AUDIO_DIR) + "/output.wav";
    FILE * wf = fopen(wav_path.c_str(), "rb");
    if (!wf) {
        return R"({"error":"Generation failed","log":")" + output.substr(0, 500) + "\"}";
    }
    fseek(wf, 0, SEEK_END);
    long sz = ftell(wf);
    fclose(wf);
    
    int samples = (sz - 44) / 2; // WAV header = 44 bytes, 16-bit samples
    float duration = samples / 48000.0f;
    
    char json[2048];
    snprintf(json, sizeof(json),
        R"({"file":"output.wav","samples":%d,"duration":"%.2f","log":"%s"})",
        samples, duration, output.substr(output.size() > 400 ? output.size() - 400 : 0, 400).c_str());
    
    // Escape any quotes in log
    std::string j(json);
    return j;
}

static void handle_client(int client_fd) {
    char buf[65536];
    int n = recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) { close(client_fd); return; }
    buf[n] = 0;
    
    HttpRequest req = parse_request(buf, n);
    
    if (req.method == "GET" && (req.path == "/" || req.path == "/index.html")) {
        send_response(client_fd, 200, "text/html; charset=utf-8", HTML_PAGE);
    } else if (req.method == "GET" && req.path.find("/audio/") == 0) {
        std::string file = req.path.substr(7);
        // Prevent directory traversal
        if (file.find("..") != std::string::npos) {
            send_response(client_fd, 403, "text/plain", "Forbidden");
        } else {
            std::string fpath = std::string(AUDIO_DIR) + "/" + file;
            send_file(client_fd, fpath.c_str(), "audio/wav");
        }
    } else if (req.method == "POST" && req.path == "/generate") {
        // Extract text from JSON body
        std::string text = "Hello world";
        size_t tp = req.body.find("\"text\"");
        if (tp != std::string::npos) {
            size_t ts = req.body.find('"', tp + 7);
            size_t te = req.body.find('"', ts + 1);
            if (ts != std::string::npos && te != std::string::npos) {
                text = req.body.substr(ts + 1, te - ts - 1);
            }
        }
        printf("Generate: '%s'\n", text.c_str());
        std::string result = run_pipeline(text);
        send_response(client_fd, 200, "application/json", result);
    } else if (req.method == "OPTIONS") {
        // CORS preflight
        send_response(client_fd, 200, "text/plain", "");
    } else {
        send_response(client_fd, 404, "text/plain", "Not found");
    }
    
    close(client_fd);
}

int main() {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);
    
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);
    
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    
    if (listen(server_fd, 5) < 0) { perror("listen"); return 1; }
    
    printf("dots.tts.cpp web demo: http://localhost:%d\n", PORT);
    printf("Pipeline binary: %s\n", PIPELINE_BIN);
    
    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;
        
        std::thread(handle_client, client_fd).detach();
    }
    
    close(server_fd);
    return 0;
}
