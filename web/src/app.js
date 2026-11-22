// ClearView web demo: photo / video / live camera -> YOLO-World (onnxruntime-web, in a worker)
// -> same tracker + safety policy as the Quest app -> soft patches drawn on a canvas.
(() => {
  "use strict";
  const C = window.ClearViewCore;
  const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.wasm.min.js";
  const ORT_GPU_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.webgpu.min.js";
  const IN_W = 416, IN_H = 320, COL_W = 96, COL_H = 72, PATCH = 32;
  const FADE_IN = 0.3, FADE_OUT = 0.1, FOLLOW = 10;

  const LABEL_ES = {
    billboard: "valla", "advertisement poster": "cartel", logo: "logo", "shop sign": "rótulo",
    "advertising screen": "pantalla", person: "persona", bicycle: "bici", car: "coche", motorcycle: "moto",
    bus: "autobús", truck: "camión", "traffic light": "semáforo", "stop sign": "stop", "traffic sign": "señal",
  };
  const REASON_ES = {
    safetyOverlap: "toca a una persona, vehículo o señal",
    tooLarge: "demasiado grande (está cerca)",
    groundLevel: "a ras de suelo",
    areaBudget: "límite de área tapada",
    unconfirmed: "esperando confirmación",
    stale: "ya no se ve",
  };

  // YOLO-World scores run lower than a trained YOLO: thresholds tuned on test photos.
  const cfg = C.defaults();
  cfg.decoder.adScoreThreshold = 0.25;
  cfg.decoder.safetyScoreThreshold = 0.12;

  const $ = (id) => document.getElementById(id);
  const stage = $("stage");
  const ctx = stage.getContext("2d");
  const video = $("video");
  const inCtx = mkCanvas(IN_W, IN_H).getContext("2d", { willReadFrequently: true });
  const colCtx = mkCanvas(COL_W, COL_H).getContext("2d", { willReadFrequently: true });

  let manifest = null, worker = null, ready = false, busy = false, reqId = 0, pending = null;
  let mode = null, source = null, objectUrl = null, stream = null, isSample = false;
  let tracker = new C.AdTracker(cfg.tracker);
  const visuals = new Map();
  let last = null, lastResultAt = -Infinity, avgMs = 0, dirty = true, prevTs = 0;
  let showDebug = false, showOriginal = false;
  // Kill switch (R11). Photo and video start on; live camera starts OFF so the user opts in.
  let hidingOn = true, backend = "", wakeLock = null;
  const motion = new C.MotionGuard(cfg.motion);
  let rotRate = 0, hasMotion = false, motionFactor = 1;

  function mkCanvas(w, h) { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; }
  const nowS = () => performance.now() / 1000;
  const mb = (b) => (b / 1048576).toFixed(0);

  // ---------------- status + notices ----------------
  function setStatus(kind, text) {
    const s = $("status");
    s.dataset.kind = kind;
    $("status-text").textContent = text;
    $("stage-msg").hidden = kind === "ready";
    $("stage-msg-text").textContent = text;
  }
  function notice(text) {
    const n = $("notice");
    n.textContent = text || "";
    n.hidden = !text;
  }

  // ---------------- model ----------------
  const WORKER_SRC = `
    let session = null;
    onmessage = async (e) => {
      const m = e.data;
      try {
        if (m.type === "init") {
          const gpu = m.gpuUrl && self.navigator && self.navigator.gpu;
          let gpuNote = !m.gpuUrl ? "" : !gpu ? "este navegador no ofrece WebGPU" : "";
          importScripts(gpu ? m.gpuUrl : m.ortUrl);
          ort.env.wasm.wasmPaths = m.wasmBase;
          ort.env.wasm.numThreads = 1;
          let backend = "wasm";
          if (gpu) {
            try {
              session = await ort.InferenceSession.create(m.model, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
              // Warm-up run: some GPUs create the session but fail on the first inference.
              const z = new ort.Tensor("float32", new Float32Array(3 * m.h * m.w), [1, 3, m.h, m.w]);
              await session.run({ [session.inputNames[0]]: z });
              backend = "webgpu";
            } catch (err) {
              session = null;
              gpuNote = "WebGPU falló: " + String(err && err.message || err).slice(0, 80);
            }
          }
          if (!session) session = await ort.InferenceSession.create(m.model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
          postMessage({ type: "ready", backend, gpuNote });
        } else if (m.type === "run") {
          const t0 = performance.now();
          const input = new ort.Tensor("float32", m.data, [1, 3, m.h, m.w]);
          const out = await session.run({ [session.inputNames[0]]: input });
          const o = out[session.outputNames[0]];
          postMessage({ type: "result", id: m.id, data: o.data, dims: o.dims, ms: performance.now() - t0 }, [o.data.buffer]);
        }
      } catch (err) {
        postMessage({ type: "error", message: String(err && err.message || err) });
      }
    };`;

  async function loadModel() {
    setStatus("loading", "Preparando el detector…");
    const res = await fetch("model/manifest.json");
    if (!res.ok) throw new Error("no se encontró model/manifest.json");
    manifest = await res.json();
    const total = manifest.bytes;
    const buf = new Uint8Array(total);
    let off = 0;
    for (const name of manifest.chunks) {
      const r = await fetch("model/" + name);
      if (!r.ok) throw new Error("no se pudo descargar " + name);
      if (r.body && r.body.getReader) {
        const reader = r.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf.set(value, off);
          off += value.length;
          setStatus("loading", `Descargando el detector · ${mb(off)} de ${mb(total)} MB`);
        }
      } else {
        const b = new Uint8Array(await r.arrayBuffer());
        buf.set(b, off);
        off += b.length;
      }
    }
    if (off !== total) throw new Error("descarga incompleta del modelo");
    setStatus("loading", "Iniciando el detector…");
    worker = new Worker(URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" })));
    worker.onmessage = onWorker;
    worker.onerror = (e) => fail("El detector se detuvo: " + (e.message || "error desconocido"));
    worker.postMessage({
      type: "init", model: buf.buffer, ortUrl: ORT_URL, gpuUrl: manifest.webgpu ? ORT_GPU_URL : null,
      wasmBase: new URL("ort/", location.href).href, w: IN_W, h: IN_H,
    }, [buf.buffer]);
  }

  function onWorker(e) {
    const m = e.data;
    if (m.type === "ready") {
      ready = true;
      backend = m.backend === "webgpu" ? "GPU" : "CPU";
      setStatus("ready", "Listo · " + backend + (m.gpuNote ? " (" + m.gpuNote + ")" : ""));
      dirty = true;
    } else if (m.type === "result") {
      if (pending && pending.id === m.id) onResult(m.data, m.dims, m.ms);
    } else if (m.type === "error") {
      fail("Error del detector: " + m.message);
    }
  }

  function fail(text) {
    ready = false;
    busy = false;
    visuals.clear(); // fail-open: never leave patches up
    dirty = true;
    setStatus("error", text);
  }

  // ---------------- detection ----------------
  function sourceReady() {
    if (!source) return false;
    if (source instanceof HTMLVideoElement) return source.readyState >= 2 && source.videoWidth > 0;
    return source.complete && source.naturalWidth > 0;
  }

  function maybeDetect() {
    if (!ready || busy || !sourceReady() || !hidingOn || document.hidden) return;
    if (mode === "video" && video.paused && last) return;
    if (mode === "photo" && last && last.sourceToken === sourceToken) return;
    busy = true;
    inCtx.drawImage(source, 0, 0, IN_W, IN_H);
    const px = inCtx.getImageData(0, 0, IN_W, IN_H).data;
    const plane = IN_W * IN_H;
    const f = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      f[i] = px[p] / 255;
      f[i + plane] = px[p + 1] / 255;
      f[i + 2 * plane] = px[p + 2] / 255;
    }
    colCtx.drawImage(source, 0, 0, COL_W, COL_H);
    pending = { id: ++reqId, colors: colCtx.getImageData(0, 0, COL_W, COL_H).data, token: sourceToken };
    worker.postMessage({ type: "run", id: pending.id, data: f, w: IN_W, h: IN_H }, [f.buffer]);
  }

  let sourceToken = 0;

  function onResult(data, dims, ms) {
    busy = false;
    if (pending.token !== sourceToken) return; // source changed meanwhile
    let dets;
    try {
      dets = C.decode(data, dims[1] - 4, dims[2], IN_W, IN_H, manifest.categories, cfg.decoder);
    } catch (err) {
      fail("Salida del modelo inesperada: " + err.message);
      return;
    }
    const ads = dets.filter((d) => d.category === "ad");
    const safety = dets.filter((d) => d.category === "safety");
    const now = nowS();
    if (mode === "photo") {
      // A still photo has no "next frames": confirmation (R7) is skipped explicitly.
      tracker.clear();
      for (let i = 0; i < cfg.tracker.minHitsToConfirm; i++) tracker.update(ads, now);
    } else {
      tracker.update(ads, now);
    }
    const policy = C.evaluatePolicy(tracker.tracks, safety, now, cfg.safety, cfg.tracker);

    const seen = new Set();
    for (const m of policy.masks) {
      seen.add(m.trackId);
      const edges = C.sampleEdges(pending.colors, COL_W, COL_H, m.sourceBox, 0.15);
      let v = visuals.get(m.trackId);
      if (!v) {
        v = { box: m.box, opacity: mode === "photo" ? m.opacity : 0 };
        visuals.set(m.trackId, v);
      }
      if (mode === "photo") v.box = m.box;
      v.target = m.box;
      v.targetOpacity = m.opacity;
      v.patch = makePatch(edges);
    }
    for (const [id, v] of visuals) if (!seen.has(id)) v.targetOpacity = 0;

    last = { ads, safety, policy, sourceToken };
    lastResultAt = now;
    avgMs = avgMs ? avgMs * 0.7 + ms * 0.3 : ms;
    dirty = true;
    renderSummary();
  }

  // Same blend as Shaders/ClearViewSoftFill.shader: 4 edge colours, feathered alpha.
  function makePatch(e) {
    const c = mkCanvas(PATCH, PATCH);
    const g = c.getContext("2d");
    const im = g.createImageData(PATCH, PATCH);
    const feather = 0.12;
    const lerp = (a, b, t) => a + (b - a) * t;
    for (let y = 0; y < PATCH; y++) {
      for (let x = 0; x < PATCH; x++) {
        const u = (x + 0.5) / PATCH, v = 1 - (y + 0.5) / PATCH;
        const dh = Math.min(u, 1 - u), dv = Math.min(v, 1 - v);
        const wH = dv / (dh + dv + 1e-5);
        const edge = Math.min(dh, dv);
        const t = Math.min(1, Math.max(0, edge / feather));
        const a = t * t * (3 - 2 * t);
        const i = (y * PATCH + x) * 4;
        for (let k = 0; k < 3; k++) {
          const horiz = lerp(e.left[k], e.right[k], u);
          const vert = lerp(e.bottom[k], e.top[k], v);
          im.data[i + k] = lerp(vert, horiz, wH) * 255;
        }
        im.data[i + 3] = a * 255;
      }
    }
    g.putImageData(im, 0, 0);
    return c;
  }

  // ---------------- rendering ----------------
  function resizeStage() {
    const w = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
    const h = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
    if (!w || !h) return;
    const scale = Math.min(1, 1280 / Math.max(w, h));
    const cw = Math.round(w * scale), ch = Math.round(h * scale);
    if (stage.width !== cw || stage.height !== ch) { stage.width = cw; stage.height = ch; }
  }

  function render(dt) {
    if (!sourceReady()) return;
    resizeStage();
    const W = stage.width, H = stage.height;
    ctx.drawImage(source, 0, 0, W, H);

    const now = nowS();
    const timeout = Math.max(cfg.failSafe.resultTimeoutSeconds, (3 * avgMs) / 1000);
    const healthy = ready && hidingOn && (mode === "photo" || now - lastResultAt <= timeout);
    const gf = healthy ? (mode === "live" && hasMotion ? motionFactor : 1) : 0;
    updateLiveState(healthy, gf);
    const k = 1 - Math.exp(-FOLLOW * dt);

    for (const [id, v] of visuals) {
      if (mode !== "photo") {
        const goal = v.targetOpacity * gf;
        const secs = goal > v.opacity ? FADE_IN : FADE_OUT;
        v.opacity = goal > v.opacity ? Math.min(goal, v.opacity + dt / secs) : Math.max(goal, v.opacity - dt / secs);
        v.box = C.rect(v.box.x + (v.target.x - v.box.x) * k, v.box.y + (v.target.y - v.box.y) * k,
          v.box.w + (v.target.w - v.box.w) * k, v.box.h + (v.target.h - v.box.h) * k);
      } else if (!healthy) {
        v.opacity = 0;
      } else {
        v.opacity = v.targetOpacity;
      }
      if (v.opacity <= 0.001 && v.targetOpacity === 0) { visuals.delete(id); continue; }
      if (showOriginal || v.opacity <= 0.001) continue;
      ctx.globalAlpha = v.opacity;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(v.patch, v.box.x * W, v.box.y * H, v.box.w * W, v.box.h * H);
      ctx.globalAlpha = 1;
    }
    if (showDebug && !showOriginal && last) drawOverlay(W, H);
  }

  // Live overlay: always tells the user whether reality is being altered (R12).
  let liveStateKey = "";
  function updateLiveState(healthy, gf) {
    if (mode !== "live") return;
    let key, text;
    if (!hidingOn) { key = "off"; text = "Desactivado: ves la cámara tal cual"; }
    else if (!ready) { key = "wait"; text = "Preparando el detector…"; }
    else if (!last) { key = "wait"; text = "Buscando publicidad…"; }
    else if (!healthy) { key = "pause"; text = "En pausa: el detector no responde"; }
    else if (gf < 0.5) { key = "pause"; text = "En pausa: giro rápido"; }
    else { key = "on"; text = `Ocultando publicidad · ${last.policy.masks.length} ahora`; }
    text += backend && avgMs ? ` · ${Math.round(avgMs)} ms ${backend}` : "";
    if (key + text === liveStateKey) return;
    liveStateKey = key + text;
    const el = $("live-state");
    el.dataset.kind = key;
    el.textContent = text;
  }

  function drawOverlay(W, H) {
    const lw = Math.max(2, W / 400), fs = Math.max(12, Math.round(W / 55));
    ctx.font = `600 ${fs}px Archivo, system-ui, sans-serif`;
    ctx.textBaseline = "top";
    const label = (text, x, y, color) => {
      const pad = fs * 0.3, tw = ctx.measureText(text).width;
      const ly = Math.max(0, y - fs - pad * 2);
      x = Math.max(0, Math.min(x, W - tw - pad * 2));
      ctx.fillStyle = color;
      ctx.fillRect(x, ly, tw + pad * 2, fs + pad * 2);
      ctx.fillStyle = "#0d1014";
      ctx.fillText(text, x + pad, ly + pad);
    };
    const box = (b, color, dash, text) => {
      ctx.setLineDash(dash ? [lw * 3, lw * 2] : []);
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.strokeRect(b.x * W, b.y * H, b.w * W, b.h * H);
      ctx.setLineDash([]);
      if (text) label(text, b.x * W, b.y * H, color);
    };
    const css = getComputedStyle(document.documentElement);
    const cHide = css.getPropertyValue("--hide").trim(), cSafe = css.getPropertyValue("--safe").trim(), cWarn = css.getPropertyValue("--warn").trim();
    for (const s of last.safety) box(s.box, cSafe, false, LABEL_ES[manifest.classes[s.classId]] || "seguridad");
    for (const sk of last.policy.skipped) {
      if (sk.reason === "unconfirmed" || sk.reason === "stale") continue;
      box(sk.track.box, cWarn, true, "visible: " + REASON_ES[sk.reason]);
    }
    for (const m of last.policy.masks) box(m.sourceBox, cHide, true, "oculto");
  }

  function loop(ts) {
    const dt = prevTs ? Math.min(0.1, (ts - prevTs) / 1000) : 0.016;
    prevTs = ts;
    if (mode === "live" && hasMotion) motionFactor = motion.update(rotRate, dt);
    maybeDetect();
    if (mode !== "photo" || dirty) {
      render(dt);
      dirty = false;
    }
    requestAnimationFrame(loop);
  }

  // ---------------- summary ----------------
  function renderSummary() {
    if (!last) return;
    const hidden = last.policy.masks.length;
    const kept = last.policy.skipped.filter((s) => s.reason !== "unconfirmed" && s.reason !== "stale");
    $("sum-hidden").textContent = hidden;
    $("sum-kept").textContent = kept.length;
    $("sum-safe").textContent = last.safety.length;
    $("sum-ms").textContent = Math.round(avgMs);

    const reasons = {};
    kept.forEach((s) => (reasons[s.reason] = (reasons[s.reason] || 0) + 1));
    const ul = $("reasons");
    ul.replaceChildren();
    for (const [r, n] of Object.entries(reasons)) {
      const li = document.createElement("li");
      li.textContent = `${n} × ${REASON_ES[r]}`;
      ul.appendChild(li);
    }
    const safeCounts = {};
    last.safety.forEach((s) => {
      const n = LABEL_ES[manifest.classes[s.classId]] || "otro";
      safeCounts[n] = (safeCounts[n] || 0) + 1;
    });
    $("safe-list").textContent = Object.entries(safeCounts).map(([n, c]) => `${c} ${n}`).join(" · ") || "ninguno";
    $("summary").hidden = false;
  }

  // ---------------- sources ----------------
  function resetPipeline() {
    sourceToken++;
    tracker = new C.AdTracker(cfg.tracker);
    visuals.clear();
    last = null;
    lastResultAt = -Infinity;
    $("summary").hidden = true;
    dirty = true;
  }

  function stopMedia() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    video.pause();
    video.removeAttribute("src");
    video.srcObject = null;
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
  }

  async function requestWakeLock() {
    try { if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen"); } catch (_) { /* optional */ }
  }

  function setMode(m) {
    mode = m;
    document.body.dataset.mode = m;
    document.body.classList.toggle("live", m === "live");
    $("live-ui").hidden = m !== "live";
    if (m === "live") { setHiding(false); requestWakeLock(); }
    else {
      hidingOn = true;
      if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
    }
    $("speed-wrap").hidden = m !== "video";
    $("credit").hidden = !isSample;
  }

  function usePhoto(src, sample) {
    stopMedia();
    resetPipeline();
    isSample = !!sample;
    const img = new Image();
    img.onload = () => { source = img; setMode("photo"); dirty = true; };
    img.onerror = () => notice("No se pudo abrir esa imagen.");
    if (!sample) objectUrl = src;
    img.src = src;
    notice("");
  }

  function useVideoFile(file) {
    stopMedia();
    resetPipeline();
    isSample = false;
    objectUrl = URL.createObjectURL(file);
    video.srcObject = null;
    video.src = objectUrl;
    video.loop = true;
    video.playbackRate = parseFloat($("speed").value);
    video.play().catch(() => notice("Toca la imagen para reproducir el vídeo."));
    source = video;
    setMode("video");
    notice("");
  }

  async function useLive() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } }, audio: false });
      stopMedia();
      resetPipeline();
      isSample = false;
      stream = s;
      video.srcObject = s;
      await video.play();
      source = video;
      setMode("live");
      notice("");
      enableMotion();
    } catch (err) {
      const inClaude = /claude\.ai|claudeusercontent|claude\.site/.test(location.hostname + document.referrer);
      notice(inClaude
        ? "Dentro de Claude la cámara en vivo está bloqueada. Abre la versión alojada de ClearView en Chrome para usarla."
        : "No se pudo abrir la cámara. Revisa que el navegador tenga permiso de cámara para esta página y que se abra por https.");
    }
  }

  async function enableMotion() {
    hasMotion = false;
    try {
      if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
        if ((await DeviceMotionEvent.requestPermission()) !== "granted") throw new Error("denied");
      }
      window.addEventListener("devicemotion", (e) => {
        const r = e.rotationRate;
        if (!r || r.alpha == null) return;
        hasMotion = true;
        rotRate = Math.hypot(r.alpha || 0, r.beta || 0, r.gamma || 0);
      });
    } catch (_) { /* no sensor: rule R8 simply does not apply */ }
    setTimeout(() => {
      if (mode === "live" && !hasMotion) notice("Sin sensor de movimiento: la regla de «giro rápido» no se aplica en esta prueba.");
    }, 1500);
  }

  // ---------------- UI wiring ----------------
  $("in-photo").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) usePhoto(URL.createObjectURL(f), false);
    e.target.value = "";
  });
  $("in-video").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) useVideoFile(f);
    e.target.value = "";
  });
  $("btn-live").addEventListener("click", useLive);
  $("btn-sample").addEventListener("click", () => usePhoto("sample.jpg", true));
  $("debug").addEventListener("change", (e) => { showDebug = e.target.checked; dirty = true; });
  $("speed").addEventListener("change", (e) => { video.playbackRate = parseFloat(e.target.value); });
  stage.addEventListener("click", () => {
    if (mode !== "video") return;
    if (video.paused) video.play(); else video.pause();
  });
  const compare = $("btn-compare");
  compare.addEventListener("click", () => {
    showOriginal = !showOriginal;
    compare.setAttribute("aria-pressed", String(showOriginal));
    compare.textContent = showOriginal ? "Viendo el original · toca para volver" : "Ver el original";
    dirty = true;
  });

  // Live controls
  function setHiding(on) {
    hidingOn = on;
    const b = $("live-toggle");
    b.setAttribute("aria-pressed", String(on));
    b.textContent = on ? "Desactivar" : "Activar ClearView";
    if (!on) { visuals.clear(); tracker.clear(); last = null; } // instant off, no fade
    dirty = true;
  }
  $("live-toggle").addEventListener("click", () => setHiding(!hidingOn));
  $("live-exit").addEventListener("click", () => usePhoto("sample.jpg", true));
  $("live-debug").addEventListener("change", (e) => { showDebug = e.target.checked; $("debug").checked = showDebug; });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && mode === "live") requestWakeLock();
  });

  usePhoto("sample.jpg", true);
  requestAnimationFrame(loop);
  loadModel().catch((err) => fail("No se pudo cargar el detector: " + err.message));
})();
