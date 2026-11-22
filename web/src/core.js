// ClearView core for the browser: a line-by-line port of unity/Assets/ClearView/Core (C#).
// Boxes are normalized, origin top-left, y down: {x, y, w, h}.
// Keep the defaults identical to the C# SafetyConfig / TrackerConfig / DecoderConfig.
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ClearViewCore = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------- RectN ----------
  const rect = (x, y, w, h) => ({ x, y, w: Math.max(0, w), h: Math.max(0, h) });
  const right = (r) => r.x + r.w;
  const bottom = (r) => r.y + r.h;
  const area = (r) => r.w * r.h;
  function intersectionArea(a, b) {
    const w = Math.min(right(a), right(b)) - Math.max(a.x, b.x);
    const h = Math.min(bottom(a), bottom(b)) - Math.max(a.y, b.y);
    return w <= 0 || h <= 0 ? 0 : w * h;
  }
  function iou(a, b) {
    const inter = intersectionArea(a, b);
    const union = area(a) + area(b) - inter;
    return union <= 0 ? 0 : inter / union;
  }
  const inset = (r, f) => rect(r.x + r.w * f, r.y + r.h * f, r.w - 2 * r.w * f, r.h - 2 * r.h * f);
  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  function clampToUnit(r) {
    const x0 = clamp01(r.x), y0 = clamp01(r.y), x1 = clamp01(right(r)), y1 = clamp01(bottom(r));
    return rect(x0, y0, x1 - x0, y1 - y0);
  }
  const lerpRect = (a, b, t) => rect(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.w + (b.w - a.w) * t, a.h + (b.h - a.h) * t);

  // ---------- Letterbox: fit the frame into the model input without distortion ----------
  // YOLO models are trained on letterboxed images; stretching a portrait camera frame into a
  // landscape input (or vice versa) degrades detections.
  function letterbox(srcW, srcH, inW, inH) {
    const scale = Math.min(inW / srcW, inH / srcH);
    const nw = Math.round(srcW * scale), nh = Math.round(srcH * scale);
    return { scale, nw, nh, dx: Math.floor((inW - nw) / 2), dy: Math.floor((inH - nh) / 2), inW, inH, srcW, srcH };
  }
  // Box normalized to the model input -> box normalized to the source frame (clamped).
  function unletterbox(box, lb) {
    const x0 = (box.x * lb.inW - lb.dx) / lb.nw, y0 = (box.y * lb.inH - lb.dy) / lb.nh;
    const x1 = ((box.x + box.w) * lb.inW - lb.dx) / lb.nw, y1 = ((box.y + box.h) * lb.inH - lb.dy) / lb.nh;
    return clampToUnit(rect(x0, y0, x1 - x0, y1 - y0));
  }

  // ---------- Configs (same defaults as C#) ----------
  const defaults = () => ({
    decoder: { adScoreThreshold: 0.45, safetyScoreThreshold: 0.2, nmsIoU: 0.45, maxDetections: 100 },
    tracker: { matchIoU: 0.3, minHitsToConfirm: 3, maxMisses: 2, maxAgeSeconds: 1.0, boxSmoothing: 0.5 },
    safety: {
      insetFraction: 0.06, safetyMargin: 0.1, safetyMarginMin: 0.02,
      maxSingleMaskArea: 0.12, maxTotalMaskArea: 0.25, groundGuardY: 0.8,
      minDistanceMeters: 2.0, maxOpacity: 0.9,
    },
    motion: { angularSoftDegPerSec: 60, angularHardDegPerSec: 120, recoverPerSecond: 1.5 },
    failSafe: { resultTimeoutSeconds: 1.0 },
  });

  // ---------- YOLO decoder: output [1, 4+nc, anchors] ----------
  function decode(output, numClasses, numAnchors, inputW, inputH, categories, cfg) {
    if (categories.length !== numClasses) throw new Error(`labels ${categories.length} != model classes ${numClasses}`);
    if (output.length < (4 + numClasses) * numAnchors) throw new Error("output too short");
    const cand = [];
    for (let a = 0; a < numAnchors; a++) {
      let best = -1, bestScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const s = output[(4 + c) * numAnchors + a];
        if (s > bestScore) { bestScore = s; best = c; }
      }
      if (best < 0) continue;
      const cat = categories[best];
      const th = cat === "safety" ? cfg.safetyScoreThreshold : cfg.adScoreThreshold;
      if (bestScore < th) continue;
      const cx = output[a] / inputW, cy = output[numAnchors + a] / inputH;
      const w = output[2 * numAnchors + a] / inputW, h = output[3 * numAnchors + a] / inputH;
      const box = clampToUnit(rect(cx - w / 2, cy - h / 2, w, h));
      if (area(box) <= 0) continue;
      cand.push({ box, classId: best, score: bestScore, category: cat, anchor: a });
    }
    cand.sort((p, q) => q.score - p.score);
    const kept = [];
    for (const d of cand) {
      if (kept.length >= cfg.maxDetections) break;
      if (!kept.some((k) => k.classId === d.classId && iou(k.box, d.box) > cfg.nmsIoU)) kept.push(d);
    }
    return kept;
  }

  // ---------- Tracker ----------
  class AdTracker {
    constructor(cfg) { this.cfg = cfg; this.tracks = []; this.nextId = 1; }
    clear() { this.tracks = []; }
    update(ads, now) {
      const pairs = [];
      this.tracks.forEach((t, ti) => ads.forEach((d, di) => {
        const v = iou(t.box, d.box);
        if (v >= this.cfg.matchIoU) pairs.push([v, ti, di]);
      }));
      pairs.sort((p, q) => q[0] - p[0]);
      const tUsed = new Array(this.tracks.length).fill(false);
      const dUsed = new Array(ads.length).fill(false);
      for (const [, ti, di] of pairs) {
        if (tUsed[ti] || dUsed[di]) continue;
        tUsed[ti] = dUsed[di] = true;
        const t = this.tracks[ti], d = ads[di];
        t.box = lerpRect(t.box, d.box, this.cfg.boxSmoothing);
        t.score = d.score; t.classId = d.classId; t.hits++; t.misses = 0; t.lastSeen = now;
      }
      for (let i = this.tracks.length - 1; i >= 0; i--) {
        if (tUsed[i]) continue;
        const t = this.tracks[i];
        t.misses++;
        if (t.misses > this.cfg.maxMisses || now - t.lastSeen > this.cfg.maxAgeSeconds) this.tracks.splice(i, 1);
      }
      ads.forEach((d, di) => {
        if (!dUsed[di]) this.tracks.push({ id: this.nextId++, box: d.box, classId: d.classId, score: d.score, hits: 1, misses: 0, lastSeen: now });
      });
    }
  }

  // ---------- Safety policy ----------
  const SKIP = {
    Unconfirmed: "unconfirmed", Stale: "stale", SafetyOverlap: "safetyOverlap",
    TooLarge: "tooLarge", GroundLevel: "groundLevel", AreaBudget: "areaBudget",
  };

  // Safety box grown by the safety margin (relative, with an absolute minimum for small, distant objects).
  function guardRect(b, s) {
    const mx = Math.max(b.w * s.safetyMargin, s.safetyMarginMin);
    const my = Math.max(b.h * s.safetyMargin, s.safetyMarginMin);
    return rect(b.x - mx, b.y - my, b.w + 2 * mx, b.h + 2 * my);
  }

  function safetyOverlaps(mask, safetyObjects, s) {
    return safetyObjects.filter((o) => o.category === "safety" && intersectionArea(mask, guardRect(o.box, s)) > 0);
  }

  // Returns {masks: [{trackId, sourceBox, box, opacity, cutouts}], skipped: [{track, reason, blocker?}]}
  // With safetyCfg.cutOutSafety, an ad touching a person/vehicle/sign is still hidden, but the renderer
  // MUST leave every object in `cutouts` (grown by the safety margin) visible in front of the panel.
  function evaluatePolicy(tracks, safetyObjects, now, safetyCfg, trackerCfg) {
    const masks = [], skipped = [], cand = [];
    for (const t of tracks) {
      if (t.hits < trackerCfg.minHitsToConfirm) { skipped.push({ track: t, reason: SKIP.Unconfirmed }); continue; }
      if (now - t.lastSeen > trackerCfg.maxAgeSeconds) { skipped.push({ track: t, reason: SKIP.Stale }); continue; }
      const box = inset(t.box, safetyCfg.insetFraction);
      if (area(box) <= 0 || area(box) > safetyCfg.maxSingleMaskArea) { skipped.push({ track: t, reason: SKIP.TooLarge }); continue; }
      if (bottom(t.box) > safetyCfg.groundGuardY) { skipped.push({ track: t, reason: SKIP.GroundLevel }); continue; }
      const blockers = safetyOverlaps(box, safetyObjects, safetyCfg);
      if (blockers.length && !safetyCfg.cutOutSafety) {
        skipped.push({ track: t, reason: SKIP.SafetyOverlap, blocker: blockers[0] });
        continue;
      }
      cand.push({ t, box, cutouts: blockers });
    }
    cand.sort((p, q) => q.t.score - p.t.score);
    let used = 0;
    for (const c of cand) {
      if (used + area(c.box) > safetyCfg.maxTotalMaskArea) { skipped.push({ track: c.t, reason: SKIP.AreaBudget }); continue; }
      used += area(c.box);
      masks.push({ trackId: c.t.id, sourceBox: c.t.box, box: c.box, opacity: clamp01(safetyCfg.maxOpacity), cutouts: c.cutouts });
    }
    return { masks, skipped };
  }

  // ---------- Segmentation masks (YOLO-seg: output0 has 32 mask coefficients after the class scores) ----------
  // Returns {data: Float32Array (0..1), w, h, box} where box is the detection box in INPUT-normalized coords,
  // data covers exactly that box at prototype resolution, values are the mask probability.
  function maskFor(det, output, numClasses, numAnchors, protos, protoW, protoH) {
    const nm = protos.length / (protoW * protoH);
    const coef = new Float32Array(nm);
    for (let k = 0; k < nm; k++) coef[k] = output[(4 + numClasses + k) * numAnchors + det.anchor];
    const b = det.box;
    const x0 = Math.max(0, Math.floor(b.x * protoW)), x1 = Math.min(protoW, Math.ceil((b.x + b.w) * protoW));
    const y0 = Math.max(0, Math.floor(b.y * protoH)), y1 = Math.min(protoH, Math.ceil((b.y + b.h) * protoH));
    const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
    const data = new Float32Array(w * h);
    const plane = protoW * protoH;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const p = (y0 + y) * protoW + (x0 + x);
        let v = 0;
        for (let k = 0; k < nm; k++) v += coef[k] * protos[k * plane + p];
        data[y * w + x] = 1 / (1 + Math.exp(-v));
      }
    }
    return { data, w, h, box: rect(x0 / protoW, y0 / protoH, w / protoW, h / protoH) };
  }

  // ---------- Motion guard (rotation only on phones) ----------
  class MotionGuard {
    constructor(cfg) { this.cfg = cfg; this.factor = 1; }
    update(angularDegPerSec, dt) {
      if (!Number.isFinite(angularDegPerSec) || dt <= 0) { this.factor = 0; return 0; }
      const { angularSoftDegPerSec: soft, angularHardDegPerSec: hard } = this.cfg;
      const target = angularDegPerSec <= soft ? 1 : angularDegPerSec >= hard ? 0 : 1 - (angularDegPerSec - soft) / (hard - soft);
      this.factor = target < this.factor ? target : Math.min(target, this.factor + this.cfg.recoverPerSecond * dt);
      return this.factor;
    }
  }

  // ---------- Border colours (RGBA, top-down rows) ----------
  function sampleEdges(px, w, h, box, ring) {
    const gray = [0.5, 0.5, 0.5];
    if (!px || px.length < w * h * 4) return { left: gray, right: gray, top: gray, bottom: gray };
    const toPx = (v, n) => Math.min(n, Math.max(0, Math.round(v * n)));
    let x0 = toPx(box.x, w), x1 = toPx(right(box), w), y0 = toPx(box.y, h), y1 = toPx(bottom(box), h);
    if (x1 <= x0) x1 = Math.min(w, x0 + 1);
    if (y1 <= y0) y1 = Math.min(h, y0 + 1);
    const bx = Math.max(1, Math.round((x1 - x0) * ring)), by = Math.max(1, Math.round((y1 - y0) * ring));
    const avg = (ax0, ax1, ay0, ay1) => {
      ax0 = Math.max(0, ax0); ax1 = Math.min(w, ax1); ay0 = Math.max(0, ay0); ay1 = Math.min(h, ay1);
      let r = 0, g = 0, b = 0, n = 0;
      for (let y = ay0; y < ay1; y++) for (let x = ax0, i = (y * w + ax0) * 4; x < ax1; x++, i += 4) { r += px[i]; g += px[i + 1]; b += px[i + 2]; n++; }
      return n ? [r / 255 / n, g / 255 / n, b / 255 / n] : gray;
    };
    const bandX = (o0, o1, i0, i1) => (o0 < 0 || o1 > w ? avg(i0, i1, y0, y1) : avg(o0, o1, y0, y1));
    const bandY = (o0, o1, i0, i1) => (o0 < 0 || o1 > h ? avg(x0, x1, i0, i1) : avg(x0, x1, o0, o1));
    return {
      left: bandX(x0 - bx, x0, x0, x0 + bx),
      right: bandX(x1, x1 + bx, x1 - bx, x1),
      top: bandY(y0 - by, y0, y0, y0 + by),
      bottom: bandY(y1, y1 + by, y1 - by, y1),
    };
  }

  return { rect, iou, inset, intersectionArea, area, defaults, letterbox, unletterbox, decode, maskFor, guardRect, AdTracker, evaluatePolicy, MotionGuard, sampleEdges, SKIP };
});
