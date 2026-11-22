// Mirrors the key C# safety tests so the web port cannot silently diverge.  Run: node --test web/test
const test = require("node:test");
const assert = require("node:assert/strict");
const C = require("../src/core.js");

const cfg = C.defaults();
const track = (id, box, o = {}) => ({ id, box, score: 0.9, hits: 5, lastSeen: 10, ...o });
const person = (box) => ({ box, classId: 5, score: 0.5, category: "safety" });
const evalP = (tracks, safety = [], s = cfg.safety) => C.evaluatePolicy(tracks, safety, 10, s, cfg.tracker);

test("hides a normal confirmed ad, inset, opacity < 1", () => {
  const r = evalP([track(1, C.rect(0.4, 0.2, 0.2, 0.2))]);
  assert.equal(r.masks.length, 1);
  assert.ok(r.masks[0].box.w < 0.2);
  assert.ok(r.masks[0].opacity < 1);
});

test("never hides unconfirmed or stale", () => {
  const r = evalP([track(1, C.rect(0.1, 0.1, 0.1, 0.1), { hits: 1 }), track(2, C.rect(0.5, 0.1, 0.1, 0.1), { lastSeen: 5 })]);
  assert.equal(r.masks.length, 0);
  assert.deepEqual(r.skipped.map((s) => s.reason).sort(), ["stale", "unconfirmed"]);
});

test("never covers a person, even partially, and reports the blocker", () => {
  const r = evalP([track(1, C.rect(0.3, 0.2, 0.3, 0.3))], [person(C.rect(0.55, 0.1, 0.1, 0.6))]);
  assert.equal(r.masks.length, 0);
  assert.equal(r.skipped[0].reason, "safetyOverlap");
  assert.ok(r.skipped[0].blocker);
});

test("distant small person still gets absolute margin", () => {
  const r = evalP([track(1, C.rect(0.3, 0.2, 0.2, 0.2))], [person(C.rect(0.505, 0.3, 0.01, 0.03))]);
  assert.equal(r.masks.length, 0);
});

test("large and ground-level ads stay visible", () => {
  assert.equal(evalP([track(1, C.rect(0.1, 0.1, 0.6, 0.6))]).skipped[0].reason, "tooLarge");
  assert.equal(evalP([track(1, C.rect(0.4, 0.75, 0.1, 0.15))]).skipped[0].reason, "groundLevel");
});

test("area budget keeps the most confident", () => {
  const s = { ...cfg.safety, maxTotalMaskArea: 0.05, insetFraction: 0 };
  const r = evalP([track(1, C.rect(0, 0.1, 0.2, 0.2), { score: 0.6 }), track(2, C.rect(0.5, 0.1, 0.2, 0.2), { score: 0.95 })], [], s);
  assert.equal(r.masks.length, 1);
  assert.equal(r.masks[0].trackId, 2);
});

test("decoder: safety threshold lower than ad threshold", () => {
  // anchors: (cx,cy,w,h,ad,person) x2, laid out [channel][anchor]
  const a = [[100, 100, 20, 20, 0.3, 0], [300, 100, 20, 20, 0, 0.3]];
  const t = new Float32Array(6 * 2);
  a.forEach((v, i) => v.forEach((x, c) => (t[c * 2 + i] = x)));
  const d = C.decode(t, 2, 2, 400, 200, ["ad", "safety"], cfg.decoder);
  assert.equal(d.length, 1);
  assert.equal(d[0].category, "safety");
});

test("tracker confirms after 3 hits with stable id", () => {
  const tr = new C.AdTracker(cfg.tracker);
  const det = (x) => [{ box: C.rect(x, 0.2, 0.1, 0.1), classId: 0, score: 0.9, category: "ad" }];
  tr.update(det(0.2), 0); const id = tr.tracks[0].id;
  tr.update(det(0.201), 0.1); tr.update(det(0.202), 0.2);
  assert.equal(tr.tracks.length, 1);
  assert.equal(tr.tracks[0].id, id);
  assert.equal(tr.tracks[0].hits, 3);
});

test("motion guard drops instantly, recovers slowly", () => {
  const g = new C.MotionGuard(cfg.motion);
  assert.equal(g.update(200, 0.016), 0);
  const f = g.update(0, 0.1);
  assert.ok(f > 0.1 && f < 0.2);
});

test("edge sampler reads outside each edge", () => {
  const w = 10, h = 10, px = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4; px[i] = x < 5 ? 255 : 0; px[i + 1] = y < 5 ? 255 : 0; px[i + 2] = x >= 5 ? 255 : 0;
  }
  const c = C.sampleEdges(px, w, h, C.rect(0.3, 0.3, 0.4, 0.4), 0.25);
  assert.equal(c.left[0], 1); assert.equal(c.right[2], 1); assert.equal(c.top[1], 1); assert.equal(c.bottom[1], 0);
});

test("defaults match the C# SafetyConfig", () => {
  const fs = require("node:fs");
  const cs = fs.readFileSync(__dirname + "/../../unity/Assets/ClearView/Core/SafetyPolicy.cs", "utf8");
  const val = (name) => parseFloat(cs.match(new RegExp(`public float ${name} = ([0-9.]+)f`))[1]);
  assert.equal(cfg.safety.insetFraction, val("InsetFraction"));
  assert.equal(cfg.safety.safetyMargin, val("SafetyMargin"));
  assert.equal(cfg.safety.safetyMarginMin, val("SafetyMarginMin"));
  assert.equal(cfg.safety.maxSingleMaskArea, val("MaxSingleMaskArea"));
  assert.equal(cfg.safety.maxTotalMaskArea, val("MaxTotalMaskArea"));
  assert.equal(cfg.safety.groundGuardY, val("GroundGuardY"));
  assert.equal(cfg.safety.maxOpacity, val("MaxOpacity"));
});

test("letterbox fits portrait frame into landscape input and maps boxes back", () => {
  const lb = C.letterbox(1080, 1920, 416, 320);
  assert.equal(lb.nh, 320);
  assert.equal(lb.nw, 180);
  assert.equal(lb.dx, 118);
  // a box covering exactly the image area in the input maps back to the whole frame
  const full = C.unletterbox(C.rect(lb.dx / 416, 0, lb.nw / 416, 1), lb);
  assert.ok(Math.abs(full.x) < 1e-6 && Math.abs(full.w - 1) < 1e-6 && Math.abs(full.h - 1) < 1e-6);
  // a box in the padding collapses to nothing after clamping
  assert.equal(C.area(C.unletterbox(C.rect(0, 0, 0.1, 0.1), lb)), 0);
});

test("letterbox is the identity when aspect ratios match", () => {
  const lb = C.letterbox(832, 640, 416, 320);
  const b = C.unletterbox(C.rect(0.25, 0.5, 0.1, 0.2), lb);
  assert.ok(Math.abs(b.x - 0.25) < 1e-6 && Math.abs(b.h - 0.2) < 1e-6);
});
