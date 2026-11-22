// ClearView labelling tool: photos + ad polygons stored in IndexedDB, exported as a YOLO-seg zip.
(() => {
  "use strict";
  const CLASSES = ["billboard", "poster", "logo", "ad_screen"];
  const MAX_SIDE = 1600;
  const $ = (id) => document.getElementById(id);
  const css = getComputedStyle(document.documentElement);
  const color = (cls) => css.getPropertyValue("--c-" + cls).trim() || "#fff";

  // ---------------- storage ----------------
  let db = null;
  const photos = [];
  const thumbUrls = new Map();

  function openDb() {
    return new Promise((resolve, reject) => {
      const r = indexedDB.open("clearview-labels", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("photos", { keyPath: "id" });
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
  }
  const store = (mode) => db.transaction("photos", mode).objectStore("photos");
  const req = (r) => new Promise((resolve, reject) => { r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });
  const save = (rec) => (db ? req(store("readwrite").put(rec)) : Promise.resolve());
  const remove = (id) => (db ? req(store("readwrite").delete(id)) : Promise.resolve());

  function msg(text) { $("msg").textContent = text || ""; }

  // ---------------- grid ----------------
  function renderGrid() {
    const grid = $("grid");
    grid.replaceChildren();
    let ads = 0;
    for (const rec of photos) {
      ads += rec.polys.length;
      if (!thumbUrls.has(rec.id)) thumbUrls.set(rec.id, URL.createObjectURL(rec.blob));
      const b = document.createElement("button");
      b.type = "button";
      b.className = "thumb";
      b.setAttribute("aria-label", `Foto con ${rec.polys.length} anuncios marcados`);
      const img = document.createElement("img");
      img.src = thumbUrls.get(rec.id);
      img.alt = "";
      const n = document.createElement("span");
      n.className = "n" + (rec.polys.length ? "" : " zero");
      n.textContent = rec.polys.length ? `${rec.polys.length} anuncio${rec.polys.length > 1 ? "s" : ""}` : "sin marcar";
      b.append(img, n);
      b.addEventListener("click", () => openEditor(rec));
      grid.appendChild(b);
    }
    $("stats").textContent = `${photos.length} foto${photos.length === 1 ? "" : "s"} · ${ads} anuncio${ads === 1 ? "" : "s"}`;
    $("export").disabled = photos.length === 0;
    $("empty").hidden = photos.length > 0;
  }

  async function decode(file) {
    try { return await createImageBitmap(file, { imageOrientation: "from-image" }); } catch (_) { return createImageBitmap(file); }
  }

  async function addFiles(files) {
    let added = 0;
    for (const f of files) {
      try {
        const bmp = await decode(f);
        const s = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
        const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        c.getContext("2d").drawImage(bmp, 0, 0, w, h);
        const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
        const rec = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 7), blob, w, h, polys: [], created: Date.now() };
        await save(rec);
        photos.push(rec);
        added++;
      } catch (err) {
        msg("No se pudo abrir una de las fotos: " + err.message);
      }
    }
    if (added) msg(`${added} foto${added > 1 ? "s" : ""} añadida${added > 1 ? "s" : ""}. Toca una para marcar sus anuncios.`);
    renderGrid();
  }

  // ---------------- editor ----------------
  const canvas = $("ed-canvas");
  const g = canvas.getContext("2d");
  let cur = null, bmp = null, cls = "billboard", freeMode = false, draft = [], sel = -1, drag = null, armedDelete = 0;

  async function openEditor(rec) {
    cur = rec;
    bmp = await createImageBitmap(rec.blob);
    canvas.width = rec.w;
    canvas.height = rec.h;
    draft = []; sel = -1; drag = null;
    $("editor").hidden = false;
    document.body.style.overflow = "hidden";
    updateUi();
    draw();
  }

  function closeEditor() {
    $("editor").hidden = true;
    document.body.style.overflow = "";
    cur = null;
    renderGrid();
  }

  const vr = () => Math.max(7, canvas.width / 110); // vertex radius in canvas px

  function draw() {
    if (!cur) return;
    const W = canvas.width, H = canvas.height, lw = Math.max(2, W / 400);
    g.drawImage(bmp, 0, 0);
    cur.polys.forEach((p, i) => {
      const c = color(p.cls);
      g.beginPath();
      p.pts.forEach(([x, y], k) => (k ? g.lineTo(x * W, y * H) : g.moveTo(x * W, y * H)));
      g.closePath();
      g.globalAlpha = i === sel ? 0.35 : 0.2;
      g.fillStyle = c;
      g.fill();
      g.globalAlpha = 1;
      g.lineWidth = i === sel ? lw * 2 : lw;
      g.strokeStyle = c;
      g.stroke();
      // Corner handles: always draggable, larger on the selected ad.
      for (const [x, y] of p.pts) {
        g.beginPath(); g.arc(x * W, y * H, i === sel ? vr() : vr() * 0.55, 0, Math.PI * 2);
        g.fillStyle = "#fff"; g.fill(); g.strokeStyle = c; g.lineWidth = lw; g.stroke();
      }
    });
    if (draft.length) {
      g.strokeStyle = "#fff"; g.lineWidth = lw; g.setLineDash([lw * 3, lw * 2]);
      g.beginPath();
      draft.forEach(([x, y], k) => (k ? g.lineTo(x * W, y * H) : g.moveTo(x * W, y * H)));
      g.stroke(); g.setLineDash([]);
      for (const [x, y] of draft) { g.beginPath(); g.arc(x * W, y * H, vr() * 0.7, 0, Math.PI * 2); g.fillStyle = color(cls); g.fill(); }
    }
  }

  function updateUi() {
    document.querySelectorAll("[data-cls]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.cls === cls)));
    $("mode").setAttribute("aria-pressed", String(freeMode));
    $("close-poly").classList.toggle("invisible", !(freeMode && draft.length >= 3));
    $("del-poly").disabled = sel < 0;
    $("hint").textContent = freeMode
      ? (draft.length ? `Sigue el borde del anuncio (${draft.length} puntos). Toca el primer punto o «Cerrar contorno».` : "Toca puntos alrededor del borde del anuncio.")
      : (draft.length ? `Esquina ${draft.length + 1} de 4.` : sel >= 0 ? "Anuncio seleccionado: cambia su tipo arriba, arrastra sus esquinas o bórralo." : "Toca las 4 esquinas de un anuncio. Toca uno ya marcado para cambiarlo.");
  }

  function toNorm(e) {
    const r = canvas.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  }

  function hitVertex(p) {
    const r = vr() / canvas.width * 1.6;
    const order = sel >= 0 ? [sel, ...cur.polys.keys()].filter((v, i, a) => a.indexOf(v) === i) : [...cur.polys.keys()];
    for (const i of order) {
      const pts = cur.polys[i].pts;
      for (let k = 0; k < pts.length; k++) {
        const dx = pts[k][0] - p[0], dy = (pts[k][1] - p[1]) * canvas.height / canvas.width;
        if (Math.hypot(dx, dy) < r) return { poly: i, idx: k };
      }
    }
    return null;
  }

  function inside(pts, [x, y]) {
    let c = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
    }
    return c;
  }

  // Taps can come in any order: sort the 4 corners around their centre so the quad never self-intersects.
  function orderQuad(pts) {
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    return [...pts].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  }

  function commitDraft() {
    const pts = freeMode ? draft : orderQuad(draft);
    cur.polys.push({ cls, pts });
    sel = -1; // so picking another class next starts a new ad instead of relabelling this one
    draft = [];
    save(cur);
    updateUi();
    draw();
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (!cur) return;
    e.preventDefault();
    const p = toNorm(e);
    if (!draft.length) {
      const v = hitVertex(p);
      if (v) { drag = v; sel = v.poly; canvas.setPointerCapture(e.pointerId); updateUi(); draw(); return; }
      const hit = cur.polys.findIndex((q) => inside(q.pts, p));
      if (hit >= 0 && hit !== sel) { sel = hit; updateUi(); draw(); return; }
    }
    if (freeMode && draft.length >= 3) {
      const [fx, fy] = draft[0];
      if (Math.hypot(fx - p[0], (fy - p[1]) * canvas.height / canvas.width) < vr() / canvas.width * 1.6) { commitDraft(); return; }
    }
    sel = -1;
    draft.push(p);
    if (!freeMode && draft.length === 4) commitDraft();
    else { updateUi(); draw(); }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    cur.polys[drag.poly].pts[drag.idx] = toNorm(e);
    draw();
  });
  const endDrag = () => { if (drag) { drag = null; save(cur); } };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  document.querySelectorAll("[data-cls]").forEach((b) => b.addEventListener("click", () => {
    cls = b.dataset.cls;
    if (sel >= 0) { cur.polys[sel].cls = cls; save(cur); }
    updateUi(); draw();
  }));
  $("mode").addEventListener("click", () => { freeMode = !freeMode; draft = []; updateUi(); draw(); });
  $("close-poly").addEventListener("click", commitDraft);
  $("undo").addEventListener("click", () => {
    if (draft.length) draft.pop();
    else if (cur.polys.length) { cur.polys.pop(); sel = -1; save(cur); }
    updateUi(); draw();
  });
  $("del-poly").addEventListener("click", () => {
    if (sel < 0) return;
    cur.polys.splice(sel, 1); sel = -1; save(cur); updateUi(); draw();
  });
  $("del-photo").addEventListener("click", async () => {
    const b = $("del-photo");
    if (Date.now() - armedDelete > 3000) {
      armedDelete = Date.now();
      b.textContent = "Toca otra vez para borrar";
      setTimeout(() => { b.textContent = "Borrar foto"; }, 3000);
      return;
    }
    armedDelete = 0;
    b.textContent = "Borrar foto";
    const id = cur.id;
    photos.splice(photos.findIndex((r) => r.id === id), 1);
    if (thumbUrls.has(id)) { URL.revokeObjectURL(thumbUrls.get(id)); thumbUrls.delete(id); }
    await remove(id);
    closeEditor();
  });
  $("done").addEventListener("click", closeEditor);

  // ---------------- export (YOLO segmentation format) ----------------
  function labelText(rec) {
    return rec.polys
      .filter((p) => p.pts.length >= 3)
      .map((p) => `${CLASSES.indexOf(p.cls)} ` + p.pts.map(([x, y]) => `${x.toFixed(6)} ${y.toFixed(6)}`).join(" "))
      .join("\n") + (rec.polys.length ? "\n" : "");
  }

  $("export").addEventListener("click", async () => {
    if (typeof JSZip === "undefined") { msg("No se pudo cargar el compresor (JSZip). Revisa la conexión y recarga."); return; }
    const btn = $("export");
    btn.disabled = true;
    msg("Preparando el zip…");
    try {
      const zip = new JSZip();
      for (const rec of photos) {
        zip.file(`images/all/${rec.id}.jpg`, rec.blob);
        zip.file(`labels/all/${rec.id}.txt`, labelText(rec));
      }
      zip.file("data.yaml", "names:\n" + CLASSES.map((c) => `- ${c}\n`).join(""));
      zip.file("LEEME.txt", "Fotos etiquetadas con ClearView (formato YOLO de segmentación).\nSúbelo en el paso 4 del cuaderno de Colab, en «Tus fotos».\n");
      const blob = await zip.generateAsync({ type: "blob" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `clearview_etiquetas_${new Date().toISOString().slice(0, 10)}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      msg(`Zip exportado con ${photos.length} fotos. Tus fotos siguen guardadas aquí por si quieres seguir añadiendo.`);
    } catch (err) {
      msg("No se pudo exportar: " + err.message);
    } finally {
      btn.disabled = false;
    }
  });

  $("add").addEventListener("change", (e) => { addFiles([...e.target.files]); e.target.value = ""; });

  // ---------------- start ----------------
  openDb()
    .then(async (d) => {
      db = d;
      photos.push(...(await req(store("readonly").getAll())).sort((a, b) => a.created - b.created));
      renderGrid();
    })
    .catch(() => {
      msg("Este navegador no deja guardar datos aquí: las fotos se perderán al cerrar la página. Exporta el zip antes de salir.");
      renderGrid();
    });

  window.__clearviewLabelTest = { photos, labelText }; // used by the automated browser test
})();
