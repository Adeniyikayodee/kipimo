"use strict";
const $ = (id) => document.getElementById(id);
const fmtKES = (v) => "KES " + Math.round(v).toLocaleString("en-US");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const words = (s) => String(s).replace(/_/g, " ");
const b64u8 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const clock = (t) => { t = Math.max(0, Math.round(t)); return String(Math.floor(t / 60)).padStart(2, "0") + ":" + String(t % 60).padStart(2, "0"); };
const PILOT = { agents: "3 agents", agent: "agent", infogain: "planner", lawnmower: "lawnmower" };
const STATUS = { paid: "Paid", signoff: "With the actuary", rejected: "Rejected", refused: "Look again", surveying: "Surveying", review: "With a person", "": "Waiting" };
// A claim is paid once the actuary approves it. Runs recorded before sign-off count it paid when signed.
const payState = (c) => (!c ? "" : c.approved ? "paid" : c.rejected ? "rejected" : "signoff");
const REASON = { objective_met: "every parcel is confident enough", diminishing_returns: "more flying would not change a payout", distance_budget: "flight distance used up",
  battery_reserve: "battery at its reserve", replan_limit: "too many replans", insufficient_confidence: "not confident enough", no_trigger: "loss below the first tier", already_issued: "already paid" };
const reason = (r) => REASON[r] || words(r);
const who = (d) => d.role && d.role !== "field" ? d.role : "agent";

// ---------- state ----------
let S;
function fresh() {
  S = {
    world: null, pmap: null, truth: null, frame: null, layers: {},
    path: [], pose: null, wind: null, power: null, alt: 40, battery: null, battery0: null, distance: 0, quality: null,
    lanes: [], focus: new Set(), claims: new Map(), refused: new Map(), cases: new Map(), paidTotal: 0,
    uplink: { state: "online", depth: 0, bytes: 0, settled: 0, pkt: 0 },
    perception: null, sensorDegraded: false, events: 0, head: "", t: 0, policy: "", gate: 0.8,
    sortieStart: 0, done: false, first: null, thread: [], lang: {},
    approval: null, replay: false, review: new Map(), decided: [], busy: new Set(), rejecting: null, draft: "",
  };
}
fresh();
let layer = "belief", sel = null;
let dirty = true;
let pmapCanvas = null;

// ---------- colour ramps (sequential, light to dark) ----------
function ramp(stops) {
  return (v) => {
    for (let i = 1; i < stops.length; i++) {
      if (v <= stops[i][0]) {
        const [a, ca] = stops[i - 1], [b, cb] = stops[i], f = (v - a) / (b - a);
        return ca.map((c, k) => c + (cb[k] - c) * f);
      }
    }
    return stops[stops.length - 1][1];
  };
}
const lossColor = ramp([[0, [42, 48, 44]], [0.35, [104, 84, 48]], [0.65, [214, 128, 40]], [1, [255, 196, 128]]]);
const voiColor = ramp([[0, [26, 26, 28]], [0.3, [16, 58, 112]], [0.65, [10, 132, 255]], [1, [170, 214, 255]]]);
function gridCanvas(u8, n, colorFn, alphaU8, alphaMin) {
  const c = document.createElement("canvas"); c.width = n; c.height = n;
  const g = c.getContext("2d"), img = g.createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const v = u8[i] / 255, [r, gr, b] = colorFn(v);
    img.data[i * 4] = r; img.data[i * 4 + 1] = gr; img.data[i * 4 + 2] = b;
    const a = alphaU8 ? alphaMin + (1 - alphaMin) * (alphaU8[i] / 255) : 1;
    img.data[i * 4 + 3] = Math.round(255 * a);
  }
  g.putImageData(img, 0, 0);
  return c;
}

// ---------- map ----------
const cv = $("map"), ctx = cv.getContext("2d");
let size = 600, dpr = 1;
function resize() {
  const wrap = $("mapWrap");
  size = Math.max(280, Math.min(wrap.clientWidth - 16, wrap.clientHeight - 16));
  dpr = window.devicePixelRatio || 1;
  cv.width = size * dpr; cv.height = size * dpr;
  cv.style.width = size + "px"; cv.style.height = size + "px";
  pmapCanvas = null; dirty = true;
}
new ResizeObserver(resize).observe($("mapWrap")); // the legend, footer and panels move it, not only the window

let edgeCache = { pm: null, data: null };
function fieldEdges() {
  if (edgeCache.pm === S.pmap) return edgeCache.data;
  // each parcel's mask, blurred and upsampled, so borders follow the field, not the 4 m cells.
  // R: bund line, G: unplanted headland, B: inside a field
  const W = S.world, n = W.map_px, R = 1024, pm = S.pmap, cnv = (w) => Object.assign(document.createElement("canvas"), { width: w, height: w });
  const m = cnv(n), mg = m.getContext("2d"), img = mg.createImageData(n, n), up = cnv(R), ug = up.getContext("2d", { willReadFrequently: true }), out = new Uint8Array(R * R * 4);
  ug.filter = "blur(4px)";
  for (const id of new Set(pm)) {
    if (id === 255) continue;
    for (let i = 0; i < n * n; i++) img.data[i * 4 + 3] = pm[i] === id ? 255 : 0;
    mg.putImageData(img, 0, 0); ug.clearRect(0, 0, R, R); ug.drawImage(m, 0, 0, R, R);
    const a = ug.getImageData(0, 0, R, R).data;
    for (let j = 3; j < a.length; j += 4) {
      const d = Math.abs(a[j] - 128);
      out[j - 3] = Math.max(out[j - 3], 255 - d * 9);
      out[j - 2] = Math.max(out[j - 2], a[j] > 20 ? 255 - Math.max(0, a[j] - 128) * 2.2 : 0);
      out[j - 1] = Math.max(out[j - 1], a[j]);
    }
  }
  return (edgeCache = { pm: S.pmap, data: out }).data;
}
function buildParcelLines() {
  // borders from the blurred parcel masks, so they follow the field instead of the 4 m cells
  const e = fieldEdges(), R = 1024, src = document.createElement("canvas"); src.width = src.height = R;
  const sg = src.getContext("2d"), img = sg.createImageData(R, R);
  for (let j = 0; j < R * R; j++) { img.data.fill(255, j * 4, j * 4 + 3); img.data[j * 4 + 3] = e[j * 4] * 0.42; }
  sg.putImageData(img, 0, 0);
  const c = document.createElement("canvas"); c.width = c.height = size * dpr;
  const g = c.getContext("2d"); g.imageSmoothingQuality = "high"; g.drawImage(src, 0, 0, c.width, c.height);
  pmapCanvas = c;
}

const INK = "#f5f5f7", PAPER = "#1c1c1e", BLUE = "10,132,255", GREEN = "#30d158", YELLOW = "#ffd60a", RED = "255,69,58";
function draw() {
  dirty = false;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#111113"; ctx.fillRect(0, 0, cv.width, cv.height);
  const W = S.world;
  if (!W) return;
  const px = (size * dpr) / W.span_m; // world metres -> device pixels
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  const L = S.layers;
  const base = layer === "truth" ? L.truth : layer === "voi" ? L.voi : L.belief;
  if (base) ctx.drawImage(base, 0, 0, cv.width, cv.height);
  if (L.relief) { ctx.globalCompositeOperation = "soft-light"; ctx.drawImage(L.relief, 0, 0, cv.width, cv.height); ctx.globalCompositeOperation = "source-over"; }
  if (layer === "voi" && L.observed) { ctx.globalAlpha = 0.18; ctx.drawImage(L.observed, 0, 0, cv.width, cv.height); ctx.globalAlpha = 1; }
  if (!pmapCanvas && S.pmap) buildParcelLines();
  if (pmapCanvas) ctx.drawImage(pmapCanvas, 0, 0);

  // geofence
  const [fx0, fy0, fx1, fy1] = W.fence;
  ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.strokeStyle = "rgba(235,235,245,0.35)"; ctx.lineWidth = dpr;
  ctx.strokeRect(fx0 * px, fy0 * px, (fx1 - fx0) * px, (fy1 - fy0) * px); ctx.setLineDash([]);

  // no fly zones
  for (const z of W.no_fly) {
    ctx.beginPath(); ctx.arc(z.x * px, z.y * px, z.r * px, 0, 2 * Math.PI);
    ctx.fillStyle = `rgba(${RED},0.10)`; ctx.fill();
    ctx.strokeStyle = `rgba(${RED},0.9)`; ctx.lineWidth = 1.2 * dpr; ctx.stroke();
    ctx.save(); ctx.clip();
    ctx.strokeStyle = `rgba(${RED},0.45)`; ctx.lineWidth = dpr;
    for (let d = -z.r * 2; d < z.r * 2; d += 8) { ctx.beginPath(); ctx.moveTo((z.x + d - z.r) * px, (z.y + z.r) * px); ctx.lineTo((z.x + d + z.r) * px, (z.y - z.r) * px); ctx.stroke(); }
    ctx.restore();
    label("No fly: " + words(z.name || ""), z.x * px, (z.y + z.r) * px + 15 * dpr, INK, 11, true);
  }

  // planned lanes and the active swath
  ctx.setLineDash([4 * dpr, 4 * dpr]); ctx.lineWidth = dpr; ctx.strokeStyle = `rgba(${BLUE},0.7)`;
  for (const ln of S.lanes) { ctx.beginPath(); ctx.moveTo(ln.a[0] * px, ln.a[1] * px); ctx.lineTo(ln.b[0] * px, ln.b[1] * px); ctx.stroke(); }
  ctx.setLineDash([]);
  if (S.activeLane && !S.done) {
    const ln = S.activeLane; ctx.strokeStyle = `rgba(${BLUE},0.16)`; ctx.lineWidth = 2 * S.alt * S.world.fov_tan * px;
    ctx.lineCap = "butt"; ctx.beginPath(); ctx.moveTo(ln.a[0] * px, ln.a[1] * px); ctx.lineTo(ln.b[0] * px, ln.b[1] * px); ctx.stroke();
  }

  // flown path
  if (S.path.length > 1) {
    ctx.beginPath(); ctx.moveTo(S.path[0][0] * px, S.path[0][1] * px);
    for (const p of S.path) ctx.lineTo(p[0] * px, p[1] * px);
    ctx.strokeStyle = "rgba(245,245,247,0.9)"; ctx.lineWidth = 1.5 * dpr; ctx.lineJoin = "round"; ctx.stroke();
  }

  // parcel markers
  for (const p of W.parcels) {
    const x = p.cx * px, y = p.cy * px;
    const claimed = S.claims.has(p.id), refused = S.refused.has(p.id) && !claimed, focus = S.focus.has(p.id);
    if (p.id === sel) {
      ctx.beginPath(); ctx.arc(x, y, 17 * dpr, 0, 2 * Math.PI); ctx.fillStyle = `rgba(${BLUE},0.3)`; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, 13 * dpr, 0, 2 * Math.PI); ctx.fillStyle = INK; ctx.fill();
      label(p.id, x, y + 4 * dpr, PAPER, 10, true, true);
      continue;
    }
    ctx.beginPath(); ctx.arc(x, y, 12 * dpr, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(28,28,30,0.82)"; ctx.fill();
    ctx.lineWidth = (claimed || refused || focus ? 2 : 1) * dpr;
    ctx.strokeStyle = claimed ? GREEN : refused ? YELLOW : focus ? `rgb(${BLUE})` : "rgba(235,235,245,0.3)";
    ctx.stroke();
    label(p.id, x, y + 4 * dpr, INK, 10, true);
  }

  // home pad
  const [hx, hy] = W.home;
  ctx.fillStyle = INK; ctx.fillRect(hx * px - 7 * dpr, hy * px - 7 * dpr, 14 * dpr, 14 * dpr);
  label("H", hx * px, hy * px + 4 * dpr, PAPER, 10, true);

  // drone and camera footprint
  if (S.pose) {
    const { x, y, yaw } = S.pose, h = S.alt * W.fov_tan;
    ctx.strokeStyle = S.sensorDegraded ? YELLOW : INK;
    ctx.fillStyle = S.sensorDegraded ? "rgba(255,214,10,0.14)" : "rgba(255,255,255,0.08)";
    ctx.lineWidth = dpr;
    ctx.fillRect((x - h) * px, (y - h) * px, 2 * h * px, 2 * h * px);
    ctx.strokeRect((x - h) * px, (y - h) * px, 2 * h * px, 2 * h * px);
    ctx.save(); ctx.translate(x * px, y * px); ctx.rotate(yaw);
    ctx.beginPath(); ctx.moveTo(10 * dpr, 0); ctx.lineTo(-7 * dpr, 6 * dpr); ctx.lineTo(-4 * dpr, 0); ctx.lineTo(-7 * dpr, -6 * dpr); ctx.closePath();
    ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#000"; ctx.stroke();
    ctx.restore();
  }
  windBadge();
}
function windBadge() {
  if (!S.wind) return;
  const [wx, wy] = S.wind, v = Math.hypot(wx, wy), r = 16 * dpr, cx = cv.width - r - 14 * dpr, cy = cv.height - r - 30 * dpr;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fillStyle = "rgba(28,28,30,0.82)"; ctx.fill();
  ctx.strokeStyle = "rgba(235,235,245,0.3)"; ctx.lineWidth = dpr; ctx.stroke();
  if (v > 0.1) {
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(Math.atan2(wy, wx));
    ctx.beginPath(); ctx.moveTo(-r * 0.6, 0); ctx.lineTo(r * 0.55, 0); ctx.moveTo(r * 0.25, -r * 0.3); ctx.lineTo(r * 0.6, 0); ctx.lineTo(r * 0.25, r * 0.3);
    ctx.strokeStyle = INK; ctx.lineWidth = 1.6 * dpr; ctx.lineCap = "round"; ctx.stroke(); ctx.restore();
  }
  label(`Wind ${v.toFixed(1)} m/s`, cx, cy + r + 14 * dpr, INK, 10);
}
function label(text, x, y, color, sz, bold, flat) {
  ctx.font = `${bold ? 600 : 400} ${sz * dpr}px -apple-system, "Segoe UI", sans-serif`;
  ctx.textAlign = "center";
  if (!flat) { ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineJoin = "round"; ctx.strokeText(text, x, y); }
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}
function loop() { if (dirty) draw(); requestAnimationFrame(loop); }

function legend() {
  const grad = (fn) => { const c = []; for (let i = 0; i <= 10; i++) { const [r, g, b] = fn(i / 10); c.push(`rgb(${r | 0},${g | 0},${b | 0}) ${i * 10}%`); } return `linear-gradient(90deg, ${c.join(",")})`; };
  const keys = `<span class="key"><i class="sw" style="border-top: 2px solid ${INK}"></i>flight path</span>
    <span class="key"><i class="sw" style="border-top: 1px dashed rgb(${BLUE})"></i>planned</span>
    <span class="key"><i class="sw" style="border-top: 2px solid ${GREEN}"></i>paid</span>
    <span class="key"><i class="sw" style="border-top: 2px solid ${YELLOW}"></i>look again</span>`;
  if (layer === "voi") {
    $("legend").innerHTML = `Payout money a photo could change: low <span class="ramp" style="background:${grad(voiColor)}"></span> high ${keys}`;
    $("mapNote").textContent = S.done ? "Money a photo could still change" : "Bright where to look next";
  } else {
    $("legend").innerHTML = `Crop loss: 0% <span class="ramp" style="background:${grad(lossColor)}"></span> 100% ${keys}`;
    $("mapNote").textContent = layer === "truth" ? "Simulator only. The drone never sees this" : "Faded where the drone hasn't looked yet";
  }
}

// ---------- selection: one parcel, highlighted on the map, the table and in 3D ----------
function select(id) {
  sel = id && id !== sel ? id : null;
  renderParcels(); renderClaims(); dirty = true;
  window.view3d?.select(sel);
  if (sel) document.querySelector(`#parcels tr[data-id="${sel}"]`)?.scrollIntoView({ block: "nearest", behavior: calm ? "auto" : "smooth" });
}
function parcelAtPx(e) {
  if (!S.world || !S.pmap) return null;
  const r = cv.getBoundingClientRect(), n = S.world.map_px;
  const j = Math.floor(((e.clientX - r.left) / r.width) * n), i = Math.floor(((e.clientY - r.top) / r.height) * n);
  const k = i >= 0 && j >= 0 && i < n && j < n ? S.pmap[i * n + j] : 255;
  return k === 255 ? null : S.world.parcels[k];
}
cv.addEventListener("click", (e) => { const p = parcelAtPx(e); select(p ? p.id : null); });
cv.addEventListener("mousemove", (e) => { const p = parcelAtPx(e); cv.style.cursor = p ? "pointer" : ""; cv.title = p ? `${p.id}, ${p.holder}` : ""; });
for (const id of ["parcels", "claims"]) {
  $(id).addEventListener("click", (e) => { const r = e.target.closest("[data-id]"); if (r) select(r.dataset.id); });
  $(id).addEventListener("keydown", (e) => { const r = e.target.closest("[data-id]"); if (r && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); e.stopPropagation(); select(r.dataset.id); } });
}
$("shClose").onclick = () => select(null);
function renderSheet() {
  const sh = $("sheet"), W = S.world, q = sel && W && W.parcels.find((p) => p.id === sel);
  sh.hidden = !q;
  if (!q) return;
  const row = S.frame?.situation.parcels.find((p) => p.id === sel), c = S.claims.get(sel), k = S.cases.get(sel);
  const st = c ? payState(c) : k && k.steps.at(-1).tone === "bad" ? "review" : S.refused.has(sel) ? "refused" : S.focus.has(sel) ? "surveying" : "";
  const pct = (v) => Math.round(v * 100) + "%", gate = pct(S.gate);
  $("shTitle").textContent = `${q.id} · ${q.holder}`;
  $("shSub").textContent = `${q.crop}, planted ${q.age_days} days before the flood, insured for ${fmtKES(q.sum_insured)}`;
  $("shPill").className = "pill " + st;
  $("shPill").textContent = STATUS[st];
  if (!row) { $("shBody").innerHTML = '<p class="reason">Details appear once the drone starts its survey.</p>'; return; }
  const why = c && !c.approved ? (c.rejected ? "The actuary rejected this claim. Nothing was paid." : `A claim for ${fmtKES(c.amount)} is signed. It is paid once the actuary approves it.`)
    : c ? `Paid at the ${pct(c.tier)} tier. The drone measured ${pct(row.loss)} crop loss and was ${pct(c.confidence)} sure, above the ${gate} bar.`
    : S.refused.has(sel) ? `Held back at ${pct(S.refused.get(sel))} confidence, below the ${gate} bar. The drone goes back for a closer look.`
    : row.confidence >= S.gate ? (row.payout_at_estimate ? `Confident enough at ${pct(row.confidence)}. A claim for ${fmtKES(row.payout_at_estimate)} will be signed.` : `Confident at ${pct(row.confidence)} that the loss is below the first payout tier. Nothing is owed.`)
    : `The drone has seen ${pct(row.coverage)} of this field. A payout needs ${gate} confidence; it is at ${pct(row.confidence)}.`;
  const steps = k ? k.steps.map((x) => `<li class="${x.tone}">${esc(x.text)}</li>`).join("") : "";
  $("shBody").innerHTML = `<div class="facts">
      <div><small>Crop loss</small><b>${pct(row.loss)}</b><span>likely ${pct(row.ci95[0])} to ${pct(row.ci95[1])}</span></div>
      <div><small>Confidence</small><b>${pct(row.confidence)}</b><span>pays at ${gate}</span></div>
      <div><small>${c?.approved ? "Paid" : c ? "Claim" : "Payout"}</small><b>${c ? fmtKES(c.amount) : row.payout_at_estimate ? fmtKES(row.payout_at_estimate) : "None"}</b><span>${c ? (c.status === "sent" ? "delivered by SMS" : "queued to send") : "if the estimate holds"}</span></div>
    </div>
    <p class="reason">${why}</p>
    ${steps ? `<div class="sub-h">Claim</div><ul class="timeline">${steps}${c ? `<li class="${c.status === "sent" ? "ok" : ""}">${c.claim_id}, ${c.bytes ? c.bytes + " bytes, " : ""}${c.status === "sent" ? "delivered" : "waiting for signal"}</li>` : ""}</ul>` : ""}`;
}

// ---------- log ----------
function log(cls, src, html, t) {
  const el = document.createElement("div");
  el.className = "ev " + cls;
  el.innerHTML = `<span class="t">${clock(t ?? S.t)}</span><span class="src">${src}</span><span class="msg">${html}</span>`;
  const box = $("traceScroll"), stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  $("log").appendChild(el);
  if (stick) box.scrollTop = box.scrollHeight;
}
function toolSummary(name, inp) {
  if (name === "survey") return `<span class="fn">survey</span>${[].concat(inp.parcel_ids ?? []).map(esc).join(", ")} to ${(+inp.target_confidence).toFixed(2)}, max ${Math.round(inp.max_distance_m)} m<span class="why">${esc(inp.reason)}</span>`;
  if (name === "issue_claim") return `<span class="fn">issue_claim</span>${esc(inp.parcel_id)}<span class="why">${esc(inp.justification)}</span>`;
  if (name === "score_payout" || name === "get_priors") return `<span class="fn">${name}</span>${esc(inp.parcel_id)}`;
  return `<span class="fn">${esc(name)}</span>`;
}
function status(text) { $("status").textContent = text || "Ready"; }

// ---------- events ----------
function onEvent(ev) {
  const d = ev.data || {};
  if (typeof ev.t === "number" && ev.seq >= 0) S.t = ev.t;
  if (ev.hash) { S.events = ev.seq + 1; S.head = ev.hash; }
  switch (ev.kind) {
    case "HELLO": hello(d); return;
    case "SESSION_ERROR": status(`Run stopped: ${d.error}`); return;
    case "COMMAND_ERROR": status(`Not done: ${ev.data}`); return;
    case "SESSION_RESET":
      fresh(); $("finish").hidden = true; $("log").innerHTML = ""; paused = false; armed = false; unseen = 0; badge(); renderClaims(); renderReview(); S.policy = d.policy; S.replay = !!d.replay; $("overlay").style.display = "none";
      if (d.replay) log("", "replay", `${esc(words(d.replay))}, recorded run`);
      break;
    case "EPISODE_START": S.policy = d.policy; S.approval = d.approval || null; renderReview(); break;
    case "WORLD_INFO":
      if (d.approval) { S.approval = d.approval; renderReview(); }
      if (sel && !d.parcels.some((p) => p.id === sel)) select(null);
      S.world = d; S.pmap = b64u8(d.parcel_map); pmapCanvas = null;
      S.layers.truth = gridCanvas(b64u8(d.truth), d.grid, lossColor);
      window.view3d?.world(d);
      S.layers.relief = d.relief ? gridCanvas(b64u8(d.relief), d.relief_grid, (v) => { const g = Math.max(0, Math.min(255, 128 + (v - 0.71) * 600)); return [g, g, g]; }) : null;
      S.sensor = d.sensor;
      status(S.policy === "agent" ? "Agent reading the parcel table" : "Ranking parcels by money at risk");
      if (d.site && d.site !== "synthetic") log("", "terrain", `${esc(d.site)}, Copernicus 30 m elevation`);
      log("", "mission", `${d.parcels.length} parcels, pilot ${PILOT[S.policy] || S.policy}, camera ${d.sensor}`);
      break;
    case "FRAME": frame(d); break;
    case "OBSERVATION":
      S.pose = d.pose; S.alt = d.alt; S.battery = d.battery_wh; if (S.battery0 === null) S.battery0 = d.battery_wh;
      S.distance = d.distance_m; S.quality = d.quality; S.wind = d.wind || null; S.power = d.power_w ?? null;
      S.path.push([d.pose.x, d.pose.y]);
      break;
    case "PLAN_BATCH": {
      S.lanes = (d.path || []).map(([a, b]) => ({ a, b })); S.focus = new Set(d.parcels);
      if (S.policy !== "agent") log("", "planner", `Planned ${d.lanes} passes over ${d.parcels.length} parcels`);
      status(`Surveying ${d.parcels.length > 6 ? d.parcels.length + " parcels" : d.parcels.join(", ")}`);
      break;
    }
    case "PLAN_LANE": S.activeLane = { a: d.a, b: d.b }; break;
    case "SURVEY_END": if (S.policy !== "agent") log("", "planner", `Pass finished: ${reason(d.reason)}`); break;
    case "AGENT_START": log("agent", "agent", `${esc(d.model)}, limit ${d.max_tool_calls} tool calls`); break;
    case "AGENT_THINKING": log("agent think", who(d), esc(d.text.length > 420 ? d.text.slice(0, 420) + "…" : d.text)); break;
    case "AGENT_TEXT": log("agent", who(d), esc(d.text)); status(d.text.length > 140 ? d.text.slice(0, 140) + "…" : d.text); break;
    case "AGENT_TOOL_CALL": log("agent", who(d), toolSummary(d.name, d.input)); if (d.name === "survey") { S.sortieStart = S.distance; S.focus = new Set(d.input.parcel_ids); } break;
    case "AGENT_TOOL_RESULT": toolResult(d); break;
    case "OBJECTIVE_ACCEPTED": log("", "envelope", `accepted, est. ${(d.estimated_distance_m / 1000).toFixed(2)} km`); break;
    case "OBJECTIVE_REJECTED": log("bad", "envelope", `<b>rejected</b>: ${esc(d.reason)}`); break;
    case "AGENT_BUDGET_EXHAUSTED": log("warn", "agent", `${d.tool_calls} tool calls used, planner takes over`); break;
    case "AGENT_FALLBACK": log("warn", who(d), `model unavailable (${esc(words(d.reason))}), rules take over`); if (!d.role || d.role === "field" || d.role === "all") status("Model unavailable, rules flying"); break;
    case "AGENT_REPORT": log("agent", "agent", esc(d.summary)); break;
    case "SMS_RECEIVED": S.lang[d.parcel] = d.lang; log("", "sms in", `${d.parcel} (${d.lang}): <i>${esc(d.text)}</i>`); break;
    case "FARMER_REPORT": log(d.suspicious ? "bad" : d.conflict ? "warn" : "", "farmer", `${d.parcel} reports <b>${d.severity}</b>${d.where && d.where !== "unknown" ? ", " + words(d.where) : ""}${d.suspicious ? ", flagged suspicious" : ""}${d.discounted ? ", discounted: record of overstating" : d.doubted ? ", conflicts with estimate: flagged for a look" : ""}`); break;
    case "FLIGHT_SET": log("", "flight", `${d.altitude_m} m, ${d.speed_mps} m/s, camera ${d.quality.toFixed(2)}<span class="why">${esc(d.reason)}</span>`); break;
    case "WAITED": log("warn", "weather", `holding ${d.minutes} min at the pad${d.wind != null ? `, wind ${d.wind} m/s` : ""}<span class="why">${esc(d.reason)}</span>`); status(d.wind != null ? `Holding at the pad for the flight window, wind ${d.wind} m/s` : "Holding at the pad"); break;
    case "CLAIM_AWAITING_APPROVAL": {
      S.review.set(d.case_id, { ...d, t: S.t });
      const f = d.facts, n = d.flags.length;
      log(n ? "warn" : "", "actuary", d.subject === "hold" ? `${d.parcel} <b>waits for the actuary</b>: the adjuster still holds it under ${esc(d.rule_id)}` : `${d.parcel} reached the insurer: <b>${fmtKES(f.amount)}</b> waits for sign-off${n ? `, ${n} flag${n > 1 ? "s" : ""}` : ", no flags"}`);
      if (tab !== "actuary") status(`${S.review.size} claim${S.review.size > 1 ? "s" : ""} waiting for the actuary`);
      renderReview(); break;
    }
    case "ACTUARY_DECISION": {
      const x = d.decision, ok = x.decision === "approve", c = S.claims.get(x.parcel_id);
      S.review.delete(x.case_id); S.busy.delete(x.case_id); if (S.rejecting === x.case_id) S.rejecting = null;
      S.decided.push(x);
      if (x.subject === "hold") {
        step(x.parcel_id, ok ? `hold released by ${x.by}` : `hold kept by ${x.by}: ${x.note}`, ok ? "ok" : "bad");
        log(ok ? "ok" : "bad", "actuary", `${x.parcel_id} hold <b>${ok ? "released" : "kept"}</b> by ${esc(x.by)}${x.note ? `<span class="why">${esc(x.note)}</span>` : ""}`);
      } else if (c && ok) {
        c.approved = true; c.fresh = true; S.paidTotal += c.amount;
        S.thread.push({ pid: x.parcel_id, sms: true }); if (tab !== "phones") { unseen++; badge(); }
        step(x.parcel_id, `approved by ${x.by}, paid`, "ok");
        log("ok", "actuary", `<b>Approved</b> ${x.parcel_id}: paid ${esc(c.holder)} <b>${fmtKES(c.amount)}</b>. Signed by ${esc(x.by)}`);
      } else if (c) {
        c.rejected = true;
        step(x.parcel_id, `rejected by ${x.by}: ${x.note}`, "bad");
        log("bad", "actuary", `<b>Rejected</b> ${x.parcel_id} by ${esc(x.by)}<span class="why">${esc(x.note)}</span>`);
      }
      status(S.review.size ? `${S.review.size} claim${S.review.size > 1 ? "s" : ""} waiting for the actuary` : "Every claim has a decision");
      renderClaims(); renderReview(); break;
    }
    case "CLAIM_ESCALATED": step(d.parcel, "with a person", "bad"); log("bad", "adjuster", `${d.parcel} <b>handed to a person</b> after ${d.rounds} rounds under ${esc(d.rule_id)}`); break;
    case "MEMORY_UPDATED": log("", "memory", `learned from ${d.scored} settled reports, ${d.farmers} farmers on record`); break;
    case "SMS_SENT": log("ok", "sms out", `${d.parcel}: ${esc(d.text)}`); break;
    case "SMS_REJECTED": log("bad", "sms out", `${d.parcel}: <b>blocked</b>, quoted an amount not issued`); break;
    case "CLAIM_REVIEW": step(d.parcel, d.decision === "approve" ? `approved, round ${d.round || 1}` : `challenged ${(d.rule_id || "").split("_")[0]}, round ${d.round || 1}`, d.decision === "approve" ? "" : "warn"); log(d.decision === "approve" ? "ok" : "warn", "adjuster", `${d.parcel} <b>${d.decision}</b>${d.rule_id ? " under " + esc(d.rule_id) : ""}${d.round > 1 ? `, round ${d.round}` : ""}`); break;
    case "CHALLENGE_REJECTED": log("bad", "adjuster", `${d.parcel}: challenge under ${esc(d.rule_id)} <b>rejected</b>, rule does not hold`); break;
    case "CLAIM_REFUSED": {
      const sc = d.score || {};
      if (d.reason === "insufficient_confidence") {
        S.refused.set(d.parcel, sc.confidence);
        step(d.parcel, `held back at ${Math.round(sc.confidence * 100)}% confidence`, "warn");
        log("warn", "payout", `<b>Held back</b> ${d.parcel}: ${Math.round(sc.confidence * 100)}% sure, needs ${Math.round(sc.min_confidence * 100)}%`);
        status(`${d.parcel} needs another look`);
      } else log("", "payout", `${d.parcel} not paid: ${reason(d.reason)}`);
      break;
    }
    case "CLAIM_ISSUED": {
      const c = d.claim; S.first ??= S.t; S.claims.set(c.parcel_id, { ...c, pubkey: d.pubkey, status: "queued", bytes: null, verified: null, fresh: true, approved: !S.approval });
      if (S.approval) {
        step(c.parcel_id, "claim signed, sent to the actuary", "ok");
        log("ok", "payout", `Signed ${esc(c.holder)}'s claim for <b>${fmtKES(c.amount)}</b> on ${c.parcel_id}, ${Math.round(c.tier * 100)}% tier, ${Math.round(c.confidence * 100)}% sure. The actuary signs off before money moves`);
        renderClaims(); break;
      }
      S.paidTotal += c.amount;
      S.thread.push({ pid: c.parcel_id, sms: true }); if (tab !== "phones") { unseen++; badge(); }
      step(c.parcel_id, "claim signed", "ok");
      log("ok", "payout", `Paid ${esc(c.holder)} <b>${fmtKES(c.amount)}</b> for ${c.parcel_id}, ${Math.round(c.tier * 100)}% tier, ${Math.round(c.confidence * 100)}% sure`);
      renderClaims(); break;
    }
    case "UPLINK_QUEUED": { const c = claimById(d.claim_id); if (c) { c.bytes = d.bytes; c.status = "queued"; check(c, d.packet); } S.uplink.depth = d.depth; S.uplink.pkt = Math.max(S.uplink.pkt, d.bytes); renderClaims(); break; }
    case "UPLINK_SENT": { const c = claimById(d.claim_id); if (c) c.status = "sent"; S.uplink.depth = d.depth; S.uplink.bytes = d.total_bytes; S.uplink.settled++; renderClaims(); break; }
    case "LINK_STATE":
      S.uplink.state = d.state; S.uplink.depth = d.depth; setLinkButtons(d.state);
      if (d.state === "offline") log("bad", "link", "<b>offline</b>, claims will queue");
      else if (d.state === "degraded") log("warn", "link", "<b>weak signal</b>, retrying in order");
      else log("ok", "link", `online, sending ${d.depth} queued`);
      break;
    case "SENSOR_STATE":
      S.sensorDegraded = d.degraded; $("degrade").classList.toggle("on", d.degraded);
      log(d.degraded ? "warn" : "", "camera", d.degraded ? "<b>blurred</b>, image quality 30%" : "restored");
      break;
    case "DEGRADED_SENSOR_REPLAN": log("warn", "planner", `Replanned for ${Math.round(d.quality * 100)}% image quality`); break;
    case "FLOOD_ALERT": log("warn", "flood hub", `${esc(d.gauge)} passes danger level ${d.danger_m} m at ${String(d.local_hour).padStart(2, "0")}:00, ${d.lead_h} h ahead. No farmer had to call`); break;
    case "FLOOD_WATCH": log("", "flood hub", `${esc(d.gauge)} peaks at ${d.peak_m} m, below danger ${d.danger_m} m. No flight`); break;
    case "FLIGHT_BOOKED": log("", "dispatch", d.in_h ? `Flight booked for the first calm window, in ${d.in_h} h` : "Flight booked: the drone leaves the hub now"); break;
    case "RETURN_TO_HOME": log("warn", "planner", `Returning home: ${esc(reason(d.reason))}`); break;
    case "EPISODE_END":
      if (!SITE) loadRecordings();
      S.done = true; S.focus = new Set(); S.lanes = []; S.activeLane = null;
      log("ok", "mission", `Complete: ${(d.distance_m / 1000).toFixed(1)} km, ${d.claims} ${S.approval ? "claims signed" : "paid"}, payout error ${d.payout_mae_pct.toFixed(2)}% of sum insured`);
      status(S.approval ? `Mission complete. ${d.claims} claims go to the actuary` : `Mission complete. ${d.claims} farmers paid`);
      finish(d); legend();
      break;
  }
  dirty = true;
}
function finish(d) {
  const min = Math.max(1, Math.round(d.duration_s / 60)), plural = (k, w) => `${k} ${w}${k === 1 ? "" : "s"}`;
  $("finLede").textContent = !d.claims ? `No payouts were due. Checked in ${min} minutes.`
    : S.approval ? `${plural(d.claims, "claim")} signed in ${min} minutes. Each one is paid once the actuary approves it.`
    : `${plural(d.claims, "farmer")} paid ${fmtKES(S.paidTotal)} in ${min} minutes.`;
  $("finGrid").innerHTML = [
    [d.claims, S.approval ? "claims signed" : "farmers paid"], [d.first_claim_s == null ? "-" : Math.max(1, Math.round(d.first_claim_s / 60)) + " min", S.approval ? "to first claim" : "to first payout"], [(d.distance_m / 1000).toFixed(1) + " km", "flown"],
    [d.missed_claims ?? "-", "owed but missed"], [d.wrong_claims ?? "-", "wrong amounts"], [d.payout_mae_pct.toFixed(2) + "%", "payout error"],
  ].map(([v, k]) => `<div><b>${v}</b><small>${k}</small></div>`).join("");
  $("finish").hidden = false;
}
$("finClose").onclick = () => ($("finish").hidden = true);
$("finAgain").onclick = () => $("start").click();
function step(pid, text, tone) {
  const c = S.cases.get(pid) || { steps: [] };
  S.cases.delete(pid);
  c.steps.push({ text, tone });
  S.cases.set(pid, c);
  if (tone === "warn" || tone === "bad") S.thread.push({ pid, text, tone });
  renderClaims();
}
// The browser checks each claim packet itself: ed25519 over the exact CBOR body, as the insurer does.
const hexU8 = (h) => Uint8Array.from(h.match(/../g) || [], (x) => parseInt(x, 16));
async function check(c, hex) {
  try {
    const u = hexU8(hex), long = u[1] === 0x58, at = long ? 3 : 2, n = long ? u[2] : u[1] - 0x40, end = at + n;
    if (u[0] !== 0x82 || u[end] !== 0x58 || u[end + 1] !== 0x40) throw new Error("not a claim packet");
    const key = await crypto.subtle.importKey("raw", hexU8(c.pubkey), { name: "Ed25519" }, false, ["verify"]);
    c.verified = await crypto.subtle.verify({ name: "Ed25519" }, key, u.slice(end + 2, end + 66), u.slice(at, end));
  } catch { c.verified = "n/a"; } // no Ed25519 in this browser
  renderClaims();
}
const SMS = {
  sw: (n, a, p) => `Habari ${n}. Dai lako la <em>${a}</em> kwa shamba ${p} limeidhinishwa.`,
  en: (n, a, p) => `Hello ${n}. Your claim of <em>${a}</em> for field ${p} is approved.`,
};
let shownAmount = 0;
function countUp() {
  const from = shownAmount, to = S.paidTotal, t0 = performance.now();
  shownAmount = to;
  if (calm || to <= from) { $("hAmount").textContent = fmtKES(to); return; }
  const f = (now) => { const k = Math.min(1, (now - t0) / 700), e = 1 - (1 - k) ** 3; $("hAmount").textContent = fmtKES(from + (to - from) * e); if (k < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}
function claimById(id) { for (const c of S.claims.values()) if (c.claim_id === id) return c; return null; }
function toolResult(d) {
  const r = d.result || {};
  if (r.error) { log("bad", "agent", `<b>${esc(d.name)}</b>: ${esc(r.error)}`); return; }
  if (d.name === "survey" && r.accepted) {
    const ps = r.parcels.map((p) => `${p.id} ${Math.round(p.confidence * 100)}%`).join(", ");
    log("", "drone", `Flew ${(r.flown_m / 1000).toFixed(2)} km, ${reason(r.stop_reason)}. ${ps}`);
  } else if (d.name === "get_situation") {
    log("", "belief", "Most money in doubt: " + r.parcels.slice(0, 3).map((p) => `${p.id} ${fmtKES(p.money_at_risk)}`).join(", "));
  } else if (d.name === "score_payout") {
    log("", "payout", `${r.parcel_id} would pay ${fmtKES(r.amount)}, ${Math.round(r.tier * 100)}% tier`);
  } else if (d.name === "get_priors") {
    log("", "priors", `${r.parcel_id} rain +${r.rainfall_anomaly_mm} mm, NDVI ${r.ndvi_delta}, radar loss ${r.radar_loss_prior}`);
  }
}
function frame(d) {
  S.frame = d; S.gate = d.situation.gate_min_confidence; $("gateV").textContent = Math.round(S.gate * 100) + "%";
  const n = S.world ? S.world.grid : 64;
  S.layers.belief = gridCanvas(b64u8(d.mean), n, lossColor, b64u8(d.observed), 0.35);
  S.layers.voi = gridCanvas(b64u8(d.voi), n, voiColor);
  const obs = b64u8(d.observed); S.layers.observed = gridCanvas(obs, n, () => [255, 255, 255], obs, 0);
  S.perception = d.perception; S.uplink = { ...S.uplink, ...d.uplink, bytes: d.uplink.bytes_sent, pkt: d.uplink.packet_bytes };
  setLinkButtons(d.uplink.state);
  renderParcels();
  renderSheet();
  conditions(d.conditions);
}
function conditions(c) {
  if (!c) return;
  const w = c.weather, f = c.flight, cam = c.camera;
  const cls = (v, warn, bad) => (v >= bad ? "bad" : v >= warn ? "warn" : "");
  const tone = { "": "var(--label)", warn: "var(--yellow)", bad: "var(--red)" };
  let wind = "";
  if (w) {
    const k = cls(w.wind_mps / w.limit_mps, 0.8, 1);
    const bars = w.calm_share_next_hours.map((p) => `<i style="height:${Math.max(2, p * 14)}px;background:${p >= 0.95 ? "var(--label)" : "var(--label-3)"}"></i>`).join("");
    const when = w.next_window_in_min == null ? "no window in range" : w.next_window_in_min === 0 ? "flyable now" : `window in ${Math.floor(w.next_window_in_min / 60)} h ${w.next_window_in_min % 60} min`;
    wind = `<span>Wind<b class="${k}">${w.wind_mps.toFixed(1)} / ${w.limit_mps} m/s</b><span class="meter"><i style="width:${Math.min(100, (100 * w.wind_mps) / w.limit_mps)}%;background:${tone[k]}"></i></span></span>
      <span>Forecast<span class="spark">${bars}</span><b>${when}</b></span>`;
  }
  const compass = (deg) => ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(deg / 45) % 8];
  const air = c.wind_at_drone ? `<span>Wind at drone<b>${c.wind_at_drone.mps.toFixed(1)} m/s ${compass(c.wind_at_drone.from_deg)}</b></span>`: "";
  $("cond").innerHTML = `${wind}${air}<span>Flying<b>${f.altitude_m} m up, ${f.speed_mps} m/s</b></span>
    <span>Image quality<b class="${cam.degraded ? "warn" : ""}">${Math.round(cam.quality * 100)}%${cam.degraded ? ", blurred" : ""}</b></span>`;
}
function renderParcels() {
  const f = S.frame; if (!f) return;
  const W = S.world, holders = {}; if (W) for (const p of W.parcels) holders[p.id] = p.holder;
  $("parcels").innerHTML = f.situation.parcels.map((p) => {
    const conf = p.confidence, gate = S.gate, pass = conf >= gate;
    const claimed = S.claims.has(p.id), refused = S.refused.has(p.id) && !claimed, focus = S.focus.has(p.id);
    const c = S.claims.get(p.id), st = claimed ? payState(c) : refused ? "refused" : focus ? "surveying" : "";
    const lo = p.ci95[0] * 100, hi = p.ci95[1] * 100;
    return `<tr data-id="${p.id}" tabindex="0" class="${focus ? "focus" : ""}${p.id === sel ? " sel" : ""}" aria-selected="${p.id === sel}"><td><b>${p.id}</b> <span class="muted crop">${p.crop}</span></td><td>${esc(holders[p.id] || "")}</td>
      <td><div style="display:flex;align-items:center;gap:8px"><div class="ci"><i style="left:${lo}%;width:${Math.max(1, hi - lo)}%"></i><b style="left:${p.loss * 100}%"></b></div><span class="mono">${Math.round(p.loss * 100)}%</span></div></td>
      <td><div style="display:flex;align-items:center;gap:8px"><div class="bar"><i style="width:${conf * 100}%;background:${pass ? "var(--label)" : "rgba(235,235,245,0.4)"}"></i><span class="gate" style="left:${gate * 100}%"></span></div><span class="${pass ? "" : "muted"}">${Math.round(conf * 100)}%</span></div></td>
      <td class="num">${p.money_at_risk ? fmtKES(p.money_at_risk) : '<span class="muted">-</span>'}</td>
      <td class="num">${c ? `<b>${fmtKES(c.amount)}</b>` : p.payout_at_estimate ? `<span class="muted">${fmtKES(p.payout_at_estimate)}</span>` : '<span class="muted">-</span>'}</td>
      <td>${st ? `<span class="pill ${st}">${STATUS[st]}</span>` : ""}</td></tr>`;
  }).join("");
}
function renderClaims() {
  const sent = [...S.claims.values()].filter((c) => c.status === "sent").length, paidN = [...S.claims.values()].filter((c) => c.approved).length;
  const holder = (pid) => (S.world ? S.world.parcels.find((p) => p.id === pid)?.holder || "" : "");
  $("claimNote").textContent = S.claims.size ? `${paidN} paid, ${sent} delivered${S.review.size ? `, ${S.review.size} with the actuary` : ""}` : "";
  const box = $("claimScroll"), stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  $("claims").innerHTML = S.thread.length ? S.thread.map((m) => {
    const name = holder(m.pid);
    if (!m.sms) return `<div class="note-line ${m.tone}" data-id="${m.pid}"><b>${m.pid}</b> ${esc(name)}: ${esc(m.text)}</div>`;
    const c = S.claims.get(m.pid), first = esc(name.split(" ")[0]), amt = fmtKES(c.amount), sw = S.lang[m.pid] !== "en";
    const sig = c.verified === true ? '<span class="ok">✓ Signature verified in this browser</span>' : c.verified === false ? '<span class="bad">✕ Signature does not match</span>' : c.verified === "n/a" ? "<span>Signed ed25519</span>" : "<span>Checking signature…</span>";
    const del = c.status === "sent" ? '<span class="ok">Delivered</span>' : '<span class="wait">Waiting for signal</span>';
    const html = `<div class="claim${c.fresh ? " new" : ""}${m.pid === sel ? " sel" : ""}" data-id="${m.pid}" tabindex="0">
      <div class="to"><b>${esc(name)}</b> ${m.pid} · ${c.crop}<time>${clock(c.t)}</time></div>
      <div class="bubble">${sw ? SMS.sw(first, amt, m.pid) + `<small>${SMS.en(first, amt, m.pid).replace(/<\/?em>/g, "")}</small>` : SMS.en(first, amt, m.pid)}</div>
      <div class="proof">${sig}${c.bytes ? `<span>${c.bytes} bytes, fits one SMS</span>` : ""}<span>rule ${c.trigger_id}, ${Math.round(c.tier * 100)}% tier</span>${del}</div></div>`;
    c.fresh = false;
    return html;
  }).join("") : '<div class="empty"><b>No payments yet</b>When the actuary approves a claim, the farmer\'s text appears here, with its signature checked by this browser.</div>';
  if (stick) box.scrollTop = box.scrollHeight;
  renderSheet();
  $("hPaid").textContent = paidN;
  if (shownAmount !== S.paidTotal) countUp();
}
function renderReview() {
  const open = [...S.review.values()], live = !SITE && !S.replay, pct = (v) => Math.round(v * 100) + "%";
  $("tabs").querySelector('[data-t="actuary"]').textContent = open.length ? `Actuary ${open.length}` : "Actuary";
  if (!S.approval) {
    $("review").innerHTML = S.policy ? '<div class="empty"><b>No sign-off in this run</b>It was recorded before the actuary step, so every signed claim was paid.</div>'
      : '<div class="empty"><b>Actuary sign-off</b>Every signed claim waits here for a person. Flags show where to look first.</div>';
    return;
  }
  const clean = open.filter((c) => c.subject === "claim" && !c.flags.length).length;
  const card = (c) => {
    const f = c.facts, hold = c.subject === "hold", busy = S.busy.has(c.case_id), rej = S.rejecting === c.case_id;
    const flags = c.flags.length ? `<ul class="flags">${c.flag_text.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>` : '<p class="clean">No flags</p>';
    const act = !live ? "" : rej
      ? `<div class="why"><input id="rejNote" required maxlength="300" placeholder="Reason, required" aria-label="Reason for ${hold ? "keeping the hold" : "rejecting"}" value="${esc(S.draft)}"><button class="btn danger" data-do="confirm">${hold ? "Keep hold" : "Reject"}</button><button class="btn" data-do="cancel">Cancel</button></div>`
      : `<div class="act"><button class="btn primary" data-do="approve">${hold ? "Release and pay" : "Approve"}</button><button class="btn" data-do="reject">${hold ? "Keep on hold" : "Reject"}</button></div>`;
    return `<div class="case${busy ? " busy" : ""}" data-case="${esc(c.case_id)}">
      <div class="to"><b>${esc(f.holder)}</b> ${c.parcel} · ${f.crop}<time>${clock(c.t)}</time></div>
      <div class="ask">${hold ? `Adjuster holds under ${esc(c.rule_id)}. Release` : "Pay"} <b>${fmtKES(f.amount)}</b><span>${pct(f.share)} of ${fmtKES(f.sum_insured)} insured, rule ${f.trigger_id}, ${pct(f.tier)} tier</span></div>
      <div class="nums"><span>Loss <b>${pct(f.loss_mean)}</b> (${pct(f.loss_ci95[0])} to ${pct(f.loss_ci95[1])})</span><span>Confidence <b>${pct(f.confidence)}</b>, bar ${pct(f.gate)}</span>
        <span>Seen by drone <b>${pct(f.coverage)}</b></span><span>Farmer says <b>${esc(f.farmer_report || "nothing")}</b></span></div>
      ${flags}${act}</div>`;
  };
  const head = open.length ? `<div class="head"><span>${open.length} waiting. Money moves only when you approve.</span>${live && clean ? `<button class="btn" data-do="clean">Approve ${clean} without flags</button>` : ""}</div>` : "";
  const done = S.decided.slice(-8).reverse().map((x) => `<div class="done-line">${x.decision === "approve" ? "✓" : "✕"} <b>${x.parcel_id}</b> ${x.subject === "hold" ? (x.decision === "approve" ? "hold released" : "hold kept") : x.decision === "approve" ? "approved" : "rejected"} by ${esc(x.by)}${x.note ? `: ${esc(x.note)}` : ""}</div>`).join("");
  const empty = '<div class="empty"><b>Nothing waiting</b>Claims arrive here when their text reaches the insurer.</div>';
  $("review").innerHTML = head + (open.map(card).join("") || empty) + (done ? `<div class="sub-h" style="margin:8px 0 2px">Decided</div>${done}` : "");
  if (S.rejecting) $("rejNote")?.focus();
}
$("review").addEventListener("input", (e) => { if (e.target.id === "rejNote") S.draft = e.target.value; });
$("review").addEventListener("keydown", (e) => { if (e.target.id === "rejNote" && e.key === "Enter") e.target.closest(".case").querySelector('[data-do="confirm"]').click(); });
$("review").addEventListener("click", (e) => {
  const b = e.target.closest("[data-do]"); if (!b) return;
  const id = b.closest("[data-case]")?.dataset.case, act = b.dataset.do;
  if (act === "clean") { send({ cmd: "approve_clean" }); return; }
  if (act === "approve") { S.busy.add(id); send({ cmd: "decide", case_id: id, decision: "approve" }); }
  else if (act === "reject") { S.rejecting = id; S.draft = ""; }
  else if (act === "cancel") S.rejecting = null;
  else if (act === "confirm") {
    const note = S.draft.trim(); if (!note) { $("rejNote").focus(); return; }
    S.busy.add(id); send({ cmd: "decide", case_id: id, decision: "reject", note });
  }
  renderReview();
});
function footer() {
  const b = S.battery, b0 = S.battery0 || 1;
  $("fBatt").textContent = b == null ? "-" : b.toFixed(1) + " Wh";
  $("fBattBar").style.width = (b == null ? 100 : (100 * b) / b0) + "%";
  $("fBattBar").style.background = b != null && b / b0 < 0.25 ? "var(--red)" : b != null && b / b0 < 0.5 ? "var(--yellow)" : "var(--green)";
  $("fDist").textContent = (S.distance / 1000).toFixed(2) + " km";
  $("hDist").textContent = (S.distance / 1000).toFixed(1) + " km";
  $("hFirst").textContent = S.first == null ? "-" : Math.max(1, Math.round(S.first / 60)) + " min";
  $("live").className = S.done ? "done" : S.world ? "on" : "";
  $("fSortie").textContent = S.world && S.policy === "agent" ? `sortie cap ${(S.world.sortie_m / 1000).toFixed(1)} km` : "";
  $("fSensor").textContent = S.sensor ? (S.sensorDegraded ? "degraded" : S.sensor) : "-";
  $("fSensor").style.color = S.sensorDegraded ? "var(--yellow)" : "";
  $("fQual").textContent = S.quality != null ? `quality ${S.quality.toFixed(2)}` : "";
  const P = S.perception;
  $("fLat").textContent = P && P.n ? `${P.p50_ms.toFixed(2)} / ${P.p95_ms.toFixed(2)} ms` : S.sensor === "classical" ? "classical" : "-";
  $("fModel").textContent = P ? `p50 / p95, int8 ${P.model_mb.toFixed(2)} MB, ${P.n.toLocaleString()} tiles` : "";
  const u = S.uplink;
  $("fLink").textContent = u.state; $("fLink").className = "state-" + u.state;
  $("fQueue").textContent = `queue ${u.depth}, delivered ${u.settled}`;
  $("fBytes").textContent = (u.bytes || 0).toLocaleString() + " B";
  $("fPkt").textContent = u.pkt ? `${u.pkt} B per claim, SMS max 140` : "";
  $("fChain").textContent = S.events.toLocaleString() + " events";
  $("fHead").textContent = S.head ? "head " + S.head.slice(0, 12) : "-";
  $("fTime").textContent = clock(S.t);
  $("fPolicy").textContent = PILOT[S.policy] || "";
}
setInterval(() => { footer(); faulty(); }, 200);

// ---------- controls ----------
// On the static site there is no server: recorded runs replay here instead.
const SITE = document.documentElement.dataset.site === "static";
const api = (p, seed) => (SITE ? `data/${p}${seed ? "-" + seed : ""}.json` : `/api/${p}${seed ? "?seed=" + seed : ""}`);
const player = { evs: [], i: 0, t: 0, speed: 10, paused: false };
let ws;
function send(o) {
  if (!SITE) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); return; }
  if (o.cmd === "start") play(o.policy.replace("replay:", ""));
  if (o.cmd === "speed") player.speed = o.value;
  if (o.cmd === "pause") player.paused = o.on;
}
async function play(name) {
  status("Loading recorded run…");
  const text = await (await fetch(`data/runs/${name}`)).text();
  Object.assign(player, { evs: text.trim().split("\n").map((l) => JSON.parse(l)), i: 0, t: 0 });
  batch.push({ kind: "SESSION_RESET", seq: -1, t: 0, data: { replay: name.replace(".jsonl", "") } });
}
setInterval(() => {  // same pacing as the server's replay: sim time at speed, long gaps cut to 2 s
  const { evs, speed } = player;
  if (player.paused || player.i >= evs.length) return;
  player.t = Math.max(player.t + 0.05 * speed, evs[player.i].t - 2 * speed);
  while (player.i < evs.length && evs[player.i].t <= player.t) batch.push(evs[player.i++]);
}, 50);
function setLinkButtons(state) { for (const b of $("link").children) b.classList.toggle("active", b.dataset.v === state); }
function hello(d) {
  const pol = $("policy");
  pol.querySelector('[value="agents"]').textContent = `AI agents: field, farmer, adjuster${d.agent_available ? "" : " (needs a Gemini key)"}`;
  if (!d.running && !S.world) pol.value = d.agent_available ? "agents" : "infogain";
  $("speed").value = d.speed; $("speedv").textContent = d.speed + "×";
  const has = (sel, v) => [...$(sel).options].some((o) => o.value === v);
  if (has("scenario", d.scenario)) $("scenario").value = d.scenario;
  if (d.running && has("policy", d.policy)) pol.value = d.policy;
  if (d.running || S.world) $("overlay").style.display = "none";
}
// One button: Start, then Pause and Resume while a run is live. Changing the scenario or pilot arms a fresh Start.
let paused = false, armed = false;
const live = () => !!S.world && !S.done && !armed;
function togglePause() { if (!live()) return; paused = !paused; send({ cmd: "pause", on: paused }); }
$("start").onclick = () => {
  if (live()) return togglePause();
  paused = armed = false;
  send({ cmd: "start", scenario: $("scenario").value, policy: $("policy").value });
};
for (const id of ["scenario", "policy"]) $(id).addEventListener("change", () => (armed = !!S.world && !S.done));
function startLabel() { const b = $("start"); b.textContent = !live() ? "Start" : paused ? "Resume" : "Pause"; b.classList.toggle("primary", !live() || paused); }
$("speed").oninput = (e) => { $("speedv").textContent = e.target.value + "×"; send({ cmd: "speed", value: +e.target.value }); };
for (const b of $("link").children) b.onclick = () => send({ cmd: "link", state: b.dataset.v });
$("degrade").onclick = () => send({ cmd: "degrade", on: !S.sensorDegraded });
$("more").addEventListener("beforetoggle", (e) => {
  if (e.newState !== "open") return;
  const r = $("moreBtn").getBoundingClientRect();
  Object.assign($("more").style, { top: r.bottom + 8 + "px", left: Math.max(12, r.right - 280) + "px" });
});
const faulty = () => { $("moreBtn").classList.toggle("dot", S.sensorDegraded || S.uplink.state !== "online"); startLabel(); };
const store = (k, v) => { try { return v === undefined ? localStorage.getItem(k) : localStorage.setItem(k, v); } catch { return null; } };
let diagOn = store("kipimo.diag") === "1";
function diag(on) {
  diagOn = on; store("kipimo.diag", on ? "1" : "0");
  $("diag").hidden = !on || view !== "mission"; $("diagBtn").setAttribute("aria-pressed", on); $("diagBtn").textContent = on ? "Hide diagnostics" : "Show diagnostics";
  resize();
}
$("diagBtn").onclick = () => diag(!diagOn);
for (const c of $("layers").children) c.onclick = () => { layer = c.dataset.l; for (const x of $("layers").children) x.classList.toggle("active", x === c); legend(); dirty = true; };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "1" || e.key === "2" || e.key === "3") $("layers").children[+e.key - 1].click();
  if (e.key === " " && !e.target.closest("button, [data-id]")) { e.preventDefault(); togglePause(); }
  if (e.key === "Escape" && sel && !document.querySelector(":popover-open")) select(null); // one thing per press
});

let batch = [];
setInterval(() => {
  if (!batch.length) return;
  const b = batch; batch = [];
  for (const ev of b) try { onEvent(ev); } catch (e) { console.error(ev.kind, e); } // one bad event never drops the rest
}, 50);
function connect() {
  const tok = new URLSearchParams(location.search).get("token"); // set when KIPIMO_TOKEN guards the server
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws${tok ? "?token=" + encodeURIComponent(tok) : ""}`);
  ws.onopen = () => { batch = []; fresh(); $("log").innerHTML = ""; $("cond").innerHTML = ""; renderClaims(); $("parcels").innerHTML = ""; status(""); dirty = true; };
  ws.onmessage = (m) => { try { batch.push(JSON.parse(m.data)); } catch (e) { console.error(e); } };
  ws.onclose = () => setTimeout(connect, 1000);
}
if (SITE) {
  for (const el of [$("scenario"), $("faultCtl")]) el.hidden = true;
  $("policy").innerHTML = "";
  $("overlay").textContent = "Loading a recorded run…";
} else fetch("/api/scenarios").then((r) => r.json()).then((list) => {
  $("scenario").innerHTML = list.map((s) => `<option value="${s}">${words(s)}</option>`).join("");
}).finally(connect);
function loadRecordings() {
  return fetch(api("recordings")).then((r) => r.json()).then((list) => {
    const keep = $("policy").value;
    $("policy").querySelector("optgroup")?.remove();
    if (!list.length) return;
    const g = document.createElement("optgroup"); g.label = "Replay";
    g.innerHTML = list.map((r) => `<option value="replay:${esc(r.name)}">${esc(r.label)}</option>`).join("");
    $("policy").appendChild(g);
    if ([...$("policy").options].some((o) => o.value === keep)) $("policy").value = keep;
  });
}
loadRecordings().then(() => SITE && $("start").click());
legend(); renderClaims(); loop();

// segmented controls: aria-pressed mirrors the active class, wherever it is set
const pressed = (b) => b.setAttribute("aria-pressed", b.classList.contains("active"));
document.querySelectorAll(".seg button").forEach(pressed);
new MutationObserver((ms) => ms.forEach((m) => m.target.matches(".seg button") && pressed(m.target))).observe(document.body, { subtree: true, attributeFilter: ["class"] });
const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------- views ----------
const cache = {};
const getJSON = (url) => (cache[url] ??= fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${url}: ${r.status}`)))).catch((e) => { delete cache[url]; throw e; }));
let view = "mission", plan = "greedy";
function show(v) {
  view = v;
  for (const b of $("view").children) b.classList.toggle("active", b.dataset.v === v);
  for (const id of ["mission", "district", "evals"]) $("v-" + id).hidden = id !== v;
  $("mctl").hidden = v !== "mission"; $("back").hidden = v === "mission";
  $("diag").hidden = v !== "mission" || !diagOn;
  if (v === "district") district().catch((e) => ($("dnote").textContent = `could not load: ${e.message}`));
  if (v === "evals") evals().catch((e) => ($("v-evals").innerHTML = `<div class="banner">Could not load evals: ${esc(e.message)}</div>`));
  if (v === "mission") resize();
}
for (const b of $("view").children) b.onclick = () => { show(b.dataset.v); $("more").hidePopover(); };
$("back").onclick = () => show("mission");
// side panel: one view at a time; a new payment while another tab is open shows a count on Phones
let tab = "phones", unseen = 0;
function showTab(t) {
  tab = t;
  for (const b of $("tabs").children) b.classList.toggle("active", b.dataset.t === t);
  for (const el of $("right").querySelectorAll(".scroll[data-t]")) el.hidden = el.dataset.t !== t;
  if (t === "phones") unseen = 0;
  badge();
  const box = $("right").querySelector(`.scroll[data-t="${t}"]`);
  if (t !== "fields") box.scrollTop = box.scrollHeight;
}
function badge() { $("tabs").children[0].textContent = unseen ? `Phones ${unseen}` : "Phones"; }
for (const b of $("tabs").children) b.onclick = () => showTab(b.dataset.t);
for (const b of $("dplan").children) b.onclick = () => { plan = b.dataset.p; for (const x of $("dplan").children) x.classList.toggle("active", x === b); district(); };
$("dseed").innerHTML = Array.from({ length: 12 }, (_, i) => `<option value="${i + 11}">district ${i + 11}</option>`).join("");
$("dseed").onchange = () => district();

async function district() {
  $("dnote").textContent = "flying both plans…";
  const D = await getJSON(api("district", $("dseed").value)), P = D.plans[plan];
  const flooded = D.villages.filter((v) => v.flooded && v.expected > 0);
  $("dnote").textContent = `${flooded.length} of ${D.villages.length} villages flooded, ${D.hubs.length} drone hubs, ${D.reach_km} km reach`;
  const g = D.plans.greedy, n = D.plans.nearest, less = 1 - g.kes_hours_waiting / n.kes_hours_waiting, top = Math.max(g.kes_hours_waiting, n.kes_hours_waiting);
  $("dstats").innerHTML = [
    ["Less money waiting", `${Math.round(less * 100)}%`, "most money first, against nearest first"],
    ["Last village paid", `${P.makespan_h} h`, "after the flood"],
    ["Owed but out of reach", fmtKES(P.owed_unreached), "beyond every hub's battery"],
    ["Hubs cover", `${Math.round(D.covered * 100)}%`, "of insured value"],
  ].map(([k, v, sub]) => `<div><small>${k}</small><b>${v}</b><span>${sub}</span></div>`).join("");
  $("dcompare").innerHTML = `<div class="cmp">${[["Most money first", g], ["Nearest first", n]].map(([k, x]) =>
    `<span>${k}</span><i class="${x === g ? "best" : ""}" style="width:${(100 * x.kes_hours_waiting) / top}%"></i><b>${Math.round(x.kes_hours_waiting).toLocaleString()}</b>`).join("")}</div>`;
  $("dqueues").innerHTML = Object.entries(P.queues).map(([h, q]) => `<div class="queue"><b>${h}</b>${q.length ? q.map((v) => `<span>${v}<small>${P.settled_h[v]} h</small></span>`).join("→") : "idle"}</div>`).join("");
  drawDistrict(D, P);
}
function drawDistrict(D, P) {
  const c = $("dmap"), wrap = $("dmapWrap"), r = window.devicePixelRatio || 1;
  const px = Math.max(280, Math.min(wrap.clientWidth - 24, wrap.clientHeight - 16));
  c.width = c.height = px * r; c.style.width = c.style.height = px + "px";
  const pad = 24 * r, g = c.getContext("2d"), k = (px * r - 2 * pad) / D.span_km, at = (x, y) => [pad + x * k, pad + y * k];
  g.fillStyle = "#111113"; g.fillRect(0, 0, c.width, c.height);
  const hubs = Object.fromEntries(D.hubs.map((h) => [h.id, h]));
  g.setLineDash([5 * r, 5 * r]); g.strokeStyle = `rgba(${BLUE},0.55)`; g.lineWidth = r;
  for (const h of D.hubs) { g.beginPath(); g.arc(...at(h.x, h.y), D.reach_km * k, 0, 2 * Math.PI); g.stroke(); }
  g.setLineDash([]);
  const vs = Object.fromEntries(D.villages.map((v) => [v.id, v]));
  for (const [h, q] of Object.entries(P.queues)) q.forEach((vid, i) => {
    const a = at(hubs[h].x, hubs[h].y), b = at(vs[vid].x, vs[vid].y);
    g.strokeStyle = "rgba(245,245,247,0.55)"; g.lineWidth = 1.2 * r; g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
    const [mx, my] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    g.fillStyle = "#1c1c1e"; g.beginPath(); g.arc(mx, my, 8 * r, 0, 2 * Math.PI); g.fill();
    g.fillStyle = INK; g.font = `600 ${10 * r}px -apple-system, sans-serif`; g.textAlign = "center"; g.fillText(i + 1, mx, my + 3.5 * r);
  });
  for (const v of D.villages) {
    const [x, y] = at(v.x, v.y), live = v.flooded && v.expected > 0, rad = (4 + Math.sqrt(v.expected) / 22) * r;
    g.beginPath(); g.arc(x, y, rad, 0, 2 * Math.PI);
    g.fillStyle = live ? "rgba(255,159,10,0.85)" : "rgba(235,235,245,0.18)"; g.fill();
    if (live && !(v.id in P.settled_h)) { g.strokeStyle = `rgb(${RED})`; g.lineWidth = 2 * r; g.stroke(); }
    if (live) {
      g.fillStyle = INK; g.font = `500 ${10 * r}px -apple-system, sans-serif`; g.textAlign = "center";
      g.fillText(v.id, x, y + rad + 12 * r);
    }
  }
  for (const h of D.hubs) {
    const [x, y] = at(h.x, h.y);
    g.fillStyle = INK; g.beginPath(); g.roundRect(x - 13 * r, y - 10 * r, 26 * r, 20 * r, 5 * r); g.fill();
    g.fillStyle = "#000"; g.font = `700 ${10 * r}px -apple-system, sans-serif`; g.textAlign = "center"; g.fillText(h.id, x, y + 3.5 * r);
  }
}

const SECTIONS = [
  ["Outcomes", [["outcome.payout_mae_pct", "Payout error, % of sum insured", 2], ["outcome.owed_unpaid", "Owed farmers left unpaid", 2], ["outcome.paid_not_owed", "Paid but not owed", 2], ["outcome.distance_km", "Distance, km", 1], ["outcome.first_claim_min", "First claim, sim minutes", 0]]],
  ["Farmer agent", [["farmer.severity_accuracy", "Damage read correctly", "%"], ["farmer.where_accuracy", "Part of field read correctly", "%"], ["farmer.when_accuracy", "Timing read correctly", "%"], ["farmer.attack_recall", "Attacks flagged", "%"], ["farmer.warning_recall", "Warnings recognised", "%"], ["farmer.discounted", "Reports discounted by memory", 2]]],
  ["Adjuster agent", [["adjuster.audit_misses", "Audit misses", 2], ["adjuster.challenges", "Challenges per flood", 2], ["adjuster.ungrounded_attempts", "Ungrounded challenge attempts", 2], ["adjuster.escalations", "Handed to a person", 2]]],
  ["Field agent", [["field.refusal_recovery", "Refused claims recovered", "%"], ["field.flight_changes", "Flight changes per flood", 2], ["field.minutes_waited", "Minutes waited for weather", 0], ["field.degraded_quality", "Camera quality under faults", 2]]],
];
const val = (m, f) => (m == null || m[0] == null ? "n/a" : f === "%" ? Math.round(m[0] * 100) + "%" : m[0].toFixed(f));
const ciTxt = (m, f) => (m == null || m[0] == null || m[1] === m[2] ? "" : `<small>${val([m[1]], f)} to ${val([m[2]], f)}</small>`);
async function evals() {
  const E = await getJSON(api("evals"));
  const base = E.suites.find((x) => x.name === "rules-default-test"), mem = E.suites.find((x) => x.name.endsWith("-memory"));
  const live = E.suites.filter((x) => x.name.startsWith("live"));
  const safety = Object.entries(base.summary).filter(([k]) => k.startsWith("safety.") && k !== "safety.pass_k");
  const card = (title, body) => `<section class="panel"><h2>${title}</h2>${body}</section>`;
  const rows = (sm, list) => `<div class="kv">${list.map(([k, l, f]) => `<span>${l}</span><b>${val(sm[k], f)}${ciTxt(sm[k], f)}</b>`).join("")}</div>`;
  const dist = E.districts[0]?.summary.greedy;
  const hero = (k, v, sub) => `<section class="panel hero"><small>${k}</small><b>${v}</b><span>${sub}</span></section>`, mae = base.summary["outcome.payout_mae_pct"];
  $("v-evals").innerHTML = `<div class="banner">${live.length ? `Live model runs: ${live.map((x) => x.name).join(", ")}.` : "<b>No live model scored yet.</b> These are the rule-based fallbacks, the bar a live model has to beat."} ${base.cases} held-out cases, means with 95% bootstrap intervals.</div>`
    + hero("Safety", `${Math.round(base.summary["safety.pass_k"][0] * 100)}%`, "of cases with no rule broken in any trial")
    + hero("Payout error", `${val(mae, 2)}%`, `of sum insured, ${val([mae[1]], 2)} to ${val([mae[2]], 2)}%`)
    + (dist ? hero("District dispatch", `${Math.round(dist.less_than_nearest[0] * 100)}%`, `less money waiting than nearest first, ${Math.round(dist.less_than_nearest[1] * 100)} to ${Math.round(dist.less_than_nearest[2] * 100)}% across ${dist.districts} districts`) : "")
    + SECTIONS.map(([t, list]) => card(t, rows(base.summary, list))).join("")
    + (mem ? card("Memory across missions", rows(mem.summary, [["farmer.discounted", "Reports discounted per flood", 2], ["outcome.distance_km", "Distance with memory, km", 1], ["outcome.payout_mae_pct", "Payout error with memory, %", 2]])) : "")
    + card("Rules never broken", `<div class="kv">${safety.map(([k, m]) => `<span>${words(k.slice(7)).replace("nofly", "no-fly")}</span><b>${val(m, 0)}</b>`).join("")}</div>`);
}
window.addEventListener("resize", () => { if (view === "district") district(); });
diag(diagOn);
