// 3D: the live map draped on real terrain, water shaded by depth, the drone flown with its physics.
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const box = $("scene3d"), tags = $("labels3d"), TEX = 2048, BASE = -12, DRONE_X = 40, MAX_TILT = 0.44;
const UP = new THREE.Vector3(0, 1, 0), clock = new THREE.Clock();
let EX = 3, mode = "orbit", touched = false, fly = null;

let renderer;
try { renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" }); }
catch { $("viewMode").style.display = "none"; throw new Error("no WebGL, top view only"); }
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
box.prepend(renderer.domElement);
renderer.domElement.setAttribute("role", "img");
renderer.domElement.setAttribute("aria-label", "3D view of the fields on real terrain, with the drone. Click a field for details.");

const scene = new THREE.Scene();
scene.background = gradient(["#050608", "#12151b", "#1e232c"]);
scene.fog = new THREE.Fog(0x161a20, 2400, 4600);
const camera = new THREE.PerspectiveCamera(38, 1, 2, 12000);
camera.position.set(820, 760, 1050);
const controls = new OrbitControls(camera, renderer.domElement);
Object.assign(controls, { enableDamping: true, dampingFactor: 0.08, maxPolarAngle: 1.45, minDistance: 80, maxDistance: 3400, autoRotateSpeed: 0.4 });
controls.addEventListener("start", () => { touched = true; fly = null; $("hint").hidden = true; });

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a241c, 1.4));
const sun = new THREE.DirectionalLight(0xfff1dc, 2.4);
sun.position.set(-700, 1100, -500); sun.castShadow = true;
Object.assign(sun.shadow.camera, { left: -820, right: 820, top: 820, bottom: -820, near: 10, far: 3600 });
sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0005; sun.shadow.normalBias = 1.5;
scene.add(sun);

function gradient(stops) {
  const c = Object.assign(document.createElement("canvas"), { width: 2, height: 256 }), g = c.getContext("2d"), gr = g.createLinearGradient(0, 0, 0, 256);
  stops.forEach((s, i) => gr.addColorStop(i / (stops.length - 1), s));
  g.fillStyle = gr; g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const fade = (m) => Object.assign(m, { transparent: true, depthWrite: false });
// layer 1 holds helpers the main view draws and the drone's own camera does not
const helper = (...objs) => objs.forEach((o) => o.traverse((c) => c.layers.set(1)));
camera.layers.enable(1); sun.shadow.camera.layers.enable(1);

// ---------- ground: the dashboard's own layers, redrawn only when they change ----------
const groundCv = Object.assign(document.createElement("canvas"), { width: TEX, height: TEX });
const groundTex = new THREE.CanvasTexture(groundCv);
groundTex.colorSpace = THREE.SRGBColorSpace; groundTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
let lines = null, seen = [];
function paintGround() {
  const L = S.layers, base = layer === "truth" ? L.truth : layer === "voi" ? L.voi : L.belief, key = [base, L.relief, S.pmap];
  if (key.every((k, i) => k === seen[i])) return;
  seen = key;
  const g = groundCv.getContext("2d");
  g.fillStyle = "#2a302c"; g.fillRect(0, 0, TEX, TEX); g.imageSmoothingQuality = "high";
  if (base) g.drawImage(base, 0, 0, TEX, TEX);
  if (L.relief) { g.globalCompositeOperation = "soft-light"; g.drawImage(L.relief, 0, 0, TEX, TEX); g.globalCompositeOperation = "source-over"; }
  if (edges) g.drawImage(lines ||= parcelLines(), 0, 0);
  groundTex.needsUpdate = true;
}
const canvas = (w) => Object.assign(document.createElement("canvas"), { width: w, height: w });
let edges = null;
function parcelLines() {
  const R = 1024, c = canvas(R), g = c.getContext("2d"), img = g.createImageData(R, R);
  for (let j = 0; j < R * R; j++) { img.data.fill(255, j * 4, j * 4 + 3); img.data[j * 4 + 3] = edges[j * 4]; }
  g.putImageData(img, 0, 0);
  const out = canvas(TEX), og = out.getContext("2d");
  og.globalAlpha = 0.38; og.drawImage(c, 0, 0, TEX, TEX); return out;
}

// ---------- terrain, soil skirt and water ----------
const land = new THREE.Group(), statics = new THREE.Group();
scene.add(land, statics);
let span = 1024, n = 0, hgt = null, stage = 0, waterMat = null;
const P = (x, y, h) => new THREE.Vector3(x - span / 2, h, y - span / 2);
function h0(x, y) {
  if (!hgt) return 0;
  const u = Math.min(n - 1.001, Math.max(0, (x / span) * (n - 1))), v = Math.min(n - 1.001, Math.max(0, (y / span) * (n - 1)));
  const i = v | 0, j = u | 0, fu = u - j, fv = v - i, r = i * n + j;
  return (hgt[r] * (1 - fu) + hgt[r + 1] * fu) * (1 - fv) + (hgt[r + n] * (1 - fu) + hgt[r + n + 1] * fu) * fv;
}
const dispose = (g) => { g.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); }); g.clear(); };

function world(d) {
  if (!d.height) return;
  span = d.span_m; n = d.relief_grid; stage = d.stage;
  bareEarth(Float32Array.from(b64u8(d.height), (v) => (v / 255) * d.height_max));
  dispose(land); lines = null; seen = []; trail.n = -1; paid.clear();
  farmTextures(d);
  const g = new THREE.PlaneGeometry(span, span, n - 1, n - 1).rotateX(-Math.PI / 2), p = g.attributes.position;
  for (let k = 0; k < p.count; k++) p.setY(k, hgt[k]);
  g.computeVertexNormals();
  const ground = new THREE.Mesh(g, farmMat);
  ground.receiveShadow = true;
  land.add(ground, skirt(), water(g));
  land.scale.y = EX;
  tags.innerHTML = "";
  parcelTags = d.parcels.map((q) => {
    const el = tag(`<i></i><b>${q.id}</b><span></span>`);
    el.onclick = () => window.select(q.id);
    return { q, el };
  });
  $("s3site").textContent = d.site && d.site !== "synthetic" ? d.site : "Procedural floodplain";
  build();
  if (!touched) flyTo(new THREE.Vector3(span * 0.55, span * 1.05, span * 0.95), new THREE.Vector3(0, 0, 0));
}

// ---------- the farm: crops, bunds, silt and wet ground, drawn per pixel on the GPU ----------
let canopy = null;
function bareEarth(dsm) {
  // GLO-30 is a surface model: cells standing over their neighbourhood are tree crowns, not ground
  const mean = new Float32Array(n * n), r = 2;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0, c = 0;
    for (let a = Math.max(0, i - r); a <= Math.min(n - 1, i + r); a++) for (let b = Math.max(0, j - r); b <= Math.min(n - 1, j + r); b++) { s += dsm[a * n + b]; c++; }
    mean[i * n + j] = s / c;
  }
  canopy = dsm.map((h, k) => Math.max(0, h - mean[k]));
  hgt = dsm.map((h, k) => (canopy[k] > 1 ? mean[k] : h));
}
const CROPS = { maize: 0, rice: 1, sorghum: 2 };
const U = { tSat: { value: null }, uSatRect: { value: new THREE.Vector4(0, 0, 1, 1) }, uSat: { value: 1 }, tEdge: { value: null }, tParcel: { value: null }, tInfo: { value: null }, tDamage: { value: null }, tOverlay: { value: groundTex }, uSpan: { value: 1024 }, uStage: { value: 0 }, uOverlay: { value: 0 }, uSel: { value: -1 } };
function farmTextures(d) {
  const tex = (data, w, h, fmt, filter) => Object.assign(new THREE.DataTexture(data, w, h, fmt), { magFilter: filter, minFilter: filter, needsUpdate: true });
  const info = new Uint8Array(64 * 4);
  d.parcels.forEach((q, i) => {
    const r = rng(i * 7919 + q.cx);
    info.set([CROPS[q.crop] ?? 0, r() * 255, 255 * Math.min(1, Math.max(0.15, (q.age_days - 8) / 40)), r() * 255], i * 4);
  });
  for (const t of [U.tEdge.value, U.tParcel.value, U.tInfo.value, U.tDamage.value]) t?.dispose();
  edges = fieldEdges();
  U.tEdge.value = tex(edges, 1024, 1024, THREE.RGBAFormat, THREE.LinearFilter);
  U.tParcel.value = tex(S.pmap, d.map_px, d.map_px, THREE.RedFormat, THREE.NearestFilter);
  U.tInfo.value = tex(info, 64, 1, THREE.RGBAFormat, THREE.NearestFilter);
  U.tDamage.value = tex(b64u8(d.truth), d.grid, d.grid, THREE.RedFormat, THREE.LinearFilter);
  U.uSpan.value = span; U.uStage.value = stage;
  const img = d.imagery;
  $("groundMode").children[0].style.display = img ? "" : "none";
  if (!img) { $("groundMode").children[1].click(); return; }
  U.uSatRect.value.set(...img.rect);
  if (U.tSat.value?.userData.url !== img.url) {
    U.tSat.value?.dispose();
    U.tSat.value = new THREE.TextureLoader().load(img.url);
    Object.assign(U.tSat.value, { colorSpace: THREE.SRGBColorSpace, anisotropy: 8, userData: { url: img.url } });
  }
}
function rng(seed) {
  let a = seed >>> 0 || 1;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const NOISE = `
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + 1.0), u.x), u.y); }
  float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; } return s; }
  uniform sampler2D tParcel, tInfo; uniform float uSpan;
  float parcelAt(vec2 m) { return floor(texture2D(tParcel, m / uSpan).r * 255.0 + 0.5); }
  vec4 infoOf(float pid) { return texture2D(tInfo, vec2((pid + 0.5) / 64.0, 0.5)); }
  float cropHeight(float pid) { if (pid > 254.5) return 0.4; vec4 i = infoOf(pid); float c = floor(i.r * 255.0 + 0.5); return (c == 1.0 ? 0.7 : c == 2.0 ? 1.7 : 2.1) * i.b; }`;
const farmMat = new THREE.MeshStandardMaterial({ roughness: 0.93 });
farmMat.onBeforeCompile = (sh) => {
  Object.assign(sh.uniforms, U);
  sh.vertexShader = sh.vertexShader
    .replace("#include <common>", "#include <common>\nuniform float uSpan; varying vec2 vMap; varying float vH;")
    .replace("#include <begin_vertex>", "#include <begin_vertex>\nvMap = position.xz + uSpan * 0.5; vH = position.y;");
  sh.fragmentShader = sh.fragmentShader.replace("#include <common>", `#include <common>
    varying vec2 vMap; varying float vH; uniform sampler2D tDamage, tOverlay, tEdge, tSat; uniform float uStage, uOverlay, uSat, uSel; uniform vec4 uSatRect;${NOISE}`)
    .replace("#include <map_fragment>", `
    vec3 edge = texture2D(tEdge, vMap / uSpan).rgb;
    float pid = parcelAt(vMap), bund = smoothstep(0.35, 0.8, edge.r), head = smoothstep(0.55, 0.95, edge.g), inside = smoothstep(0.35, 0.65, edge.b);
    float n1 = fbm(vMap * 0.035), n2 = fbm(vMap * 0.3), n3 = vnoise(vMap * 2.3);
    vec3 soil = mix(vec3(0.43, 0.33, 0.24), vec3(0.56, 0.45, 0.33), n1) * (0.88 + 0.24 * n3);
    vec3 col = mix(vec3(0.33, 0.39, 0.18), vec3(0.6, 0.55, 0.36), smoothstep(0.4, 0.72, fbm(vMap * 0.018 + 7.0))) * (0.8 + 0.4 * n2);
    vec3 wild = col;
    if (pid < 254.5 || inside > 0.0) {
      vec4 info = infoOf(pid < 254.5 ? pid : parcelAt(vMap + vec2(4.0, 0.0)));
      float crop = floor(info.r * 255.0 + 0.5), ang = info.g * 3.14159, cover = info.b;
      vec2 dir = vec2(cos(ang), sin(ang));
      float sp = crop == 1.0 ? 0.3 : 0.75, across = dot(vMap, vec2(-dir.y, dir.x)) / sp, along = dot(vMap, dir);
      float plants = smoothstep(1.0 - cover, 1.25 - cover * 0.5, 1.0 - abs(fract(across) - 0.5) * 2.0);
      plants *= smoothstep(0.08, 0.3, vnoise(vec2(along * 1.7, floor(across) * 3.1)));
      plants = mix(plants, cover * 0.85, clamp(fwidth(across) * 1.5, 0.0, 1.0));
      vec3 leaf = crop == 1.0 ? vec3(0.4, 0.62, 0.2) : crop == 2.0 ? mix(vec3(0.38, 0.45, 0.18), vec3(0.58, 0.32, 0.16), smoothstep(0.55, 0.9, n3) * cover) : vec3(0.27, 0.45, 0.14);
      leaf *= (0.78 + 0.4 * n2) * (0.9 + 0.2 * info.a);
      vec3 bed = crop == 1.0 ? vec3(0.28, 0.27, 0.21) : soil;
      col = mix(bed, leaf, plants);
      col = mix(col, soil * 0.96, head * 0.75);
      col = mix(wild, col, inside);
      col = mix(col, vec3(0.6, 0.5, 0.37) * (0.9 + 0.2 * n3), bund);
    }
    if (uSat > 0.0) {
      // real imagery carries the colour, the procedural farm only adds detail finer than its 10 m pixels
      vec2 su = mix(uSatRect.xy, uSatRect.zw, vMap / uSpan);
      vec3 sat = pow(texture2D(tSat, vec2(su.x, 1.0 - su.y)).rgb, vec3(1.0 / 2.2));
      float detail = clamp(dot(col, vec3(0.3, 0.59, 0.11)) / 0.36, 0.6, 1.5);
      col = mix(col, sat * mix(1.0, detail, 0.5) * 1.08, uSat);
    }
    float dmg = smoothstep(0.12, 0.7, texture2D(tDamage, vMap / uSpan).r) * (0.7 + 0.3 * n2);
    vec3 silt = mix(vec3(0.6, 0.54, 0.43), vec3(0.47, 0.41, 0.31), n2);
    col = mix(col, mix(col * vec3(0.92, 0.78, 0.55), silt, 0.55 + 0.3 * n3), dmg);
    col *= 1.0 - 0.38 * (1.0 - smoothstep(0.0, 0.9, vH - uStage));
    col = pow(col, vec3(2.2));
    vec2 st = vMap / uSpan;
    col = mix(col, texture2D(tOverlay, vec2(st.x, 1.0 - st.y)).rgb, uOverlay);
    if (abs(pid - uSel) < 0.5) col = mix(col * 1.25, vec3(0.04, 0.52, 1.0), 0.12 + 0.5 * bund);
    diffuseColor.rgb = col;`);
};

function skirt() {
  const ring = [], v = [], c = [], top = [0.36, 0.29, 0.22], bot = [0.09, 0.08, 0.07];
  for (let j = 0; j < n; j++) ring.push(j);
  for (let i = 1; i < n; i++) ring.push(i * n + n - 1);
  for (let j = n - 2; j >= 0; j--) ring.push((n - 1) * n + j);
  for (let i = n - 2; i >= 0; i--) ring.push(i * n);
  const xz = (k) => [((k % n) / (n - 1) - 0.5) * span, (Math.floor(k / n) / (n - 1) - 0.5) * span];
  for (let s = 0; s < ring.length - 1; s++) {
    const [ax, az] = xz(ring[s]), [bx, bz] = xz(ring[s + 1]), ha = hgt[ring[s]], hb = hgt[ring[s + 1]];
    v.push(ax, ha, az, bx, BASE, bz, bx, hb, bz, ax, ha, az, ax, BASE, az, bx, BASE, bz);
    c.push(...top, ...bot, ...top, ...top, ...bot, ...bot);
  }
  v.push(-span / 2, BASE, -span / 2, span / 2, BASE, span / 2, span / 2, BASE, -span / 2, -span / 2, BASE, -span / 2, -span / 2, BASE, span / 2, span / 2, BASE, span / 2);
  c.push(...Array(6).fill(bot).flat());
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(v, 3)); g.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
  g.computeVertexNormals();
  return new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: THREE.DoubleSide }));
}

function water(tg) {
  const g = tg.clone(), p = g.attributes.position, depth = new Float32Array(p.count);
  for (let k = 0; k < p.count; k++) { depth[k] = stage - p.getY(k); p.setY(k, stage); }
  g.setAttribute("depth", new THREE.BufferAttribute(depth, 1));
  waterMat = fade(new THREE.ShaderMaterial({
    uniforms: { time: { value: 0 }, tParcel: U.tParcel, tInfo: U.tInfo, uSpan: U.uSpan },
    vertexShader: `attribute float depth; varying float vDepth; varying vec3 vWorld;
      void main() { vDepth = depth; vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform float time; varying float vDepth; varying vec3 vWorld;${NOISE}
      void main() {
        if (vDepth <= 0.0) discard;
        vec2 m = vWorld.xz + uSpan * 0.5;
        float d = smoothstep(0.0, 1.8, vDepth), shore = 1.0 - smoothstep(0.0, 0.06, vDepth);
        vec3 col = mix(vec3(0.6, 0.52, 0.37), vec3(0.38, 0.31, 0.2), d) * (0.92 + 0.16 * fbm(m * 0.05 + time * 0.03));
        float r = sin(m.x * 0.08 + time * 1.2) * sin(m.y * 0.065 - time) * 0.5 + vnoise(m * 0.9 + time * 0.4) - 0.5;
        float fres = pow(1.0 - clamp(normalize(cameraPosition - vWorld).y, 0.0, 1.0), 3.0);
        col += 0.03 * r + fres * vec3(0.3, 0.36, 0.44);
        col = mix(col, vec3(0.7, 0.62, 0.48), shore * 0.5);
        float emerge = clamp((cropHeight(parcelAt(m)) - vDepth) / 0.6, 0.0, 1.0) * (0.6 + 0.4 * vnoise(m * 3.0));
        gl_FragColor = vec4(pow(col, vec3(2.2)), clamp(mix(0.62, 0.9, d) + shore * 0.2 + fres * 0.1, 0.0, 0.95) * (1.0 - 0.7 * emerge));
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  }));
  return new THREE.Mesh(g, waterMat);
}

// ---------- hub, no fly airspace, planned lanes ----------
function stripes() {
  const c = Object.assign(document.createElement("canvas"), { width: 64, height: 64 }), g = c.getContext("2d");
  g.strokeStyle = "rgba(255,69,58,0.4)"; g.lineWidth = 6;
  for (const o of [-64, 0, 64]) { g.beginPath(); g.moveTo(o, 64); g.lineTo(o + 64, 0); g.stroke(); }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; return t;
}
const stripeTex = stripes();
let parcelTags = [], zoneTags = [], hubTag = null;
function build() {
  dispose(statics); zoneTags.forEach((z) => z.el.remove()); hubTag?.el.remove(); lanes.key = null;
  const W = S.world, [hx, hy] = W.home;
  statics.add(...scatter(W));
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(14, 16, 2, 40), new THREE.MeshStandardMaterial({ color: 0xf5f5f7, roughness: 0.6 }));
  pad.position.copy(P(hx + 10, hy + 10, h0(hx, hy) * EX + 1)); pad.receiveShadow = true; statics.add(pad);
  hubTag = { at: pad.position.clone().setY(pad.position.y + 6), el: tag("<b>Hub</b>", "hub") };
  zoneTags = W.no_fly.map((z) => {
    const hTop = 70 * EX, g0 = h0(z.x, z.y) * EX, col = new THREE.CylinderGeometry(z.r, z.r, hTop, 64, 1, true);
    const tex = stripeTex.clone(); tex.repeat.set((2 * Math.PI * z.r) / 16, hTop / 16); tex.needsUpdate = true;
    for (const mat of [new THREE.MeshBasicMaterial({ color: 0xff453a, opacity: 0.16, side: THREE.DoubleSide }), new THREE.MeshBasicMaterial({ map: tex, opacity: 0.9 })]) {
      const wall = new THREE.Mesh(col, fade(mat));
      wall.position.copy(P(z.x, z.y, g0 + hTop / 2)); statics.add(wall); helper(wall);
    }
    for (const y of [g0 + 0.5, g0 + hTop]) {
      const rim = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 65 }, (_, i) => P(z.x + z.r * Math.cos((i / 64) * 2 * Math.PI), z.y + z.r * Math.sin((i / 64) * 2 * Math.PI), y))), new THREE.LineBasicMaterial({ color: 0xff453a }));
      statics.add(rim); helper(rim);
    }
    return { at: P(z.x, z.y, g0 + hTop + 6), el: tag(`<b>No fly</b><span>${words(z.name || "")}</span>`, "zone") };
  });
}
function scatter(W) {
  // trees where the surface model stands over the ground, at their measured height; homesteads on dry land
  const r = rng(Math.round(W.parcels.reduce((a, q) => a + q.cx * 7 + q.cy, 0))), c = span / n, pm = S.pmap, k = W.map_px / n;
  const free = (i, j) => pm[Math.floor(i * k) * W.map_px + Math.floor(j * k)] === 255;
  const trees = [], homes = [];
  for (let i = 1; i < n - 1; i++) for (let j = 1; j < n - 1; j++) {
    const x = (j + r()) * c, y = (i + r()) * c, top = canopy[i * n + j];
    if (top > 1) trees.push([x, y, Math.min(18, 3 + top), 2.5 + Math.min(4, top * 0.5)]);
    else if (free(i, j) && r() < (hgt[i * n + j] > stage ? 0.05 : 0.015)) trees.push([x, y, 4 + r() * 6, 2 + r() * 2.5]);
  }
  const near = (i, j) => [-3, 0, 3].some((a) => [-3, 0, 3].some((b) => !free(Math.min(n - 1, Math.max(0, i + a)), Math.min(n - 1, Math.max(0, j + b)))));
  for (let tries = 0; homes.length < 60 && tries < 4000; tries++) {
    const i = 2 + Math.floor(r() * (n - 4)), j = 2 + Math.floor(r() * (n - 4));
    if (!free(i, j) || canopy[i * n + j] > 1 || hgt[i * n + j] < stage + 0.4 || !near(i, j)) continue;
    const m = 2 + Math.floor(r() * 4);
    for (let h = 0; h < m; h++) homes.push([(j + 0.5) * c + (r() - 0.5) * 22, (i + 0.5) * c + (r() - 0.5) * 22, r(), r()]);
    tries += 40;
  }
  const out = [], m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), col = new THREE.Color();
  const inst = (geo, mat, list, place) => {
    const mesh = new THREE.InstancedMesh(geo, mat, list.length);
    list.forEach((t, i) => place(t, i, mesh));
    mesh.castShadow = mesh.receiveShadow = true; mesh.instanceMatrix.needsUpdate = true; out.push(mesh); return mesh;
  };
  const at = (x, y, lift = 0) => P(x, y, h0(x, y) * EX + lift);
  const put = (mesh, i, p, yaw, sx, sy, sz) => mesh.setMatrixAt(i, m4.compose(p, q.setFromEuler(e.set(0, yaw, 0)), new THREE.Vector3(sx, sy, sz)));
  inst(new THREE.CylinderGeometry(0.1, 0.16, 1, 6).translate(0, 0.5, 0), new THREE.MeshStandardMaterial({ color: 0x4a3a2a, roughness: 1 }), trees,
    ([x, y, h, w], i, mesh) => put(mesh, i, at(x, y), 0, w, h * 0.55, w));
  inst(crown(), new THREE.MeshStandardMaterial({ roughness: 0.9, flatShading: true }), trees, ([x, y, h, w], i, mesh) => {
    put(mesh, i, at(x, y, h * 0.62), r() * 6.3, w, Math.max(1.4, h * 0.3), w);
    mesh.setColorAt(i, col.setHSL(0.2 + r() * 0.07, 0.42 + r() * 0.15, 0.13 + r() * 0.08));
  });
  const huts = homes.filter((h) => h[3] < 0.6), houses = homes.filter((h) => h[3] >= 0.6), mud = new THREE.MeshStandardMaterial({ color: 0x9a7b5a, roughness: 1 });
  inst(new THREE.CylinderGeometry(1, 1, 1, 14).translate(0, 0.5, 0), mud, huts, ([x, y], i, mesh) => put(mesh, i, at(x, y), 0, 2.4, 2.2, 2.4));
  inst(new THREE.ConeGeometry(1, 1, 14).translate(0, 0.5, 0), new THREE.MeshStandardMaterial({ color: 0xb59a62, roughness: 1 }), huts, ([x, y], i, mesh) => put(mesh, i, at(x, y, 2.2), 0, 3, 2.4, 3));
  inst(new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0), mud, houses, ([x, y, a], i, mesh) => put(mesh, i, at(x, y), a * 6.3, 7, 2.6, 4.5));
  inst(new THREE.CylinderGeometry(1, 1, 1, 3).rotateZ(Math.PI / 2).rotateX(-Math.PI / 2).translate(0, 0.5, 0), new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.6 }), houses, ([x, y, a], i, mesh) => {
    put(mesh, i, at(x, y, 2.6), a * 6.3, 7.6, 1.6, 2.6);
    mesh.setColorAt(i, col.set(r() < 0.6 ? 0x9da3a8 : 0x8a4b2c));
  });
  return out;
}
function crown() {
  const g = new THREE.IcosahedronGeometry(1, 2), p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i), s = 1 + 0.16 * Math.sin(5 * x + 2 * y) * Math.cos(4 * z + 1);
    p.setXYZ(i, x * s, y * s, z * s);
  }
  g.computeVertexNormals(); return g;
}
const lanes = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0x0a84ff, dashSize: 10, gapSize: 8, transparent: true, opacity: 0.8 }));
const active = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x64b5ff }));
scene.add(lanes, active); helper(lanes, active);
function updateLanes(alt) {
  const key = [S.lanes, S.activeLane, alt, EX];
  if (lanes.key && key.every((k, i) => k === lanes.key[i])) return;
  lanes.key = key;
  lanes.geometry.setFromPoints(S.lanes.flatMap((l) => [P(l.a[0], l.a[1], alt), P(l.b[0], l.b[1], alt)])); lanes.computeLineDistances();
  const a = !S.done && S.activeLane;
  active.visible = !!a;
  if (a) active.geometry.setFromPoints([P(a.a[0], a.a[1], alt), P(a.b[0], a.b[1], alt)]);
}

// ---------- the drone, drawn DRONE_X times life size ----------
function makeDrone() {
  const g = new THREE.Group(), shell = new THREE.MeshStandardMaterial({ color: 0xebebf0, roughness: 0.32, metalness: 0.15 }), dark = new THREE.MeshStandardMaterial({ color: 0x2c2c2e, roughness: 0.5 });
  const add = (geo, mat, x = 0, y = 0, z = 0, shadow = true) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = shadow; g.add(m); return m; };
  add(new THREE.CapsuleGeometry(0.15, 0.32, 6, 16).rotateZ(Math.PI / 2), shell);
  for (const s of [1, -1]) add(new THREE.BoxGeometry(1.0, 0.035, 0.05), dark).rotation.y = (s * Math.PI) / 4;
  g.blades = [];
  for (const [x, z] of [[0.35, 0.35], [0.35, -0.35], [-0.35, 0.35], [-0.35, -0.35]]) {
    add(new THREE.CylinderGeometry(0.045, 0.05, 0.09, 16), dark, x, 0.04, z);
    add(new THREE.CircleGeometry(0.2, 40).rotateX(-Math.PI / 2), fade(new THREE.MeshBasicMaterial({ color: 0xd6e6ff, opacity: 0.22, side: THREE.DoubleSide })), x, 0.095, z, false);
    g.blades.push(add(new THREE.BoxGeometry(0.4, 0.008, 0.035), dark, x, 0.095, z));
  }
  add(new THREE.SphereGeometry(0.06, 16, 12), dark, 0.2, -0.13, 0);
  const lamp = (color, x, z) => add(new THREE.SphereGeometry(0.03, 8, 6), new THREE.MeshBasicMaterial({ color }), x, 0.02, z, false);
  lamp(0xff3b30, 0.48, -0.48); lamp(0x34c759, 0.48, 0.48);
  g.strobe = lamp(0xffffff, -0.3, 0);
  g.scale.setScalar(DRONE_X); g.visible = false;
  return g;
}
const drone = makeDrone();
scene.add(drone);

// camera cone and its footprint on the ground
const coneMat = fade(new THREE.MeshBasicMaterial({ color: 0xffffff, opacity: 0.07, side: THREE.DoubleSide }));
const cone = new THREE.Mesh(new THREE.BufferGeometry(), coneMat);
const foot = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff }));
scene.add(cone, foot);
helper(drone, cone, foot);

// flown path with a curtain to the ground
const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xf5f5f7 }));
const curtain = new THREE.Mesh(new THREE.BufferGeometry(), fade(new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })));
scene.add(trail, curtain);
helper(trail, curtain);
trail.n = -1; trail.alts = [];
function updateTrail() {
  const path = S.path;
  if (path.length < trail.alts.length) trail.alts = [];
  while (trail.alts.length < path.length) trail.alts.push(S.alt);
  if (path.length === trail.n) return;
  trail.n = path.length;
  const top = path.map(([x, y], i) => P(x, y, trail.alts[i] * EX));
  trail.geometry.dispose(); trail.geometry = new THREE.BufferGeometry().setFromPoints(top);
  const v = new Float32Array(path.length * 6), c = new Float32Array(path.length * 8), idx = [];
  path.forEach(([x, y], i) => {
    v.set([top[i].x, top[i].y, top[i].z, top[i].x, h0(x, y) * EX + 0.5, top[i].z], i * 6);
    c.set([1, 1, 1, 0.07, 1, 1, 1, 0.0], i * 8);
    if (i) idx.push(2 * i - 2, 2 * i - 1, 2 * i, 2 * i - 1, 2 * i + 1, 2 * i);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(v, 3)); g.setAttribute("color", new THREE.BufferAttribute(c, 4)); g.setIndex(idx);
  curtain.geometry.dispose(); curtain.geometry = g;
}

// wind: streaks carried by the wind at the drone, gusts included
const NP = 320, wp = new Float32Array(NP * 6), wc = new Float32Array(NP * 8), seeds = Array.from({ length: NP }, () => ({ x: Math.random() * 1024, y: Math.random() * 1024, h: 6 + Math.random() * 34, life: Math.random() }));
const windGeo = new THREE.BufferGeometry();
windGeo.setAttribute("position", new THREE.BufferAttribute(wp, 3)); windGeo.setAttribute("color", new THREE.BufferAttribute(wc, 4));
const streaks = new THREE.LineSegments(windGeo, fade(new THREE.LineBasicMaterial({ vertexColors: true })));
streaks.frustumCulled = false; scene.add(streaks); helper(streaks);
function updateWind(dt) {
  const [wx, wy] = S.wind || [0, 0], v = Math.hypot(wx, wy), ux = v ? wx / v : 0, uy = v ? wy / v : 0, len = 8 + v * 7, a0 = Math.min(1, v / 2) * 0.32;
  seeds.forEach((s, i) => {
    s.x += wx * 14 * dt; s.y += wy * 14 * dt; s.life += dt * 0.3;
    if (s.life > 1 || s.x < 0 || s.y < 0 || s.x > span || s.y > span) Object.assign(s, { x: Math.random() * span, y: Math.random() * span, life: 0 });
    const h = s.h * EX, a = Math.sin(Math.PI * s.life) * a0;
    wp.set([s.x - span / 2, h, s.y - span / 2, s.x - ux * len - span / 2, h, s.y - uy * len - span / 2], i * 6);
    wc.set([0.85, 0.92, 1, a, 0.85, 0.92, 1, 0], i * 8);
  });
  windGeo.attributes.position.needsUpdate = windGeo.attributes.color.needsUpdate = true;
}

// ---------- labels ----------
function tag(html, cls = "") {
  const el = document.createElement("div"); el.className = "l3 " + cls; el.innerHTML = html; tags.appendChild(el); return el;
}
const v3 = new THREE.Vector3();
function place(el, at) {
  v3.copy(at).project(camera);
  const sx = ((v3.x + 1) / 2) * box.clientWidth, sy = ((1 - v3.y) / 2) * box.clientHeight, r = pipBox;
  const off = v3.z > 1 || Math.abs(v3.x) > 1.1 || Math.abs(v3.y) > 1.1 || (r && sx > r.x - 40 && sx < r.x + r.w + 40 && sy > r.y && sy < r.y + r.h + 30);
  el.style.display = off ? "none" : "";
  if (off) return;
  el.style.transform = `translate(${sx}px, ${sy}px) translate(-50%, calc(-100% - 12px))`;
  el.style.zIndex = String(10000 - Math.round(camera.position.distanceTo(at)));
}
const paid = new Map(), rings = [];
function updateTags() {
  for (const { q, el } of parcelTags) {
    const c = S.claims.get(q.id)?.approved ? S.claims.get(q.id) : null, st = (c ? "paid" : S.refused.has(q.id) ? "refused" : S.surge.has(q.id) ? "surge" : S.focus.has(q.id) ? "focus" : "") + (sel === q.id ? " sel" : "");
    if (el.dataset.st !== st) { el.className = "l3 pick " + st; el.dataset.st = st; el.querySelector("span").textContent = c ? fmtKES(c.amount) : st.startsWith("refused") ? "needs another look" : st.startsWith("surge") ? "just flooded" : ""; }
    if (c && !paid.has(q.id)) { paid.set(q.id, true); pulse(q); }
    place(el, P(q.cx, q.cy, h0(q.cx, q.cy) * EX + 14));
  }
  for (const z of zoneTags) place(z.el, z.at);
  if (hubTag) place(hubTag.el, hubTag.at);
}
function pulse(q) {
  if (calm) return;
  const m = new THREE.Mesh(new THREE.RingGeometry(10, 13, 64).rotateX(-Math.PI / 2), fade(new THREE.MeshBasicMaterial({ color: 0x30d158, side: THREE.DoubleSide })));
  m.position.copy(P(q.cx, q.cy, h0(q.cx, q.cy) * EX + 2)); m.t = 0; scene.add(m); helper(m); rings.push(m);
}

// ---------- drone state from the event stream ----------
const ds = { last: null, t: 0, vel: new THREE.Vector2(), shown: false };
const target = new THREE.Vector3(), air = new THREE.Vector2(), qYaw = new THREE.Quaternion(), qTilt = new THREE.Quaternion(), axis = new THREE.Vector3();
function updateDrone(dt) {
  drone.visible = cone.visible = foot.visible = !!(S.pose && hgt);
  if (!drone.visible) { ds.last = null; ds.shown = false; return; }
  const { x, y, yaw } = S.pose;
  if (!ds.last || ds.last.x !== x || ds.last.y !== y) {
    if (ds.last && S.t > ds.t) ds.vel.lerp(new THREE.Vector2(x - ds.last.x, y - ds.last.y).divideScalar(S.t - ds.t), 0.5);
    ds.last = { x, y }; ds.t = S.t;
  }
  target.copy(P(x, y, S.alt * EX));
  if (!ds.shown || drone.position.distanceTo(target) > 250) { drone.position.copy(target); ds.shown = true; }
  else drone.position.lerp(target, 1 - Math.exp(-dt * 6));
  // a multirotor tilts its thrust into the airspeed: ground velocity minus wind
  const [wx, wy] = S.wind || [0, 0];
  air.set(ds.vel.x - wx, ds.vel.y - wy);
  const s = air.length(), tilt = Math.min(MAX_TILT, (s / 15) * MAX_TILT);
  qYaw.setFromAxisAngle(UP, -yaw);
  axis.set(air.y, 0, -air.x).normalize();
  qTilt.setFromAxisAngle(s > 0.2 ? axis : UP, s > 0.2 ? tilt : 0);
  drone.quaternion.slerp(qTilt.multiply(qYaw), 1 - Math.exp(-dt * 4));
  for (const b of drone.blades) b.rotation.y += dt * 40;
  drone.strobe.visible = (performance.now() % 1200) < 90;
  // camera cone down to a footprint that follows the ground
  const h = S.alt * S.world.fov_tan, k = 8, ring = [];
  for (const [ax, ay, bx, by] of [[-1, -1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [-1, 1, -1, -1]])
    for (let i = 0; i < k; i++) { const f = i / k, gx = x + h * (ax + (bx - ax) * f), gy = y + h * (ay + (by - ay) * f); ring.push(P(gx, gy, h0(gx, gy) * EX + 0.8)); }
  foot.geometry.setFromPoints(ring);
  const pts = [];
  ring.forEach((p, i) => pts.push(drone.position, p, ring[(i + 1) % ring.length]));
  cone.geometry.setFromPoints(pts);
  const col = S.sensorDegraded ? 0xffd60a : 0xffffff;
  foot.material.color.setHex(col); coneMat.color.setHex(col);
}

// ---------- the drone's own camera, rendered from the same world ----------
const droneCam = new THREE.PerspectiveCamera(80, 1, 1, 6000), pip = $("pip");
let pipBox = null;
function renderPip() {
  const [hx, hy] = S.world?.home || [0, 0];
  const on = drone.visible && !!S.world && !S.done && Math.hypot(S.pose.x - hx, S.pose.y - hy) > 20;
  pip.classList.toggle("on", on);
  pipBox = null;
  if (!on) return;
  const b = box.getBoundingClientRect(), r = pip.getBoundingClientRect();
  pipBox = { x: r.left - b.left, y: r.top - b.top, w: r.width, h: r.height };
  droneCam.fov = THREE.MathUtils.radToDeg(2 * Math.atan(S.world.fov_tan / EX));
  droneCam.updateProjectionMatrix();
  droneCam.position.copy(drone.position);
  droneCam.up.set(Math.cos(S.pose.yaw), 0, Math.sin(S.pose.yaw));
  droneCam.lookAt(drone.position.x, 0, drone.position.z);
  pip.classList.toggle("degraded", !!S.sensorDegraded);
  $("pipMeta").textContent = `nadir ${Math.round(S.alt)} m, quality ${(S.quality ?? 0).toFixed(2)}`;
  const y = b.height - pipBox.y - pipBox.h;
  renderer.setScissorTest(true);
  renderer.setViewport(pipBox.x, y, pipBox.w, pipBox.h); renderer.setScissor(pipBox.x, y, pipBox.w, pipBox.h);
  renderer.shadowMap.autoUpdate = false; // the main view just drew this frame's shadows
  renderer.render(scene, droneCam);
  renderer.shadowMap.autoUpdate = true;
  renderer.setScissorTest(false); renderer.setViewport(0, 0, b.width, b.height);
}

// ---------- camera modes ----------
function flyTo(pos, look) { fly = { pos, look }; }
function setMode(m) {
  mode = m; touched = false;
  for (const b of $("camMode").children) b.classList.toggle("active", b.dataset.c === m);
  if (m === "top") flyTo(new THREE.Vector3(0, span * 1.9, 1), new THREE.Vector3(0, 0, 0));
  if (m === "orbit") flyTo(new THREE.Vector3(span * 0.55, span * 1.05, span * 0.95), new THREE.Vector3(0, 0, 0));
  if (m === "follow") fly = null;
}
for (const b of $("camMode").children) b.onclick = () => setMode(b.dataset.c);
for (const b of $("exag").children) b.onclick = () => {
  EX = +b.dataset.e; land.scale.y = EX; trail.n = -1;
  for (const x of $("exag").children) x.classList.toggle("active", x === b);
  $("s3ex").textContent = `Heights ×${EX}`;
  if (S.world) build();
};
const prevLook = new THREE.Vector3();
let auto = true; // chase the drone while it flies, show the whole map when the mission ends, unless the viewer took over
for (const b of $("camMode").children) b.addEventListener("click", () => (auto = false));
function steer(dt) {
  const k = calm ? 1 : 1 - Math.exp(-dt * 3);
  if (auto && !touched) {
    if (S.pose && !S.done && drone.visible && mode !== "follow") setMode("follow");
    if (S.done && mode === "follow") setMode("orbit");
  }
  if (mode === "follow" && S.pose && drone.visible) {
    const back = new THREE.Vector3(-Math.cos(S.pose.yaw), 0, -Math.sin(S.pose.yaw));
    prevLook.copy(controls.target);
    controls.target.lerp(drone.position, k);
    if (!touched) camera.position.lerp(drone.position.clone().addScaledVector(back, 480).setY(drone.position.y + 640), k);
    else camera.position.add(controls.target.clone().sub(prevLook));
  } else if (fly) {
    camera.position.lerp(fly.pos, k); controls.target.lerp(fly.look, k);
    if (camera.position.distanceTo(fly.pos) < 2) fly = null;
  }
  controls.autoRotate = !calm && mode === "orbit" && !touched && !fly && (!S.pose || S.done);
  $("compass").style.transform = `rotate(${controls.getAzimuthalAngle()}rad)`;
}

// ---------- view switch and loop ----------
function fit() {
  const w = box.clientWidth, h = box.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(fit).observe(box);
for (const b of $("viewMode").children) b.onclick = () => {
  const on = b.dataset.v === "3d";
  for (const x of $("viewMode").children) x.classList.toggle("active", x === b);
  box.classList.toggle("on", on); $("map").style.visibility = on ? "hidden" : ""; dirty = true; fit();
};
for (const b of $("groundMode").children) b.onclick = () => {
  const [sat, over] = b.dataset.g.split(",").map(Number);
  if (sat >= 0) U.uSat.value = sat;
  U.uOverlay.value = over;
  $("credit").style.display = U.uSat.value ? "" : "none";
  for (const x of $("groundMode").children) x.classList.toggle("active", x === b);
};
// click the ground to pick a field; a drag is an orbit, not a click
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let down = null;
renderer.domElement.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || !S.world || !land.children[0]) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObject(land.children[0])[0], W = S.world, m = W.map_px;
  if (!hit) return window.select(null);
  const j = Math.floor(((hit.point.x + span / 2) / span) * m), i = Math.floor(((hit.point.z + span / 2) / span) * m);
  const k = S.pmap[Math.min(m - 1, Math.max(0, i)) * m + Math.min(m - 1, Math.max(0, j))];
  window.select(k === 255 ? null : W.parcels[k].id);
});
function pick(id) {
  const k = id ? S.world.parcels.findIndex((q) => q.id === id) : -1;
  U.uSel.value = k;
  if (k >= 0 && mode !== "follow") {
    const q = S.world.parcels[k], at = P(q.cx, q.cy, h0(q.cx, q.cy) * EX), off = camera.position.clone().sub(controls.target).setLength(520);
    touched = true; flyTo(at.clone().add(off), at);
  }
}
window.view3d = { world, select: pick };

function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(0.1, clock.getDelta());
  if (!box.classList.contains("on")) return;
  if (S.world) {
    paintGround(); updateDrone(dt); updateTrail();
    updateLanes((S.alt || 40) * EX); updateTags();
  }
  lanes.visible = !!S.world && S.lanes.length > 0;
  updateWind(dt);
  for (let i = rings.length - 1; i >= 0; i--) {
    const r = rings[i]; r.t += dt; r.scale.setScalar(1 + r.t * 3); r.material.opacity = Math.max(0, 1 - r.t / 1.6);
    if (r.t > 1.6) { scene.remove(r); r.geometry.dispose(); r.material.dispose(); rings.splice(i, 1); }
  }
  if (waterMat) waterMat.uniforms.time.value += dt;
  steer(dt); controls.update();
  renderer.render(scene, camera);
  renderPip();
}
if (S.world) world(S.world);
$("viewMode").children[1].click();
tick();
