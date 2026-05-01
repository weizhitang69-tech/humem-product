import * as THREE from "./vendor/three.module.min.js";

const canvas = document.querySelector("#scene");
const statsEl = document.querySelector("#stats");
const vitalsEl = document.querySelector("#vitals");
const insightsEl = document.querySelector("#insights");
const searchEl = document.querySelector("#search");
const resetEl = document.querySelector("#resetView");
const layersEl = document.querySelector("#layerToggles");
const detailsEl = document.querySelector("#details");
const tooltipEl = document.querySelector("#tooltip");
const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050706);
scene.fog = new THREE.FogExp2(0x050706, 0.032);

const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 1000);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const clock = new THREE.Clock();

const layerGroup = new THREE.Group();
const graphGroup = new THREE.Group();
const synapseGroup = new THREE.Group();
const atmosphereGroup = new THREE.Group();
scene.add(atmosphereGroup, layerGroup, graphGroup, synapseGroup);

scene.add(new THREE.HemisphereLight(0xf8f0d8, 0x1b272a, 1.55));
const keyLight = new THREE.DirectionalLight(0xffdfaa, 1.95);
keyLight.position.set(8, 12, 10);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0x7ae8d3, 4.8, 48);
rimLight.position.set(-8, 6, -8);
scene.add(rimLight);

let graph = { nodes: [], links: [], meta: { totalLayers: 1, layerHistogram: [] } };
let nodeById = new Map();
let adjacency = new Map();
let visibleLayers = new Set();
let synapsePulses = [];
let selectedId = null;
let hoverId = null;
let searchTerm = "";
let viewMode = "organism";
let pointerDown = null;
let isDragging = false;
let dragMode = "rotate";

const cameraState = {
  theta: -0.78,
  phi: 1.06,
  radius: 19,
  target: new THREE.Vector3(0, 2.4, 0),
  desiredTheta: -0.78,
  desiredPhi: 1.06,
  desiredRadius: 19,
  desiredTarget: new THREE.Vector3(0, 2.4, 0),
};

init();

async function init() {
  try {
    const response = await fetch("./api/graph", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    graph = await response.json();
    visibleLayers = new Set(
      Array.from({ length: Math.max(graph.meta.totalLayers || 1, 1) }, (_, layer) => layer),
    );
    buildScene();
    buildLayerControls();
    updateStats();
    renderInsights();
    renderDetails(null);
    animate();
  } catch (error) {
    statsEl.textContent = `加载失败: ${error.message}`;
    renderDetails(null, "无法读取 memory store");
  }
}

function buildScene() {
  graphGroup.clear();
  layerGroup.clear();
  synapseGroup.clear();
  atmosphereGroup.clear();
  nodeById = new Map();
  adjacency = new Map();
  synapsePulses = [];

  const totalLayers = Math.max(graph.meta.totalLayers || 1, 1);
  const layerGap = 1.55;
  const spread = Math.max(7, Math.sqrt(Math.max(graph.nodes.length, 1)) * 1.45);
  const topColor = new THREE.Color(0xf4c56a);
  const bottomColor = new THREE.Color(0x5662d9);

  buildMemoryAtmosphere(spread, totalLayers, layerGap);
  buildLayerMembranes(totalLayers, layerGap, spread, topColor, bottomColor);
  buildNodes(totalLayers, layerGap, spread, topColor, bottomColor);
  buildLinks();
  updateVisualState();
}

function buildMemoryAtmosphere(spread, totalLayers, layerGap) {
  const geometry = new THREE.BufferGeometry();
  const positions = [];
  const colors = [];
  const colorA = new THREE.Color(0x7ae8d3);
  const colorB = new THREE.Color(0xf4c56a);
  const height = Math.max(totalLayers - 1, 1) * layerGap;
  const count = 180;
  for (let index = 0; index < count; index += 1) {
    const angle = index * 2.39996;
    const radius = Math.sqrt((index + 1) / count) * spread * 1.42;
    const jitter = Math.sin(index * 17.13) * 0.32;
    positions.push(
      Math.cos(angle) * (radius + jitter),
      (Math.sin(index * 0.71) * 0.5) * height,
      Math.sin(angle) * (radius - jitter),
    );
    const color = new THREE.Color().lerpColors(colorA, colorB, (Math.sin(index) + 1) / 2);
    colors.push(color.r, color.g, color.b);
  }
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: 0.035,
    vertexColors: true,
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  points.userData.drift = true;
  atmosphereGroup.add(points);
}

function buildLayerMembranes(totalLayers, layerGap, spread, topColor, bottomColor) {
  for (let layer = 0; layer < totalLayers; layer += 1) {
    const y = layerToY(layer, totalLayers, layerGap);
    const t = totalLayers === 1 ? 0 : layer / (totalLayers - 1);
    const color = new THREE.Color().lerpColors(topColor, bottomColor, t);
    const radius = spread * (1.18 + t * 0.08);

    const membrane = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 96),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.025 + (1 - t) * 0.022,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    membrane.rotation.x = Math.PI / 2;
    membrane.position.y = y;
    layerGroup.add(membrane);

    const ring = new THREE.Mesh(
      new THREE.RingGeometry(radius * 0.985, radius, 128),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.24 - t * 0.08,
        side: THREE.DoubleSide,
      }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.y = y + 0.003;
    layerGroup.add(ring);

    const grid = new THREE.GridHelper(radius * 2.0, 10, color, color);
    grid.position.y = y;
    grid.material.transparent = true;
    grid.material.opacity = 0.075 - t * 0.025;
    layerGroup.add(grid);
  }
}

function buildNodes(totalLayers, layerGap, spread, topColor, bottomColor) {
  for (const node of graph.nodes) {
    const strength = clamp(Number(node.strength) || 0, 0, 4);
    const activation = clamp(Number(node.activation) || 0, 0, 3);
    const depth = Number(node.depth ?? node.layer) || 0;
    const layerRatio = totalLayers === 1 ? 0 : depth / (totalLayers - 1);
    const isAnchor = Boolean(node.isConsolidationAnchor);
    const color = isAnchor
      ? new THREE.Color(0xf4c56a)
      : new THREE.Color().lerpColors(topColor, bottomColor, layerRatio);
    const radius = (isAnchor ? 0.22 : 0.13) + Math.sqrt(strength + 0.25) * (isAnchor ? 0.085 : 0.07);
    const geometry = isAnchor
      ? new THREE.IcosahedronGeometry(radius, 1)
      : new THREE.SphereGeometry(radius, 24, 18);
    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: isAnchor ? 0.26 : 0.44,
      metalness: isAnchor ? 0.28 : 0.12,
      emissive: color,
      emissiveIntensity: (isAnchor ? 0.35 : 0.16) + activation * 0.13,
      transparent: true,
      opacity: 0.94,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(nodeToPosition(node, totalLayers, layerGap, spread));
    mesh.userData.nodeId = node.id;
    graphGroup.add(mesh);

    let halo = null;
    if (isAnchor) {
      halo = new THREE.Mesh(
        new THREE.RingGeometry(radius * 1.32, radius * 1.52, 48),
        new THREE.MeshBasicMaterial({
          color: 0xf4c56a,
          transparent: true,
          opacity: 0.48,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      halo.position.copy(mesh.position);
      halo.rotation.x = Math.PI / 2;
      graphGroup.add(halo);
    }

    nodeById.set(node.id, {
      data: node,
      mesh,
      halo,
      material,
      baseRadius: radius,
      baseColor: color.clone(),
    });
    adjacency.set(node.id, []);
  }
}

function buildLinks() {
  for (const link of graph.links) {
    const source = nodeById.get(link.source);
    const target = nodeById.get(link.target);
    if (!source || !target) {
      continue;
    }

    const color = relationColor(link);
    const geometry = new THREE.BufferGeometry().setFromPoints([
      source.mesh.position,
      target.mesh.position,
    ]);
    const material = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.18 + clamp(Number(link.weight) || 0, 0, 1) * 0.34,
    });
    const line = new THREE.Line(geometry, material);
    line.userData.linkId = link.id;
    line.userData.sourceId = link.source;
    line.userData.targetId = link.target;
    graphGroup.add(line);

    const pulse = new THREE.Mesh(
      new THREE.SphereGeometry(0.035 + clamp(Number(link.weight) || 0, 0, 1) * 0.024, 12, 8),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
      }),
    );
    pulse.userData.sourceId = link.source;
    pulse.userData.targetId = link.target;
    pulse.userData.phase = Math.random();
    synapseGroup.add(pulse);
    synapsePulses.push(pulse);

    const edge = { link, other: link.target, line, pulse, material };
    adjacency.get(link.source).push(edge);
    adjacency.get(link.target).push({ link, other: link.source, line, pulse, material });
  }
}

function buildLayerControls() {
  layersEl.innerHTML = "";
  const totalLayers = Math.max(graph.meta.totalLayers || 1, 1);
  const histogram = graph.meta.layerHistogram || [];

  for (let layer = 0; layer < totalLayers; layer += 1) {
    const row = document.createElement("label");
    row.className = "layer-row";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        visibleLayers.add(layer);
      } else {
        visibleLayers.delete(layer);
      }
      updateVisualState();
    });

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = layerColor(layer, totalLayers);

    const label = document.createElement("span");
    label.className = "layer-name";
    label.innerHTML = `${layerName(layer, totalLayers)}<span class="layer-depth">layer ${layer}</span>`;

    const count = document.createElement("span");
    count.className = "count";
    count.textContent = histogram[layer] ?? 0;

    row.append(checkbox, swatch, label, count);
    layersEl.append(row);
  }
}

function updateStats() {
  const meta = graph.meta;
  const layout = meta.layoutModel || "layout";
  const scope = meta.embeddingScope || "none";
  const profile = meta.retrievalProfile || "balanced";
  const anchors = meta.consolidationAnchorCount || 0;
  statsEl.textContent = `${meta.fragmentCount} memories · ${meta.relationCount} synapses · ${anchors} anchors · ${meta.totalLayers} laminae · ${profile} · ${layout} · ${scope}`;

  const vitals = computeVitals();
  vitalsEl.innerHTML = `
    <div class="vital"><span class="vital-label">Activation</span><span class="vital-value">${formatNumber(vitals.activation)}</span></div>
    <div class="vital"><span class="vital-label">Strength</span><span class="vital-value">${formatNumber(vitals.strength)}</span></div>
    <div class="vital"><span class="vital-label">Access</span><span class="vital-value">${formatNumber(vitals.access)}</span></div>
    <div class="vital"><span class="vital-label">Anchors</span><span class="vital-value">${anchors}</span></div>
  `;
}

function renderInsights() {
  const vitals = computeVitals();
  const totalLayers = Math.max(graph.meta.totalLayers || 1, 1);
  const upper = graph.nodes.filter((node) => node.layer <= 1).length;
  const deep = graph.nodes.filter((node) => node.layer >= totalLayers - 2).length;
  const anchorRatio = graph.nodes.length ? (graph.meta.consolidationAnchorCount || 0) / graph.nodes.length : 0;
  const items = [
    { name: "Upper Recall", value: upper, ratio: graph.nodes.length ? upper / graph.nodes.length : 0 },
    { name: "Deep Trace", value: deep, ratio: graph.nodes.length ? deep / graph.nodes.length : 0 },
    { name: "Anchor Density", value: `${Math.round(anchorRatio * 100)}%`, ratio: anchorRatio },
    { name: "Mean Access", value: formatNumber(vitals.access), ratio: vitals.access },
  ];
  insightsEl.innerHTML = items
    .map(
      (item) => `
        <div class="insight">
          <span class="insight-name">${escapeHtml(item.name)}</span>
          <span class="insight-value">${escapeHtml(item.value)}</span>
          <div class="insight-bar"><span style="width:${Math.round(clamp(item.ratio, 0, 1) * 100)}%"></span></div>
        </div>
      `,
    )
    .join("");
}

function animate() {
  requestAnimationFrame(animate);
  const delta = Math.min(clock.getDelta(), 0.05);
  smoothCamera(delta);
  const time = clock.elapsedTime;

  atmosphereGroup.rotation.y += delta * 0.012;
  layerGroup.rotation.y = Math.sin(time * 0.08) * 0.018;

  for (const entry of nodeById.values()) {
    const isSelected = entry.data.id === selectedId;
    const isHover = entry.data.id === hoverId;
    const pulse = 1 + Math.sin(time * 2.2 + entry.data.layer) * (entry.data.isConsolidationAnchor ? 0.045 : 0.025);
    const scale = (isSelected ? 1.82 : isHover ? 1.42 : 1) * pulse;
    entry.mesh.scale.setScalar(scale);
    if (entry.halo) {
      entry.halo.scale.setScalar(1 + Math.sin(time * 1.6) * 0.09 + (isSelected ? 0.22 : 0));
      entry.halo.lookAt(camera.position);
    }
  }

  for (const pulse of synapsePulses) {
    const source = nodeById.get(pulse.userData.sourceId);
    const target = nodeById.get(pulse.userData.targetId);
    if (!source || !target) {
      continue;
    }
    const t = (time * 0.18 + pulse.userData.phase) % 1;
    pulse.position.lerpVectors(source.mesh.position, target.mesh.position, t);
  }

  renderer.render(scene, camera);
}

function smoothCamera(delta) {
  const factor = 1 - Math.pow(0.001, delta);
  cameraState.theta += (cameraState.desiredTheta - cameraState.theta) * factor;
  cameraState.phi += (cameraState.desiredPhi - cameraState.phi) * factor;
  cameraState.radius += (cameraState.desiredRadius - cameraState.radius) * factor;
  cameraState.target.lerp(cameraState.desiredTarget, factor);

  const sinPhi = Math.sin(cameraState.phi);
  camera.position.set(
    cameraState.target.x + cameraState.radius * sinPhi * Math.cos(cameraState.theta),
    cameraState.target.y + cameraState.radius * Math.cos(cameraState.phi),
    cameraState.target.z + cameraState.radius * sinPhi * Math.sin(cameraState.theta),
  );
  camera.lookAt(cameraState.target);
}

function updateVisualState() {
  const related = relatedSetForSelection();
  const totalLayers = Math.max(graph.meta.totalLayers || 1, 1);

  for (const [id, entry] of nodeById.entries()) {
    const layerVisible = visibleLayers.has(entry.data.layer);
    const matchesSearch = !searchTerm || entry.data.text.toLowerCase().includes(searchTerm);
    const inFocus = !selectedId || related.has(id);
    const modeVisible = isVisibleInMode(entry.data, related, totalLayers);
    const visible = layerVisible && modeVisible;
    const strong = matchesSearch && inFocus;
    entry.mesh.visible = visible;
    entry.material.opacity = visible ? (strong ? 0.96 : 0.18) : 0;
    entry.material.emissiveIntensity = strong ? 0.28 + entry.data.activation * 0.13 : 0.045;
    if (entry.halo) {
      entry.halo.visible = visible;
      entry.halo.material.opacity = strong ? 0.55 : 0.18;
    }
  }

  const seenLines = new Set();
  for (const edges of adjacency.values()) {
    for (const edge of edges) {
      if (seenLines.has(edge.link.id)) {
        continue;
      }
      seenLines.add(edge.link.id);
      const source = nodeById.get(edge.link.source);
      const target = nodeById.get(edge.link.target);
      const visible = source?.mesh.visible && target?.mesh.visible;
      const highlighted = selectedId && (edge.link.source === selectedId || edge.link.target === selectedId);
      edge.line.visible = Boolean(visible);
      edge.pulse.visible = Boolean(visible && (highlighted || !selectedId));
      edge.material.opacity = highlighted ? 0.88 : selectedId ? 0.055 : 0.2 + edge.link.weight * 0.24;
      edge.pulse.material.opacity = highlighted ? 0.92 : 0.42;
    }
  }
}

function relatedSetForSelection() {
  const related = new Set();
  if (selectedId) {
    related.add(selectedId);
    for (const edge of adjacency.get(selectedId) || []) {
      related.add(edge.other);
    }
  }
  return related;
}

function isVisibleInMode(node, related, totalLayers) {
  if (viewMode === "anchors") {
    return Boolean(node.isConsolidationAnchor) || related.has(node.id);
  }
  if (viewMode === "deep") {
    return node.layer >= totalLayers - 2 || Boolean(node.isConsolidationAnchor) || related.has(node.id);
  }
  return true;
}

function selectNode(id, focus = false) {
  selectedId = id;
  const entry = nodeById.get(id);
  if (!entry) {
    return;
  }
  if (focus) {
    focusOn(entry.mesh.position);
  }
  renderDetails(entry.data);
  updateVisualState();
}

function focusOn(position) {
  cameraState.desiredTarget.copy(position);
  cameraState.desiredRadius = clamp(cameraState.desiredRadius, 5.2, 11);
}

function renderDetails(node, message = null) {
  if (!node) {
    detailsEl.innerHTML = `<div class="empty-state">${escapeHtml(message || "选择一个记忆节点")}</div>`;
    return;
  }

  const related = (adjacency.get(node.id) || [])
    .map((edge) => ({ edge, node: nodeById.get(edge.other)?.data }))
    .filter((item) => item.node)
    .sort((left, right) => right.edge.link.weight - left.edge.link.weight);
  const activationRatio = clamp((Number(node.activation) || 0) / 3, 0, 1);
  const memoryState = node.isConsolidationAnchor
    ? "consolidated"
    : node.layer >= Math.max(graph.meta.totalLayers - 2, 0)
      ? "deep trace"
      : "retrievable";

  detailsEl.innerHTML = `
    <div class="detail-content">
      <div class="kind-row">
        <span class="kind">${escapeHtml(node.isConsolidationAnchor ? "anchor" : node.kind)}</span>
        <span class="memory-state">${escapeHtml(memoryState)}</span>
      </div>
      <div class="detail-title">${escapeHtml(node.text)}</div>
      <div class="memory-ring" style="--activation:${Math.round(activationRatio * 360)}deg">
        <div class="ring-inner">
          <span class="ring-value">${formatNumber(node.activation)}</span>
          <span class="ring-label">activation</span>
        </div>
      </div>
      <div class="metric-grid">
        <div class="metric"><span class="metric-label">Layer</span><span class="metric-value">${node.layer}</span></div>
        <div class="metric"><span class="metric-label">Depth</span><span class="metric-value">${formatNumber(node.depth)}</span></div>
        <div class="metric"><span class="metric-label">Retrievals</span><span class="metric-value">${node.retrievals || 0}</span></div>
        <div class="metric"><span class="metric-label">Strength</span><span class="metric-value">${formatNumber(node.strength)}</span></div>
        <div class="metric"><span class="metric-label">Access</span><span class="metric-value">${formatNumber(node.accessibility)}</span></div>
        <div class="metric"><span class="metric-label">Synapses</span><span class="metric-value">${related.length}</span></div>
      </div>
      ${renderSource(node)}
      ${renderChunk(node)}
      ${renderConsolidation(node)}
      ${renderLayout(node)}
      <div class="section">
        <h2>Synaptic Neighbors</h2>
        <div class="related-list">
          ${
            related.length
              ? related
                  .map(
                    (item) => `
                      <button class="related-item" data-node-id="${escapeHtml(item.node.id)}">
                        <div class="related-meta">${escapeHtml(item.edge.link.type)} · layer ${item.node.layer} · depth ${formatNumber(item.node.depth)} · ${formatNumber(item.edge.link.weight)}</div>
                        <div class="related-text">${escapeHtml(item.node.text)}</div>
                      </button>
                    `,
                  )
                  .join("")
              : `<div class="source">暂无一阶关联记忆</div>`
          }
        </div>
      </div>
    </div>
  `;

  detailsEl.querySelectorAll("[data-node-id]").forEach((button) => {
    button.addEventListener("click", () => selectNode(button.dataset.nodeId, true));
  });
}

function renderLayout(node) {
  return `
    <div class="section">
      <h2>Layout</h2>
      <p class="source">${escapeHtml(node.layoutModel || "hash-fallback")} · ${escapeHtml(node.embeddingScope || "none")} · semantic ${node.semanticEdgeCount || 0} · relation ${node.relationEdgeCount || 0}${node.layoutUpdatedAt ? ` · ${escapeHtml(node.layoutUpdatedAt)}` : ""}</p>
    </div>
  `;
}

function renderConsolidation(node) {
  if (!node.isConsolidationAnchor || !node.consolidation) {
    return "";
  }
  const terms = Array.isArray(node.consolidation.theme_terms)
    ? node.consolidation.theme_terms
    : [];
  return `
    <div class="section">
      <h2>Consolidation</h2>
      <p class="source">${escapeHtml(node.consolidation.scope || "memory")} · score ${formatNumber(node.consolidation.score)}</p>
      <div class="term-list">${terms.map((term) => `<span class="term-chip">${escapeHtml(term)}</span>`).join("")}</div>
    </div>
  `;
}

function renderSource(node) {
  if (!node.source) {
    return "";
  }
  const title = node.source.title || node.source.documentId || "memory";
  const chunk = node.source.chunkId ? ` · ${node.source.chunkId}` : "";
  return `
    <div class="section">
      <h2>Source</h2>
      <p class="source">${escapeHtml(title + chunk)}</p>
    </div>
  `;
}

function renderChunk(node) {
  if (!node.chunkText || node.chunkText === node.text) {
    return "";
  }
  return `
    <div class="section">
      <h2>Chunk</h2>
      <p class="chunk">${escapeHtml(node.chunkText)}</p>
    </div>
  `;
}

function updateHover(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(Array.from(nodeById.values()).map((entry) => entry.mesh), false);
  const hit = hits.find((item) => item.object.visible && item.object.material.opacity > 0.25);
  hoverId = hit?.object.userData.nodeId || null;
  updateTooltip(event, hoverId);
}

function updateTooltip(event, id) {
  if (!id || isDragging) {
    tooltipEl.style.opacity = "0";
    return;
  }
  const node = nodeById.get(id)?.data;
  if (!node) {
    tooltipEl.style.opacity = "0";
    return;
  }
  const kind = node.isConsolidationAnchor ? "anchor" : node.kind;
  tooltipEl.innerHTML = `${escapeHtml(node.text)}<br><span style="color:#9eaaa5">layer ${node.layer} · depth ${formatNumber(node.depth)} · ${escapeHtml(kind)}</span>`;
  tooltipEl.style.left = `${event.clientX}px`;
  tooltipEl.style.top = `${event.clientY}px`;
  tooltipEl.style.opacity = "1";
}

canvas.addEventListener("pointerdown", (event) => {
  pointerDown = { x: event.clientX, y: event.clientY, button: event.button };
  isDragging = false;
  dragMode = event.button === 1 || event.button === 2 ? "pan" : "rotate";
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (pointerDown) {
    const dx = event.clientX - pointerDown.x;
    const dy = event.clientY - pointerDown.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) {
      isDragging = true;
    }
    if (isDragging) {
      if (dragMode === "pan") {
        panCamera(dx, dy);
      } else {
        cameraState.desiredTheta -= dx * 0.0044;
        cameraState.desiredPhi = clamp(cameraState.desiredPhi - dy * 0.0034, 0.22, Math.PI - 0.18);
      }
      pointerDown.x = event.clientX;
      pointerDown.y = event.clientY;
    }
  }
  updateHover(event);
});

canvas.addEventListener("pointerup", (event) => {
  if (pointerDown && !isDragging && event.button === 0 && hoverId) {
    selectNode(hoverId);
  }
  pointerDown = null;
  isDragging = false;
});

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    if (event.shiftKey) {
      panCamera(event.deltaY, 0, 0.018);
    } else if (event.ctrlKey) {
      panCamera(0, event.deltaY, 0.018);
    } else {
      const zoom = Math.exp(event.deltaY * 0.0012);
      cameraState.desiredRadius = clamp(cameraState.desiredRadius * zoom, 2.6, 70);
    }
  },
  { passive: false },
);

function panCamera(dx, dy, scaleOverride = null) {
  const scale = scaleOverride ?? cameraState.desiredRadius * 0.0014;
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
  cameraState.desiredTarget.addScaledVector(right, -dx * scale);
  cameraState.desiredTarget.addScaledVector(up, dy * scale);
}

searchEl.addEventListener("input", () => {
  searchTerm = searchEl.value.trim().toLowerCase();
  updateVisualState();
});

searchEl.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || !searchTerm) {
    return;
  }
  const match = graph.nodes.find((node) => visibleLayers.has(node.layer) && node.text.toLowerCase().includes(searchTerm));
  if (match) {
    selectNode(match.id, true);
  }
});

resetEl.addEventListener("click", resetView);

for (const button of modeButtons) {
  button.addEventListener("click", () => {
    viewMode = button.dataset.mode || "organism";
    for (const item of modeButtons) {
      item.classList.toggle("is-active", item === button);
    }
    updateVisualState();
  });
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function resetView() {
  selectedId = null;
  hoverId = null;
  searchTerm = "";
  viewMode = "organism";
  searchEl.value = "";
  for (const item of modeButtons) {
    item.classList.toggle("is-active", item.dataset.mode === "organism");
  }
  cameraState.desiredTheta = -0.78;
  cameraState.desiredPhi = 1.06;
  cameraState.desiredRadius = 19;
  cameraState.desiredTarget.set(0, 2.4, 0);
  renderDetails(null);
  updateVisualState();
}

function layerToY(layer, totalLayers, gap) {
  const midpoint = (totalLayers - 1) / 2;
  return (midpoint - layer) * gap;
}

function nodeToPosition(node, totalLayers, gap, spread) {
  const height = clamp(Number(node.z) || 0, 0, 1);
  const vertical = (height - 0.5) * Math.max(totalLayers - 1, 1) * gap;
  return new THREE.Vector3(
    (Number(node.x) || 0) * spread,
    vertical,
    (Number(node.y) || 0) * spread,
  );
}

function layerColor(layer, totalLayers) {
  const top = new THREE.Color(0xf4c56a);
  const bottom = new THREE.Color(0x5662d9);
  const ratio = totalLayers <= 1 ? 0 : layer / (totalLayers - 1);
  return `#${new THREE.Color().lerpColors(top, bottom, ratio).getHexString()}`;
}

function layerName(layer, totalLayers) {
  if (layer === 0) {
    return "Cortical";
  }
  if (layer >= totalLayers - 2) {
    return "Submerged";
  }
  return "Trace";
}

function relationColor(link) {
  if (link.type === "consolidates") {
    return 0xf4c56a;
  }
  if (link.crossLayer) {
    return 0xff8b6e;
  }
  if (link.type === "embedding") {
    return 0x9ca3ff;
  }
  return 0x7ae8d3;
}

function computeVitals() {
  const count = Math.max(graph.nodes.length, 1);
  const totals = graph.nodes.reduce(
    (acc, node) => {
      acc.activation += Number(node.activation) || 0;
      acc.strength += Number(node.strength) || 0;
      acc.access += Number(node.accessibility) || 0;
      return acc;
    },
    { activation: 0, strength: 0, access: 0 },
  );
  return {
    activation: totals.activation / count,
    strength: totals.strength / count,
    access: totals.access / count,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatNumber(value) {
  return Number(value || 0).toFixed(2);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
