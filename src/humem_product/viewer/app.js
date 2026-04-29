import * as THREE from "./vendor/three.module.min.js";

const canvas = document.querySelector("#scene");
const statsEl = document.querySelector("#stats");
const searchEl = document.querySelector("#search");
const resetEl = document.querySelector("#resetView");
const layersEl = document.querySelector("#layerToggles");
const detailsEl = document.querySelector("#details");
const tooltipEl = document.querySelector("#tooltip");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070807);
scene.fog = new THREE.FogExp2(0x070807, 0.035);

const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 1000);
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const clock = new THREE.Clock();

const graphGroup = new THREE.Group();
const layerGroup = new THREE.Group();
scene.add(layerGroup, graphGroup);

scene.add(new THREE.HemisphereLight(0xf7f0da, 0x263134, 1.45));
const keyLight = new THREE.DirectionalLight(0xffe0aa, 1.8);
keyLight.position.set(8, 12, 10);
scene.add(keyLight);
const rimLight = new THREE.PointLight(0x66e6cf, 4.2, 42);
rimLight.position.set(-8, 6, -8);
scene.add(rimLight);

let graph = { nodes: [], links: [], meta: { totalLayers: 1, layerHistogram: [] } };
let nodeById = new Map();
let adjacency = new Map();
let visibleLayers = new Set();
let selectedId = null;
let hoverId = null;
let searchTerm = "";
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
  nodeById = new Map();
  adjacency = new Map();

  const totalLayers = Math.max(graph.meta.totalLayers || 1, 1);
  const layerGap = 1.55;
  const spread = Math.max(7, Math.sqrt(Math.max(graph.nodes.length, 1)) * 1.45);
  const topColor = new THREE.Color(0xffca70);
  const bottomColor = new THREE.Color(0x6c7cff);

  for (let layer = 0; layer < totalLayers; layer += 1) {
    const y = layerToY(layer, totalLayers, layerGap);
    const t = totalLayers === 1 ? 0 : layer / (totalLayers - 1);
    const color = new THREE.Color().lerpColors(topColor, bottomColor, t);
    const grid = new THREE.GridHelper(spread * 2.7, 12, color, color);
    grid.position.y = y;
    grid.material.transparent = true;
    grid.material.opacity = 0.16 - t * 0.07;
    layerGroup.add(grid);
  }

  for (const node of graph.nodes) {
    const strength = clamp(Number(node.strength) || 0, 0, 4);
    const activation = clamp(Number(node.activation) || 0, 0, 3);
    const depth = Number(node.depth ?? node.layer) || 0;
    const layerRatio = totalLayers === 1 ? 0 : depth / (totalLayers - 1);
    const color = new THREE.Color().lerpColors(topColor, bottomColor, layerRatio);
    const radius = 0.13 + Math.sqrt(strength + 0.25) * 0.072;
    const geometry = new THREE.SphereGeometry(radius, 24, 18);
    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.42,
      metalness: 0.12,
      emissive: color,
      emissiveIntensity: 0.18 + activation * 0.13,
      transparent: true,
      opacity: 0.94,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(nodeToPosition(node, totalLayers, layerGap, spread));
    mesh.userData.nodeId = node.id;
    graphGroup.add(mesh);
    nodeById.set(node.id, { data: node, mesh, material, baseRadius: radius, related: [] });
    adjacency.set(node.id, []);
  }

  for (const link of graph.links) {
    const source = nodeById.get(link.source);
    const target = nodeById.get(link.target);
    if (!source || !target) {
      continue;
    }

    const geometry = new THREE.BufferGeometry().setFromPoints([
      source.mesh.position,
      target.mesh.position,
    ]);
    const color = link.crossLayer ? 0xffb86c : 0x66e6cf;
    const material = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.18 + clamp(Number(link.weight) || 0, 0, 1) * 0.36,
    });
    const line = new THREE.Line(geometry, material);
    line.userData.linkId = link.id;
    line.userData.sourceId = link.source;
    line.userData.targetId = link.target;
    graphGroup.add(line);
    adjacency.get(link.source).push({ link, other: link.target, line, material });
    adjacency.get(link.target).push({ link, other: link.source, line, material });
  }

  updateVisualState();
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
    label.textContent = `Layer ${layer}`;

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
  statsEl.textContent = `${meta.fragmentCount} memories · ${meta.relationCount} relations · ${meta.totalLayers} layers · ${layout} · ${scope}`;
}

function animate() {
  requestAnimationFrame(animate);
  const delta = Math.min(clock.getDelta(), 0.05);
  smoothCamera(delta);
  const time = clock.elapsedTime;

  for (const entry of nodeById.values()) {
    const isSelected = entry.data.id === selectedId;
    const isHover = entry.data.id === hoverId;
    const pulse = 1 + Math.sin(time * 2.2 + entry.data.layer) * 0.025;
    const scale = (isSelected ? 1.82 : isHover ? 1.42 : 1) * pulse;
    entry.mesh.scale.setScalar(scale);
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
  const related = new Set();
  if (selectedId) {
    related.add(selectedId);
    for (const edge of adjacency.get(selectedId) || []) {
      related.add(edge.other);
    }
  }

  for (const [id, entry] of nodeById.entries()) {
    const layerVisible = visibleLayers.has(entry.data.layer);
    const matchesSearch = !searchTerm || entry.data.text.toLowerCase().includes(searchTerm);
    const inFocus = !selectedId || related.has(id);
    entry.mesh.visible = layerVisible;
    entry.material.opacity = layerVisible ? (matchesSearch && inFocus ? 0.96 : 0.2) : 0;
    entry.material.emissiveIntensity = matchesSearch && inFocus ? 0.28 + entry.data.activation * 0.13 : 0.04;
  }

  for (const edges of adjacency.values()) {
    for (const edge of edges) {
      const source = nodeById.get(edge.link.source);
      const target = nodeById.get(edge.link.target);
      const visible = source?.mesh.visible && target?.mesh.visible;
      const highlighted = selectedId && (edge.link.source === selectedId || edge.link.target === selectedId);
      edge.line.visible = Boolean(visible);
      edge.material.opacity = highlighted ? 0.82 : selectedId ? 0.06 : 0.22 + edge.link.weight * 0.25;
    }
  }
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

  detailsEl.innerHTML = `
    <div class="detail-content">
      <span class="kind">${escapeHtml(node.kind)}</span>
      <div class="detail-title">${escapeHtml(node.text)}</div>
      <div class="metric-grid">
        <div class="metric"><span class="metric-label">Layer</span><span class="metric-value">${node.layer}</span></div>
        <div class="metric"><span class="metric-label">Depth</span><span class="metric-value">${formatNumber(node.depth)}</span></div>
        <div class="metric"><span class="metric-label">Retrievals</span><span class="metric-value">${node.retrievals || 0}</span></div>
        <div class="metric"><span class="metric-label">Activation</span><span class="metric-value">${formatNumber(node.activation)}</span></div>
        <div class="metric"><span class="metric-label">Strength</span><span class="metric-value">${formatNumber(node.strength)}</span></div>
        <div class="metric"><span class="metric-label">Access</span><span class="metric-value">${formatNumber(node.accessibility)}</span></div>
        <div class="metric"><span class="metric-label">Layout</span><span class="metric-value">${escapeHtml(node.layoutModel || "hash")}</span></div>
        <div class="metric"><span class="metric-label">Scope</span><span class="metric-value">${escapeHtml(node.embeddingScope || "none")}</span></div>
        <div class="metric"><span class="metric-label">Edges</span><span class="metric-value">${node.semanticEdgeCount || 0}/${node.relationEdgeCount || 0}</span></div>
      </div>
      ${renderSource(node)}
      ${renderChunk(node)}
      ${renderLayout(node)}
      <div class="section">
        <h2>Related Memories</h2>
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
  tooltipEl.innerHTML = `${escapeHtml(node.text)}<br><span style="color:#aeb7b4">layer ${node.layer} · depth ${formatNumber(node.depth)} · ${escapeHtml(node.kind)}</span>`;
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

resetEl.addEventListener("click", () => {
  selectedId = null;
  hoverId = null;
  searchTerm = "";
  searchEl.value = "";
  cameraState.desiredTheta = -0.78;
  cameraState.desiredPhi = 1.06;
  cameraState.desiredRadius = 19;
  cameraState.desiredTarget.set(0, 2.4, 0);
  renderDetails(null);
  updateVisualState();
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

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
  const top = new THREE.Color(0xffca70);
  const bottom = new THREE.Color(0x6c7cff);
  const ratio = totalLayers <= 1 ? 0 : layer / (totalLayers - 1);
  return `#${new THREE.Color().lerpColors(top, bottom, ratio).getHexString()}`;
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
