(() => {
  "use strict";

  const VERSION = "20260924-owkin-grid-4";
  const TYPE_CONFIG = {
    "Cancer Epithelial": { slug: "Cancer_Epithelial", label: "Cancer Epithelial", color: "#E41A1C", count: 126164, genes: ["CEACAM6", "KRT19"] },
    "Normal Epithelial": { slug: "Normal_Epithelial", label: "Normal Epithelial", color: "#4DAF4A", count: 4533, genes: ["KRT5", "KRT15"] },
    "T cells": { slug: "T_cells", label: "T cells", color: "#FF7F00", count: 721, genes: ["CD3D", "TRAC"] },
    B_Plasma: { slug: "B_Plasma", label: "B / Plasma", color: "#377EB8", count: 269, genes: ["MS4A1", "JCHAIN"] },
    NK: { slug: "NK", label: "NK", color: "#984EA3", count: 110, genes: ["NKG7", "GNLY"] },
    Myeloid: { slug: "Myeloid", label: "Myeloid", color: "#A65628", count: 20493, genes: ["C1QA", "CD163"] },
    Dendritic: { slug: "Dendritic", label: "Dendritic", color: "#F781BF", count: 863, genes: ["CD1C", "FCER1A"] },
    Fibroblast_PVL: { slug: "Fibroblast_PVL", label: "Fibroblast / PVL", color: "#B3A400", count: 54412, genes: ["COL1A1", "RGS5"] },
    Endothelial: { slug: "Endothelial", label: "Endothelial", color: "#00A9CF", count: 4162, genes: ["VWF", "CLDN5"] },
  };
  const DEFAULT_CAPTION = "CAVG10673 · Owkin Visium · SpaceRec Stage 1/2 · 3 px grid";
  const asset = (name) => `assets/${name}?v=${VERSION}`;
  const expressionAsset = (name) => asset(`type_expression/${name}`);
  const DEFINITIONS = {
    he: { src: asset("he.png"), label: "H&E", width: 3000, height: 3000, url: null },
    gridAll: { src: asset("grid_type.png"), label: "Grid type prediction", width: 3000, height: 3000, url: null },
  };
  Object.entries(TYPE_CONFIG).forEach(([type, config]) => {
    DEFINITIONS[`type:${type}`] = {
      src: expressionAsset(`type_${config.slug}.png`), label: `${config.label} grids`, width: 3000, height: 3000, url: null,
    };
    config.genes.forEach((gene) => {
      DEFINITIONS[`gene:${gene}`] = {
        src: expressionAsset(`gene_${config.slug}_${gene}.png`), label: `${gene} expression`, width: 3000, height: 3000, url: null,
      };
    });
  });

  const freshLayer = (key) => ({ ...DEFINITIONS[key] });
  const state = {
    pair: "he-type",
    mode: "compare",
    value: 33,
    opacity: 100,
    dragging: false,
    selectedType: "Cancer Epithelial",
    selectedGenes: Object.fromEntries(Object.entries(TYPE_CONFIG).map(([type, config]) => [type, config.genes[0]])),
    layers: Object.fromEntries(Object.keys(DEFINITIONS).map((key) => [key, freshLayer(key)])),
  };
  const byId = (id) => document.getElementById(id);
  const elements = {
    stage: byId("comparisonStage"), single: byId("singleView"), side: byId("sideView"),
    leftImage: byId("leftImage"), rightImage: byId("rightImage"),
    sideLeftImage: byId("sideLeftImage"), sideRightImage: byId("sideRightImage"),
    sideLeftCaption: byId("sideLeftCaption"), sideRightCaption: byId("sideRightCaption"),
    clip: byId("comparisonClip"), divider: byId("divider"),
    range: byId("comparisonRange"), rangeGroup: byId("rangeGroup"), rangeOutput: byId("rangeOutput"),
    opacityGroup: byId("opacityGroup"), opacityLabel: byId("opacityLabel"), opacityRange: byId("opacityRange"), opacityOutput: byId("opacityOutput"),
    status: byId("imageStatus"), warning: byId("imageWarning"), caption: byId("captionInput"),
    legend: byId("legendBand"), fileGroup: byId("fileGroup"),
    heUpload: byId("heUpload"), predictionUpload: byId("predictionUpload"),
    reset: byId("resetButton"), export: byId("exportButton"),
    modes: [...document.querySelectorAll("[data-mode]")],
    pairs: [...document.querySelectorAll("[data-pair]")],
    typeGroup: byId("typeGroup"), types: [...document.querySelectorAll("[data-type]")],
    markerGroup: byId("markerGroup"), markerButtons: byId("markerButtons"),
  };
  const clamp = (value) => Math.min(100, Math.max(0, Number(value)));
  const activeKeys = () => state.pair === "he-type"
    ? ["he", "gridAll"]
    : [`type:${state.selectedType}`, `gene:${state.selectedGenes[state.selectedType]}`];
  const activeLayers = () => activeKeys().map((key) => state.layers[key]);

  function updateValue(value) {
    state.value = clamp(value);
    elements.range.value = String(state.value);
    elements.rangeOutput.value = `${Math.round(state.value)}%`;
    elements.rangeOutput.textContent = `${Math.round(state.value)}%`;
    elements.divider.style.left = `${state.value}%`;
    elements.divider.setAttribute("aria-valuenow", String(Math.round(state.value)));
    if (state.mode === "compare") elements.clip.style.clipPath = `inset(0 0 0 ${state.value}%)`;
  }

  function updateOpacity(value) {
    state.opacity = clamp(value);
    elements.opacityRange.value = String(state.opacity);
    elements.opacityOutput.value = `${Math.round(state.opacity)}%`;
    elements.opacityOutput.textContent = `${Math.round(state.opacity)}%`;
    elements.clip.style.opacity = String(state.opacity / 100);
  }

  function renderLegend() {
    elements.legend.replaceChildren();
    const groups = [];
    if (state.pair === "he-type") {
      groups.push({
        title: "Grid type",
        items: Object.values(TYPE_CONFIG).map((config) => [`${config.label} (${config.count.toLocaleString()})`, config.color]),
      });
    } else {
      const config = TYPE_CONFIG[state.selectedType];
      const gene = state.selectedGenes[state.selectedType];
      groups.push({ title: "Selected type", items: [[`${config.label} (${config.count.toLocaleString()})`, config.color]] });
      groups.push({ title: `${gene} · z-score`, items: [["Low (−2)", "#000004"], ["High (+2)", "#FCFDBF"]] });
    }
    groups.forEach(({ title, items }) => {
      const group = document.createElement("div");
      group.className = "legend-group";
      const heading = document.createElement("h2");
      heading.textContent = title;
      const list = document.createElement("div");
      list.className = "legend-list";
      items.forEach(([label, color]) => {
        const item = document.createElement("span");
        item.className = "legend-item";
        const swatch = document.createElement("i");
        swatch.style.backgroundColor = color;
        item.append(swatch, document.createTextNode(label));
        list.append(item);
      });
      group.append(heading, list);
      elements.legend.append(group);
    });
  }

  function renderSelectionControls() {
    const show = state.pair === "type-expression";
    elements.typeGroup.hidden = !show;
    elements.markerGroup.hidden = !show;
    elements.fileGroup.hidden = show;
    elements.opacityLabel.textContent = show ? "Grid expression opacity" : "Grid type opacity";
    elements.types.forEach((button) => {
      const active = button.dataset.type === state.selectedType;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    elements.markerButtons.replaceChildren();
    if (!show) return;
    TYPE_CONFIG[state.selectedType].genes.forEach((gene) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `segment${state.selectedGenes[state.selectedType] === gene ? " active" : ""}`;
      button.textContent = gene;
      button.setAttribute("aria-pressed", String(state.selectedGenes[state.selectedType] === gene));
      button.addEventListener("click", () => setGene(gene));
      elements.markerButtons.append(button);
    });
  }

  function updateCompatibility() {
    const [left, right] = activeLayers();
    const dimensionsMatch = left.width === right.width && left.height === right.height;
    const ratioDifference = Math.abs(left.width / left.height - right.width / right.height) / (left.width / left.height);
    elements.status.textContent = `${left.label} ${left.width}×${left.height} · ${right.label} ${right.width}×${right.height}`;
    if (!dimensionsMatch || ratioDifference > 0.001) {
      elements.warning.hidden = false;
      elements.warning.textContent = ratioDifference > 0.001
        ? "Aspect ratios differ; these images cannot be assumed pixel-aligned."
        : "Dimensions differ; verify that both images use the same coordinate bounds.";
    } else {
      elements.warning.hidden = true;
      elements.warning.textContent = "";
    }
  }

  function applyImages() {
    const [left, right] = activeLayers();
    elements.rightImage.src = left.src;
    elements.leftImage.src = right.src;
    elements.sideLeftImage.src = left.src;
    elements.sideRightImage.src = right.src;
    elements.sideLeftCaption.textContent = left.label;
    elements.sideRightCaption.textContent = right.label;
    renderLegend();
    updateCompatibility();
  }

  function setMode(mode) {
    state.mode = mode;
    elements.modes.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    elements.stage.className = `comparison-stage mode-${mode}`;
    const side = mode === "side";
    elements.single.hidden = side;
    elements.side.hidden = !side;
    elements.rangeGroup.hidden = side;
    elements.opacityGroup.hidden = side;
    elements.stage.style.aspectRatio = side
      ? (window.matchMedia("(max-width: 520px)").matches ? "1 / 2" : "2 / 1")
      : "1 / 1";
    elements.clip.style.clipPath = `inset(0 0 0 ${state.value}%)`;
    elements.clip.style.opacity = String(state.opacity / 100);
    updateValue(state.value);
    updateOpacity(state.opacity);
  }

  function setPair(pair) {
    state.pair = pair;
    elements.pairs.forEach((button) => {
      const active = button.dataset.pair === pair;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    renderSelectionControls();
    applyImages();
    setMode(state.mode);
  }

  function setType(type) {
    if (!TYPE_CONFIG[type]) return;
    state.selectedType = type;
    renderSelectionControls();
    applyImages();
    setMode(state.mode);
  }

  function setGene(gene) {
    if (!TYPE_CONFIG[state.selectedType].genes.includes(gene)) return;
    state.selectedGenes[state.selectedType] = gene;
    renderSelectionControls();
    applyImages();
    setMode(state.mode);
  }

  function loadDimensions(key) {
    const layer = state.layers[key];
    const image = new Image();
    image.onload = () => {
      layer.width = image.naturalWidth;
      layer.height = image.naturalHeight;
      if (activeKeys().includes(key)) updateCompatibility();
    };
    image.onerror = () => {
      elements.warning.hidden = false;
      elements.warning.textContent = `Could not load ${layer.label}.`;
    };
    image.src = layer.src;
  }

  function upload(key, file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      elements.warning.hidden = false;
      elements.warning.textContent = `${file.name} is not a supported image file.`;
      return;
    }
    if (state.layers[key].url) URL.revokeObjectURL(state.layers[key].url);
    const url = URL.createObjectURL(file);
    state.layers[key] = { src: url, label: file.name, width: 0, height: 0, url };
    applyImages();
    loadDimensions(key);
  }

  function reset() {
    Object.values(state.layers).forEach((layer) => { if (layer.url) URL.revokeObjectURL(layer.url); });
    state.layers = Object.fromEntries(Object.keys(DEFINITIONS).map((key) => [key, freshLayer(key)]));
    state.selectedType = "Cancer Epithelial";
    state.selectedGenes = Object.fromEntries(Object.entries(TYPE_CONFIG).map(([type, config]) => [type, config.genes[0]]));
    state.opacity = 100;
    elements.heUpload.value = "";
    elements.predictionUpload.value = "";
    elements.caption.value = DEFAULT_CAPTION;
    setPair("he-type");
    setMode("compare");
    updateValue(33);
    updateOpacity(100);
  }

  function loadedImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = src;
    });
  }

  async function exportPng() {
    elements.export.disabled = true;
    elements.export.textContent = "Preparing…";
    try {
      const [leftLayer, rightLayer] = activeLayers();
      const [left, right] = await Promise.all([loadedImage(leftLayer.src), loadedImage(rightLayer.src)]);
      const side = state.mode === "side";
      const gap = side ? 24 : 0;
      const canvas = document.createElement("canvas");
      canvas.width = side ? left.naturalWidth + right.naturalWidth + gap : left.naturalWidth;
      canvas.height = Math.max(left.naturalHeight, right.naturalHeight);
      const context = canvas.getContext("2d");
      context.fillStyle = "#000000";
      context.fillRect(0, 0, canvas.width, canvas.height);
      if (side) {
        context.drawImage(left, 0, 0);
        context.drawImage(right, left.naturalWidth + gap, 0);
      } else {
        context.drawImage(left, 0, 0);
        const split = Math.round(canvas.width * state.value / 100);
        context.save();
        context.beginPath();
        context.rect(split, 0, canvas.width - split, canvas.height);
        context.clip();
        context.globalAlpha = state.opacity / 100;
        context.drawImage(right, 0, 0);
        context.restore();
        context.fillStyle = "#ffffff";
        context.fillRect(split - 1, 0, 2, canvas.height);
      }
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("PNG encoding failed");
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `Owkin_${state.pair}_${state.mode}.png`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      elements.warning.hidden = false;
      elements.warning.textContent = `Export failed: ${error.message}`;
    } finally {
      elements.export.disabled = false;
      elements.export.textContent = "Export PNG";
    }
  }

  elements.modes.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  elements.pairs.forEach((button) => button.addEventListener("click", () => setPair(button.dataset.pair)));
  elements.types.forEach((button) => button.addEventListener("click", () => setType(button.dataset.type)));
  elements.range.addEventListener("input", () => updateValue(elements.range.value));
  elements.opacityRange.addEventListener("input", () => updateOpacity(elements.opacityRange.value));
  elements.single.addEventListener("pointerdown", (event) => {
    if (state.mode !== "compare") return;
    event.preventDefault();
    state.dragging = true;
    elements.single.setPointerCapture(event.pointerId);
    const rect = elements.single.getBoundingClientRect();
    updateValue(((event.clientX - rect.left) / rect.width) * 100);
  });
  elements.single.addEventListener("pointermove", (event) => {
    if (!state.dragging || state.mode !== "compare") return;
    const rect = elements.single.getBoundingClientRect();
    updateValue(((event.clientX - rect.left) / rect.width) * 100);
  });
  elements.single.addEventListener("pointerup", () => { state.dragging = false; });
  elements.single.addEventListener("pointercancel", () => { state.dragging = false; });
  elements.divider.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const step = event.shiftKey ? 5 : 1;
    if (event.key === "Home") updateValue(0);
    else if (event.key === "End") updateValue(100);
    else updateValue(state.value + (event.key === "ArrowRight" ? step : -step));
  });
  elements.heUpload.addEventListener("change", () => upload("he", elements.heUpload.files[0]));
  elements.predictionUpload.addEventListener("change", () => upload("gridAll", elements.predictionUpload.files[0]));
  elements.reset.addEventListener("click", reset);
  elements.export.addEventListener("click", exportPng);
  window.addEventListener("resize", () => setMode(state.mode));

  renderSelectionControls();
  applyImages();
  setMode("compare");
  updateValue(33);
  updateOpacity(100);
  window.comparisonApp = { state, setMode, setPair, setType, setGene, updateValue, updateOpacity, reset, exportPng };
})();
