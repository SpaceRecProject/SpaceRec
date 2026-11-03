(() => {
  "use strict";

  const VERSION = "20261102-cerebellum-1";
  const TYPE_CONFIG = [
    ["other_neurons", "Other neurons", "#E69F00"],
    ["excitatory_neurons", "Excitatory neurons", "#CC79A7"],
    ["inhibitory_neurons", "Inhibitory neurons", "#0072B2"],
    ["astroglia", "Astroglia", "#009E73"],
    ["oligodendroglia", "Oligodendroglia", "#56B4E9"],
    ["immune", "Immune", "#D55E00"],
    ["vascular_stromal", "Vascular / stromal", "#F0E442"],
    ["ventricular_epithelium", "Ventricular epithelium", "#999999"],
  ];
  const SAMPLE_CONFIG = {
    C2: {
      tag: "GSM6320947_37_46_C2",
      grids: 156321,
      counts: [67699, 40686, 18901, 7738, 10042, 1524, 9136, 595],
    },
    C3: {
      tag: "GSM6320948_37_46_C3",
      grids: 178064,
      counts: [68084, 44991, 26052, 9767, 15498, 2443, 10832, 397],
    },
  };
  const asset = (name) => `assets/${name}?v=${VERSION}`;
  const defaultsFor = (sample) => ({
    dapi: { src: asset(`${sample.toLowerCase()}_dapi.png`), label: "DAPI", width: 3200, height: 3200, url: null },
    grid: { src: asset(`${sample.toLowerCase()}_grid_type.png`), label: "Grid type", width: 3200, height: 3200, url: null },
  });
  const captionFor = (sample) => `${SAMPLE_CONFIG[sample].tag} · DAPI · SpaceRec Stage 1 · 30 px grid`;
  const state = {
    sample: "C2",
    mode: "compare",
    value: 50,
    opacity: 85,
    dragging: false,
    layers: defaultsFor("C2"),
  };
  const byId = (id) => document.getElementById(id);
  const elements = {
    stage: byId("comparisonStage"), single: byId("singleView"), side: byId("sideView"),
    dapi: byId("dapiImage"), grid: byId("gridImage"),
    sideDapi: byId("sideDapiImage"), sideGrid: byId("sideGridImage"),
    clip: byId("comparisonClip"), divider: byId("divider"),
    range: byId("comparisonRange"), rangeGroup: byId("rangeGroup"), rangeOutput: byId("rangeOutput"),
    opacity: byId("opacityRange"), opacityGroup: byId("opacityGroup"), opacityOutput: byId("opacityOutput"),
    status: byId("imageStatus"), warning: byId("imageWarning"), caption: byId("captionInput"),
    legend: byId("legendList"), dapiUpload: byId("dapiUpload"), predictionUpload: byId("predictionUpload"),
    reset: byId("resetButton"), export: byId("exportButton"),
    samples: [...document.querySelectorAll("[data-sample]")], modes: [...document.querySelectorAll("[data-mode]")],
  };
  const clamp = (value) => Math.min(100, Math.max(0, Number(value)));

  function updatePosition(value) {
    state.value = clamp(value);
    elements.range.value = String(state.value);
    elements.rangeOutput.value = `${Math.round(state.value)}%`;
    elements.rangeOutput.textContent = `${Math.round(state.value)}%`;
    elements.divider.style.left = `${state.value}%`;
    elements.divider.setAttribute("aria-valuenow", String(Math.round(state.value)));
    elements.clip.style.clipPath = `inset(0 0 0 ${state.value}%)`;
  }

  function updateOpacity(value) {
    state.opacity = clamp(value);
    elements.opacity.value = String(state.opacity);
    elements.opacityOutput.value = `${Math.round(state.opacity)}%`;
    elements.opacityOutput.textContent = `${Math.round(state.opacity)}%`;
    elements.clip.style.opacity = String(state.opacity / 100);
  }

  function renderLegend() {
    elements.legend.replaceChildren();
    const counts = SAMPLE_CONFIG[state.sample].counts;
    TYPE_CONFIG.forEach(([, label, color], index) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      const swatch = document.createElement("i");
      swatch.style.backgroundColor = color;
      item.append(swatch, document.createTextNode(`${label} (${counts[index].toLocaleString()})`));
      elements.legend.append(item);
    });
  }

  function updateCompatibility() {
    const { dapi, grid } = state.layers;
    const match = dapi.width === grid.width && dapi.height === grid.height;
    const ratioDifference = Math.abs(dapi.width / dapi.height - grid.width / grid.height) / (dapi.width / dapi.height);
    elements.status.textContent = `${state.sample} · ${SAMPLE_CONFIG[state.sample].grids.toLocaleString()} grids · ${dapi.label} ${dapi.width}×${dapi.height} · ${grid.label} ${grid.width}×${grid.height}`;
    if (!match || ratioDifference > 0.001) {
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
    elements.dapi.src = state.layers.dapi.src;
    elements.grid.src = state.layers.grid.src;
    elements.sideDapi.src = state.layers.dapi.src;
    elements.sideGrid.src = state.layers.grid.src;
    renderLegend();
    updateCompatibility();
  }

  function setSample(sample) {
    if (!SAMPLE_CONFIG[sample]) return;
    Object.values(state.layers).forEach((layer) => { if (layer.url) URL.revokeObjectURL(layer.url); });
    state.sample = sample;
    state.layers = defaultsFor(sample);
    elements.samples.forEach((button) => {
      const active = button.dataset.sample === sample;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    elements.dapiUpload.value = "";
    elements.predictionUpload.value = "";
    elements.caption.value = captionFor(sample);
    applyImages();
  }

  function setMode(mode) {
    state.mode = mode;
    elements.modes.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const side = mode === "side";
    elements.stage.className = `comparison-stage mode-${mode}`;
    elements.single.hidden = side;
    elements.side.hidden = !side;
    elements.rangeGroup.hidden = side;
    elements.opacityGroup.hidden = side;
    elements.stage.style.aspectRatio = side && window.matchMedia("(max-width: 520px)").matches ? "1 / 2" : side ? "2 / 1" : "1 / 1";
    updatePosition(state.value);
    updateOpacity(state.opacity);
  }

  function loadDimensions(key) {
    const layer = state.layers[key];
    const image = new Image();
    image.onload = () => {
      layer.width = image.naturalWidth;
      layer.height = image.naturalHeight;
      updateCompatibility();
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
    setSample(state.sample);
    setMode("compare");
    updatePosition(50);
    updateOpacity(85);
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
      const [dapi, grid] = await Promise.all([loadedImage(state.layers.dapi.src), loadedImage(state.layers.grid.src)]);
      const side = state.mode === "side";
      const gap = side ? 24 : 0;
      const canvas = document.createElement("canvas");
      canvas.width = side ? dapi.naturalWidth + grid.naturalWidth + gap : dapi.naturalWidth;
      canvas.height = Math.max(dapi.naturalHeight, grid.naturalHeight);
      const context = canvas.getContext("2d");
      context.fillStyle = "#02050b";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(dapi, 0, 0);
      if (side) {
        context.drawImage(grid, dapi.naturalWidth + gap, 0);
      } else {
        const split = Math.round(canvas.width * state.value / 100);
        context.save();
        context.beginPath();
        context.rect(split, 0, canvas.width - split, canvas.height);
        context.clip();
        context.globalAlpha = state.opacity / 100;
        context.drawImage(grid, 0, 0);
        context.restore();
        context.fillStyle = "#ffffff";
        context.fillRect(split - 1, 0, 2, canvas.height);
      }
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("PNG encoding failed");
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `cerebellum_${state.sample}_${state.mode}.png`;
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

  elements.samples.forEach((button) => button.addEventListener("click", () => setSample(button.dataset.sample)));
  elements.modes.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  elements.range.addEventListener("input", () => updatePosition(elements.range.value));
  elements.opacity.addEventListener("input", () => updateOpacity(elements.opacity.value));
  elements.single.addEventListener("pointerdown", (event) => {
    if (state.mode !== "compare") return;
    event.preventDefault();
    state.dragging = true;
    elements.single.setPointerCapture(event.pointerId);
    const rect = elements.single.getBoundingClientRect();
    updatePosition(((event.clientX - rect.left) / rect.width) * 100);
  });
  elements.single.addEventListener("pointermove", (event) => {
    if (!state.dragging || state.mode !== "compare") return;
    const rect = elements.single.getBoundingClientRect();
    updatePosition(((event.clientX - rect.left) / rect.width) * 100);
  });
  elements.single.addEventListener("pointerup", () => { state.dragging = false; });
  elements.single.addEventListener("pointercancel", () => { state.dragging = false; });
  elements.divider.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const step = event.shiftKey ? 5 : 1;
    if (event.key === "Home") updatePosition(0);
    else if (event.key === "End") updatePosition(100);
    else updatePosition(state.value + (event.key === "ArrowRight" ? step : -step));
  });
  elements.dapiUpload.addEventListener("change", () => upload("dapi", elements.dapiUpload.files[0]));
  elements.predictionUpload.addEventListener("change", () => upload("grid", elements.predictionUpload.files[0]));
  elements.reset.addEventListener("click", reset);
  elements.export.addEventListener("click", exportPng);
  window.addEventListener("resize", () => setMode(state.mode));

  elements.caption.value = captionFor("C2");
  applyImages();
  setMode("compare");
  updatePosition(50);
  updateOpacity(85);
  window.comparisonApp = { state, setSample, setMode, updatePosition, updateOpacity, reset, exportPng };
})();
