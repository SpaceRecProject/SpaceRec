(() => {
  "use strict";

  const DEFAULTS = {
    he: { src: "assets/he.png?v=20260924-owkin-grid-1", label: "H&E", width: 3000, height: 3000, url: null },
    prediction: { src: "assets/grid_type_on_he.png?v=20260924-owkin-grid-1", label: "Grid type prediction", width: 3000, height: 3000, url: null },
  };
  const DEFAULT_CAPTION = "CAVG10673 · Owkin Visium · SpaceRec Stage 1 · 3 px grid";
  const cloneLayer = (key) => ({ ...DEFAULTS[key] });
  const state = {
    mode: "compare",
    value: 50,
    opacity: 100,
    dragging: false,
    layers: { he: cloneLayer("he"), prediction: cloneLayer("prediction") },
  };
  const byId = (id) => document.getElementById(id);
  const elements = {
    stage: byId("comparisonStage"),
    single: byId("singleView"),
    side: byId("sideView"),
    leftImage: byId("leftImage"),
    rightImage: byId("rightImage"),
    sideLeftImage: byId("sideLeftImage"),
    sideRightImage: byId("sideRightImage"),
    clip: byId("comparisonClip"),
    divider: byId("divider"),
    range: byId("comparisonRange"),
    rangeGroup: byId("rangeGroup"),
    rangeLabel: byId("rangeLabel"),
    rangeOutput: byId("rangeOutput"),
    opacityGroup: byId("opacityGroup"),
    opacityRange: byId("opacityRange"),
    opacityOutput: byId("opacityOutput"),
    status: byId("imageStatus"),
    warning: byId("imageWarning"),
    caption: byId("captionInput"),
    heUpload: byId("heUpload"),
    predictionUpload: byId("predictionUpload"),
    reset: byId("resetButton"),
    export: byId("exportButton"),
    modes: [...document.querySelectorAll("[data-mode]")],
  };
  const clamp = (value) => Math.min(100, Math.max(0, Number(value)));

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

  function updateCompatibility() {
    const { he, prediction } = state.layers;
    const dimensionsMatch = he.width === prediction.width && he.height === prediction.height;
    const heRatio = he.width / he.height;
    const predictionRatio = prediction.width / prediction.height;
    const ratioDifference = Math.abs(heRatio - predictionRatio) / heRatio;
    elements.status.textContent = `${he.label} ${he.width}×${he.height} · ${prediction.label} ${prediction.width}×${prediction.height}`;
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
    const { he, prediction } = state.layers;
    elements.rightImage.src = he.src;
    elements.leftImage.src = prediction.src;
    elements.sideLeftImage.src = he.src;
    elements.sideRightImage.src = prediction.src;
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
    elements.rangeLabel.textContent = "Position";
    elements.clip.style.clipPath = `inset(0 0 0 ${state.value}%)`;
    elements.clip.style.opacity = String(state.opacity / 100);
    applyImages();
    updateValue(state.value);
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
    Object.values(state.layers).forEach((layer) => { if (layer.url) URL.revokeObjectURL(layer.url); });
    state.layers = { he: cloneLayer("he"), prediction: cloneLayer("prediction") };
    elements.heUpload.value = "";
    elements.predictionUpload.value = "";
    elements.caption.value = DEFAULT_CAPTION;
    state.opacity = 100;
    setMode("compare");
    updateValue(50);
    updateOpacity(100);
    updateCompatibility();
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
      const [he, prediction] = await Promise.all([
        loadedImage(state.layers.he.src),
        loadedImage(state.layers.prediction.src),
      ]);
      const side = state.mode === "side";
      const gap = side ? 24 : 0;
      const canvas = document.createElement("canvas");
      canvas.width = side ? he.naturalWidth + prediction.naturalWidth + gap : he.naturalWidth;
      canvas.height = Math.max(he.naturalHeight, prediction.naturalHeight);
      const context = canvas.getContext("2d");
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      if (side) {
        context.drawImage(he, 0, 0);
        context.drawImage(prediction, he.naturalWidth + gap, 0);
      } else {
        context.drawImage(he, 0, 0);
        const split = Math.round(canvas.width * state.value / 100);
        context.save();
        context.beginPath();
        context.rect(split, 0, canvas.width - split, canvas.height);
        context.clip();
        context.globalAlpha = state.opacity / 100;
        context.drawImage(prediction, 0, 0);
        context.restore();
        context.fillStyle = "#ffffff";
        context.fillRect(split - 1, 0, 2, canvas.height);
      }
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("PNG encoding failed");
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `Owkin_HE_grid_type_${state.mode}.png`;
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
  elements.predictionUpload.addEventListener("change", () => upload("prediction", elements.predictionUpload.files[0]));
  elements.reset.addEventListener("click", reset);
  elements.export.addEventListener("click", exportPng);
  window.addEventListener("resize", () => setMode(state.mode));

  applyImages();
  setMode("compare");
  updateValue(50);
  updateOpacity(100);
  updateCompatibility();
  window.comparisonApp = { state, setMode, updateValue, updateOpacity, reset, exportPng };
})();
