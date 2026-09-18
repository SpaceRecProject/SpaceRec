(() => {
  "use strict";

  const DEFAULT_CAPTION = "GSM9239732 · Human lymph node 2 · 50 µm DBiTplus";
  const DEFINITIONS = {
    he: { src: "assets/he.png", label: "H&E", legend: [] },
    spacerec: {
      src: "assets/spacerec.png?v=20260917-rctd140-1", label: "SpaceRec",
      legend: [
        ["B cells", "#E41A1C"], ["T cells", "#377EB8"], ["DC", "#984EA3"],
        ["Endo", "#4DAF4A"], ["ILC", "#FF7F00"], ["Macrophages", "#A65628"],
        ["Monocytes", "#F781BF"], ["NK", "#00A6D6"], ["NKT", "#FFD92F"],
        ["VSMC", "#666666"],
      ],
    },
    clusters: {
      src: "assets/clusters.png?v=20260917-all2500-1", label: "Clusters 0–4",
      legend: [
        ["Cluster 0", "#4682B4"], ["Cluster 1", "#FA8072"],
        ["Cluster 2", "#E41A1C"], ["Cluster 3", "#B452CD"],
        ["Cluster 4", "#F0E68C"],
      ],
    },
  };
  const PAIRS = {
    "he-spacerec": ["he", "spacerec"],
    "he-clusters": ["he", "clusters"],
    "spacerec-clusters": ["spacerec", "clusters"],
  };
  const freshLayer = (key) => ({ ...DEFINITIONS[key], key, url: null, width: 0, height: 0 });
  const state = {
    mode: "swipe", pair: "he-spacerec", value: 50, dragging: false,
    layers: { he: freshLayer("he"), spacerec: freshLayer("spacerec"), clusters: freshLayer("clusters") },
  };
  const byId = (id) => document.getElementById(id);
  const elements = {
    stage: byId("comparisonStage"), single: byId("singleView"), side: byId("sideView"),
    leftImage: byId("leftImage"), rightImage: byId("rightImage"),
    sideLeftImage: byId("sideLeftImage"), sideRightImage: byId("sideRightImage"),
    leftLabel: byId("leftLabel"), rightLabel: byId("rightLabel"),
    sideLeftCaption: byId("sideLeftCaption"), sideRightCaption: byId("sideRightCaption"),
    clip: byId("heClip"), divider: byId("divider"), range: byId("comparisonRange"),
    rangeGroup: byId("rangeGroup"), rangeLabel: byId("rangeLabel"), rangeOutput: byId("rangeOutput"),
    status: byId("imageStatus"), warning: byId("imageWarning"), caption: byId("captionInput"),
    legend: byId("legendBand"), heUpload: byId("heUpload"), spaceRecUpload: byId("spaceRecUpload"),
    reset: byId("resetButton"), export: byId("exportButton"),
    modes: [...document.querySelectorAll("[data-mode]")], pairs: [...document.querySelectorAll("[data-pair]")],
  };

  const activeKeys = () => PAIRS[state.pair];
  const activeLayers = () => activeKeys().map((key) => state.layers[key]);
  const clamp = (value) => Math.min(100, Math.max(0, Number(value)));

  function updateValue(value) {
    state.value = clamp(value);
    elements.range.value = String(state.value);
    elements.rangeOutput.value = `${Math.round(state.value)}%`;
    elements.rangeOutput.textContent = `${Math.round(state.value)}%`;
    elements.divider.style.left = `${state.value}%`;
    elements.divider.setAttribute("aria-valuenow", String(Math.round(state.value)));
    if (state.mode === "swipe") elements.clip.style.clipPath = `inset(0 ${100 - state.value}% 0 0)`;
    if (state.mode === "opacity") elements.clip.style.opacity = String(state.value / 100);
  }

  function updateAspect() {
    const left = activeLayers()[0];
    const width = left.width || 2016;
    const height = left.height || 2048;
    elements.stage.style.aspectRatio = state.mode === "side"
      ? (window.matchMedia("(max-width: 520px)").matches ? `${width} / ${height * 2}` : `${width * 2} / ${height}`)
      : `${width} / ${height}`;
  }

  function renderLegend() {
    elements.legend.replaceChildren();
    activeLayers().filter((layer) => layer.legend.length).forEach((layer) => {
      const group = document.createElement("div");
      group.className = "legend-group";
      const title = document.createElement("h2");
      title.textContent = layer.label;
      const list = document.createElement("div");
      list.className = "legend-list";
      layer.legend.forEach(([label, color]) => {
        const item = document.createElement("span");
        item.className = "legend-item";
        const swatch = document.createElement("i");
        swatch.style.backgroundColor = color;
        item.append(swatch, document.createTextNode(label));
        list.append(item);
      });
      group.append(title, list);
      elements.legend.append(group);
    });
    elements.legend.hidden = elements.legend.childElementCount === 0;
  }

  function applyImages() {
    const [left, right] = activeLayers();
    elements.leftImage.src = left.src;
    elements.rightImage.src = right.src;
    elements.sideLeftImage.src = left.src;
    elements.sideRightImage.src = right.src;
    elements.leftLabel.textContent = left.label;
    elements.rightLabel.textContent = right.label;
    elements.sideLeftCaption.textContent = left.label;
    elements.sideRightCaption.textContent = right.label;
    elements.divider.setAttribute("aria-label", `${left.label} reveal position`);
    renderLegend();
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
    const [left, right] = activeLayers();
    if (mode === "opacity") {
      elements.rangeLabel.textContent = `${right.label} opacity`;
      elements.clip.style.clipPath = "none";
      elements.clip.style.opacity = String(state.value / 100);
      elements.leftImage.src = right.src;
      elements.rightImage.src = left.src;
    } else {
      elements.rangeLabel.textContent = "Position";
      elements.clip.style.opacity = "1";
      elements.leftImage.src = left.src;
      elements.rightImage.src = right.src;
      updateValue(state.value);
    }
    updateAspect();
  }

  function setPair(pair) {
    state.pair = pair;
    elements.pairs.forEach((button) => {
      const active = button.dataset.pair === pair;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    applyImages();
    setMode(state.mode);
    updateCompatibility();
  }

  function updateCompatibility() {
    const [left, right] = activeLayers();
    if (!left.width || !right.width) return;
    const ratioDifference = Math.abs(left.width / left.height - right.width / right.height) / (left.width / left.height);
    const dimensionsMatch = left.width === right.width && left.height === right.height;
    elements.status.textContent = `${left.label} ${left.width}×${left.height} · ${right.label} ${right.width}×${right.height}`;
    updateAspect();
    if (!dimensionsMatch || ratioDifference > 0.001) {
      elements.warning.hidden = false;
      elements.warning.textContent = ratioDifference > 0.001
        ? "Aspect ratios differ; images are preserved with contain fitting and cannot be assumed pixel-aligned."
        : "Dimensions differ; verify that both files cover the same coordinate bounds before interpreting the overlay.";
    } else {
      elements.warning.hidden = true;
      elements.warning.textContent = "";
    }
  }

  function loadDimensions(key) {
    const layer = state.layers[key];
    const probe = new Image();
    probe.onload = () => {
      layer.width = probe.naturalWidth;
      layer.height = probe.naturalHeight;
      if (activeKeys().includes(key)) updateCompatibility();
    };
    probe.onerror = () => {
      elements.warning.hidden = false;
      elements.warning.textContent = `Could not load ${layer.label}.`;
    };
    probe.src = layer.src;
  }

  function setImage(key, src, label, objectUrl) {
    if (state.layers[key].url) URL.revokeObjectURL(state.layers[key].url);
    state.layers[key] = { ...state.layers[key], src, label, url: objectUrl, width: 0, height: 0 };
    applyImages();
    setMode(state.mode);
    loadDimensions(key);
  }

  function upload(key, file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      elements.warning.hidden = false;
      elements.warning.textContent = `${file.name} is not a supported image file.`;
      return;
    }
    const url = URL.createObjectURL(file);
    setImage(key, url, file.name, url);
  }

  function reset() {
    Object.keys(DEFINITIONS).forEach((key) => {
      if (state.layers[key].url) URL.revokeObjectURL(state.layers[key].url);
      state.layers[key] = freshLayer(key);
    });
    elements.heUpload.value = "";
    elements.spaceRecUpload.value = "";
    elements.caption.value = DEFAULT_CAPTION;
    state.value = 50;
    setPair("he-spacerec");
    setMode("swipe");
    updateValue(50);
    Object.keys(state.layers).forEach(loadDimensions);
  }

  function loadedImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = src;
    });
  }

  function drawContained(context, image, x, y, width, height) {
    const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
    const drawWidth = image.naturalWidth * scale;
    const drawHeight = image.naturalHeight * scale;
    context.drawImage(image, x + (width - drawWidth) / 2, y + (height - drawHeight) / 2, drawWidth, drawHeight);
  }

  function drawLabel(context, text, x, y, rightAligned = false) {
    context.font = "700 28px Arial, sans-serif";
    const width = context.measureText(text).width + 28;
    const left = rightAligned ? x - width : x;
    context.fillStyle = "rgba(255,255,255,0.92)";
    context.fillRect(left, y, width, 48);
    context.strokeStyle = "rgba(0,0,0,0.35)";
    context.strokeRect(left, y, width, 48);
    context.fillStyle = "#111315";
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(text, left + 14, y + 25);
  }

  function legendHeight(layers, width) {
    const columns = width > 3000 ? 8 : 5;
    return layers.reduce((sum, layer) => sum + 42 + Math.ceil(layer.legend.length / columns) * 38, 0);
  }

  function drawLegends(context, layers, startY, width) {
    const columns = width > 3000 ? 8 : 5;
    const columnWidth = width / columns;
    let y = startY;
    layers.forEach((layer) => {
      context.fillStyle = "#17191c";
      context.font = "700 24px Arial, sans-serif";
      context.textAlign = "left";
      context.fillText(layer.label, 20, y + 22);
      y += 42;
      layer.legend.forEach(([label, color], index) => {
        const x = (index % columns) * columnWidth + 20;
        const itemY = y + Math.floor(index / columns) * 38;
        context.fillStyle = color;
        context.fillRect(x, itemY + 7, 22, 22);
        context.strokeStyle = "rgba(0,0,0,0.35)";
        context.strokeRect(x, itemY + 7, 22, 22);
        context.fillStyle = "#282c30";
        context.font = "500 20px Arial, sans-serif";
        context.fillText(label, x + 32, itemY + 19);
      });
      y += Math.ceil(layer.legend.length / columns) * 38;
    });
  }

  async function exportPng() {
    elements.export.disabled = true;
    elements.export.textContent = "Preparing…";
    try {
      const [leftLayer, rightLayer] = activeLayers();
      const [left, right] = await Promise.all([loadedImage(leftLayer.src), loadedImage(rightLayer.src)]);
      const imageWidth = Math.min(8192, left.naturalWidth);
      const imageHeight = Math.round(imageWidth * left.naturalHeight / left.naturalWidth);
      const header = 88;
      const captionHeight = elements.caption.value.trim() ? 72 : 0;
      const gap = state.mode === "side" ? 24 : 0;
      const canvasWidth = state.mode === "side" ? imageWidth * 2 + gap : imageWidth;
      const legendLayers = [leftLayer, rightLayer].filter((layer) => layer.legend.length);
      const canvas = document.createElement("canvas");
      canvas.width = canvasWidth;
      canvas.height = imageHeight + header + captionHeight + legendHeight(legendLayers, canvasWidth);
      const context = canvas.getContext("2d");
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#17191c";
      context.font = "700 30px Arial, sans-serif";
      context.textBaseline = "middle";
      context.fillText(`${leftLayer.label} / ${rightLayer.label}`, 20, header / 2);
      context.font = "500 24px Arial, sans-serif";
      context.textAlign = "right";
      const modeText = state.mode === "side" ? "Side by side" : state.mode === "opacity" ? `Opacity ${Math.round(state.value)}%` : `Swipe ${Math.round(state.value)}%`;
      context.fillText(modeText, canvas.width - 20, header / 2);

      if (state.mode === "side") {
        drawContained(context, left, 0, header, imageWidth, imageHeight);
        drawContained(context, right, imageWidth + gap, header, imageWidth, imageHeight);
        drawLabel(context, leftLayer.label, 18, header + 18);
        drawLabel(context, rightLayer.label, canvas.width - 18, header + 18, true);
      } else if (state.mode === "opacity") {
        drawContained(context, left, 0, header, imageWidth, imageHeight);
        context.save();
        context.globalAlpha = state.value / 100;
        drawContained(context, right, 0, header, imageWidth, imageHeight);
        context.restore();
        drawLabel(context, `${leftLayer.label} + ${rightLayer.label}`, 18, header + 18);
      } else {
        drawContained(context, right, 0, header, imageWidth, imageHeight);
        const split = Math.round(imageWidth * state.value / 100);
        context.save();
        context.beginPath();
        context.rect(0, header, split, imageHeight);
        context.clip();
        drawContained(context, left, 0, header, imageWidth, imageHeight);
        context.restore();
        context.fillStyle = "#ffffff";
        context.fillRect(split - 2, header, 4, imageHeight);
        drawLabel(context, leftLayer.label, 18, header + 18);
        drawLabel(context, rightLayer.label, canvas.width - 18, header + 18, true);
      }

      let footerY = header + imageHeight;
      if (captionHeight) {
        context.fillStyle = "#17191c";
        context.font = "500 24px Arial, sans-serif";
        context.textAlign = "left";
        context.fillText(elements.caption.value.trim(), 20, footerY + captionHeight / 2);
        footerY += captionHeight;
      }
      drawLegends(context, legendLayers, footerY, canvas.width);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("PNG encoding failed");
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      const sample = (elements.caption.value.trim() || "comparison").replace(/[^a-z0-9]+/gi, "_").replace(/^_|_$/g, "").slice(0, 70);
      anchor.href = url;
      anchor.download = `${sample}_${state.pair}_${state.mode}.png`;
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
  elements.range.addEventListener("input", () => updateValue(elements.range.value));
  elements.single.addEventListener("pointerdown", (event) => {
    if (state.mode !== "swipe") return;
    event.preventDefault();
    state.dragging = true;
    elements.single.setPointerCapture(event.pointerId);
    const rect = elements.single.getBoundingClientRect();
    updateValue(((event.clientX - rect.left) / rect.width) * 100);
  });
  elements.single.addEventListener("pointermove", (event) => {
    if (!state.dragging || state.mode !== "swipe") return;
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
  elements.spaceRecUpload.addEventListener("change", () => upload("spacerec", elements.spaceRecUpload.files[0]));
  elements.reset.addEventListener("click", reset);
  elements.export.addEventListener("click", exportPng);
  window.addEventListener("resize", updateAspect);

  applyImages();
  setMode("swipe");
  updateValue(50);
  Object.keys(state.layers).forEach(loadDimensions);
  window.comparisonApp = { state, setMode, setPair, updateValue, reset, exportPng };
})();
