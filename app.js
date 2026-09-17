(() => {
  "use strict";

  const DEFAULTS = {
    he: "assets/he.png",
    spacerec: "assets/spacerec.png",
    caption: "GSM9239732 · Human lymph node 2 · 50 µm DBiTplus",
  };

  const state = {
    mode: "swipe",
    value: 50,
    dragging: false,
    he: { src: DEFAULTS.he, name: "Registered H&E", url: null, width: 0, height: 0 },
    spacerec: { src: DEFAULTS.spacerec, name: "SpaceRec Stage 1", url: null, width: 0, height: 0 },
  };

  const elements = {
    stage: document.getElementById("comparisonStage"),
    singleView: document.getElementById("singleView"),
    sideView: document.getElementById("sideView"),
    heImage: document.getElementById("heImage"),
    spaceRecImage: document.getElementById("spaceRecImage"),
    sideHeImage: document.getElementById("sideHeImage"),
    sideSpaceRecImage: document.getElementById("sideSpaceRecImage"),
    heClip: document.getElementById("heClip"),
    divider: document.getElementById("divider"),
    range: document.getElementById("comparisonRange"),
    rangeGroup: document.getElementById("rangeGroup"),
    rangeLabel: document.getElementById("rangeLabel"),
    rangeOutput: document.getElementById("rangeOutput"),
    status: document.getElementById("imageStatus"),
    warning: document.getElementById("imageWarning"),
    caption: document.getElementById("captionInput"),
    heUpload: document.getElementById("heUpload"),
    spaceRecUpload: document.getElementById("spaceRecUpload"),
    reset: document.getElementById("resetButton"),
    export: document.getElementById("exportButton"),
    modeButtons: [...document.querySelectorAll("[data-mode]")],
  };

  function clamp(value, min = 0, max = 100) {
    return Math.min(max, Math.max(min, Number(value)));
  }

  function updateValue(value) {
    state.value = clamp(value);
    elements.range.value = String(state.value);
    elements.rangeOutput.value = `${Math.round(state.value)}%`;
    elements.rangeOutput.textContent = `${Math.round(state.value)}%`;
    elements.divider.style.left = `${state.value}%`;
    elements.divider.setAttribute("aria-valuenow", String(Math.round(state.value)));
    if (state.mode === "swipe") {
      elements.heClip.style.clipPath = `inset(0 ${100 - state.value}% 0 0)`;
    } else if (state.mode === "opacity") {
      elements.heClip.style.opacity = String(state.value / 100);
    }
  }

  function updateStageAspect() {
    const width = state.he.width || 2016;
    const height = state.he.height || 2048;
    if (state.mode === "side") {
      elements.stage.style.aspectRatio = window.matchMedia("(max-width: 520px)").matches
        ? `${width} / ${height * 2}`
        : `${width * 2} / ${height}`;
    } else {
      elements.stage.style.aspectRatio = `${width} / ${height}`;
    }
  }

  function setMode(mode) {
    state.mode = mode;
    elements.modeButtons.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    elements.stage.className = `comparison-stage mode-${mode}`;
    const side = mode === "side";
    elements.singleView.hidden = side;
    elements.sideView.hidden = !side;
    elements.rangeGroup.hidden = side;
    updateStageAspect();
    if (mode === "opacity") {
      elements.rangeLabel.textContent = "SpaceRec opacity";
      elements.heClip.style.clipPath = "none";
      elements.heClip.style.opacity = String(state.value / 100);
      elements.heClip.querySelector("img").src = state.spacerec.src;
      elements.spaceRecImage.src = state.he.src;
    } else {
      elements.rangeLabel.textContent = "Position";
      elements.heClip.style.opacity = "1";
      elements.heClip.querySelector("img").src = state.he.src;
      elements.spaceRecImage.src = state.spacerec.src;
      updateValue(state.value);
    }
  }

  function pointerValue(event) {
    const rect = elements.singleView.getBoundingClientRect();
    return ((event.clientX - rect.left) / rect.width) * 100;
  }

  function loadDimensions(kind) {
    const probe = new Image();
    probe.onload = () => {
      state[kind].width = probe.naturalWidth;
      state[kind].height = probe.naturalHeight;
      updateCompatibility();
    };
    probe.onerror = () => {
      state[kind].width = 0;
      state[kind].height = 0;
      elements.warning.hidden = false;
      elements.warning.textContent = `Could not load ${state[kind].name}.`;
    };
    probe.src = state[kind].src;
  }

  function updateCompatibility() {
    const he = state.he;
    const sr = state.spacerec;
    if (!he.width || !sr.width) return;
    const heRatio = he.width / he.height;
    const srRatio = sr.width / sr.height;
    const ratioDifference = Math.abs(heRatio - srRatio) / heRatio;
    const dimensionsMatch = he.width === sr.width && he.height === sr.height;
    updateStageAspect();
    elements.status.textContent = `H&E ${he.width}×${he.height} · SpaceRec ${sr.width}×${sr.height}`;
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

  function setImage(kind, src, name, objectUrl = null) {
    if (state[kind].url) URL.revokeObjectURL(state[kind].url);
    state[kind] = { src, name, url: objectUrl, width: 0, height: 0 };
    elements.heImage.src = state.he.src;
    elements.spaceRecImage.src = state.spacerec.src;
    elements.sideHeImage.src = state.he.src;
    elements.sideSpaceRecImage.src = state.spacerec.src;
    setMode(state.mode);
    loadDimensions(kind);
  }

  function upload(kind, file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      elements.warning.hidden = false;
      elements.warning.textContent = `${file.name} is not a supported image file.`;
      return;
    }
    const url = URL.createObjectURL(file);
    setImage(kind, url, file.name, url);
  }

  function reset() {
    setImage("he", DEFAULTS.he, "Registered H&E");
    setImage("spacerec", DEFAULTS.spacerec, "SpaceRec Stage 1");
    elements.heUpload.value = "";
    elements.spaceRecUpload.value = "";
    elements.caption.value = DEFAULTS.caption;
    state.value = 50;
    setMode("swipe");
    updateValue(50);
  }

  function drawContained(context, image, x, y, width, height) {
    const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
    const drawWidth = image.naturalWidth * scale;
    const drawHeight = image.naturalHeight * scale;
    context.drawImage(image, x + (width - drawWidth) / 2, y + (height - drawHeight) / 2, drawWidth, drawHeight);
  }

  function loadedImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = src;
    });
  }

  function exportLabel(context, text, x, y, align = "left") {
    context.save();
    context.font = "700 28px Arial, sans-serif";
    const padding = 14;
    const width = context.measureText(text).width + padding * 2;
    const left = align === "right" ? x - width : x;
    context.fillStyle = "rgba(255,255,255,0.92)";
    context.fillRect(left, y, width, 48);
    context.strokeStyle = "rgba(0,0,0,0.35)";
    context.strokeRect(left, y, width, 48);
    context.fillStyle = "#111315";
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(text, left + padding, y + 25);
    context.restore();
  }

  async function exportPng() {
    elements.export.disabled = true;
    elements.export.textContent = "Preparing…";
    try {
      const [he, sr] = await Promise.all([loadedImage(state.he.src), loadedImage(state.spacerec.src)]);
      const imageWidth = Math.min(8192, he.naturalWidth);
      const imageHeight = Math.round(imageWidth * he.naturalHeight / he.naturalWidth);
      const header = 88;
      const footer = elements.caption.value.trim() ? 72 : 0;
      const gap = state.mode === "side" ? 24 : 0;
      const canvas = document.createElement("canvas");
      canvas.width = state.mode === "side" ? imageWidth * 2 + gap : imageWidth;
      canvas.height = imageHeight + header + footer;
      const context = canvas.getContext("2d");
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = "#17191c";
      context.font = "700 30px Arial, sans-serif";
      context.textBaseline = "middle";
      context.fillText("H&E / SpaceRec comparison", 20, header / 2);
      context.font = "500 24px Arial, sans-serif";
      context.textAlign = "right";
      const modeText = state.mode === "side" ? "Side by side" : state.mode === "opacity" ? `Opacity ${Math.round(state.value)}%` : `Swipe ${Math.round(state.value)}%`;
      context.fillText(modeText, canvas.width - 20, header / 2);

      if (state.mode === "side") {
        drawContained(context, he, 0, header, imageWidth, imageHeight);
        drawContained(context, sr, imageWidth + gap, header, imageWidth, imageHeight);
        exportLabel(context, "H&E", 18, header + 18);
        exportLabel(context, "SpaceRec", canvas.width - 18, header + 18, "right");
      } else if (state.mode === "opacity") {
        drawContained(context, he, 0, header, imageWidth, imageHeight);
        context.save();
        context.globalAlpha = state.value / 100;
        drawContained(context, sr, 0, header, imageWidth, imageHeight);
        context.restore();
        exportLabel(context, "H&E + SpaceRec", 18, header + 18);
      } else {
        drawContained(context, sr, 0, header, imageWidth, imageHeight);
        const split = Math.round(imageWidth * state.value / 100);
        context.save();
        context.beginPath();
        context.rect(0, header, split, imageHeight);
        context.clip();
        drawContained(context, he, 0, header, imageWidth, imageHeight);
        context.restore();
        context.fillStyle = "#ffffff";
        context.fillRect(split - 2, header, 4, imageHeight);
        exportLabel(context, "H&E", 18, header + 18);
        exportLabel(context, "SpaceRec", canvas.width - 18, header + 18, "right");
      }

      if (footer) {
        context.fillStyle = "#17191c";
        context.font = "500 24px Arial, sans-serif";
        context.textAlign = "left";
        context.fillText(elements.caption.value.trim(), 20, canvas.height - footer / 2);
      }

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("PNG encoding failed");
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      const sample = (elements.caption.value.trim() || "comparison").replace(/[^a-z0-9]+/gi, "_").replace(/^_|_$/g, "").slice(0, 70);
      anchor.href = url;
      anchor.download = `${sample}_${state.mode}.png`;
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

  elements.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  elements.range.addEventListener("input", () => updateValue(elements.range.value));
  elements.singleView.addEventListener("pointerdown", (event) => {
    if (state.mode !== "swipe") return;
    event.preventDefault();
    state.dragging = true;
    elements.singleView.setPointerCapture(event.pointerId);
    updateValue(pointerValue(event));
  });
  elements.singleView.addEventListener("pointermove", (event) => {
    if (state.dragging && state.mode === "swipe") updateValue(pointerValue(event));
  });
  elements.singleView.addEventListener("pointerup", () => { state.dragging = false; });
  elements.singleView.addEventListener("pointercancel", () => { state.dragging = false; });
  elements.divider.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
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
  window.addEventListener("resize", updateStageAspect);

  setMode("swipe");
  updateValue(50);
  loadDimensions("he");
  loadDimensions("spacerec");

  window.comparisonApp = { state, setMode, updateValue, reset, exportPng };
})();
