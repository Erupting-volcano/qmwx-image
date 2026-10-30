"use strict";

(() => {
  const limits = { maxFiles: 20, maxFileBytes: 15 * 1024 * 1024, maxTotalBytes: 100 * 1024 * 1024 };
  const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  const ALLOWED_EXTENSION = /\.(jpe?g|png|webp)$/i;
  const elements = Object.fromEntries([
    "upload-form", "upload-fields", "object-name", "image-input", "drop-zone",
    "selection-summary", "file-list", "selection-actions", "clear-files", "form-error",
    "submit-button", "submit-label", "submit-arrow", "submit-spinner", "upload-status",
    "result-section", "result-status", "result-announcement", "profile-name", "profile-id",
    "profile-facts", "result-images", "missing-information", "warnings-section",
    "warnings-list", "profile-json", "view-json", "download-json", "drop-description", "upload-hint",
    "quality-summary", "quality-images", "segmentation-config-note", "segmentation-summary", "segmentation-images",
  ].map((id) => [id, document.getElementById(id)]));

  let selectedFiles = [];
  let isBusy = false;
  let isReadingFiles = false;
  let fileSequence = 0;

  function createNode(tagName, className, content) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return "大小未知";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  }

  function formatLimit(bytes) {
    return bytes >= 1024 * 1024 ? `${Number((bytes / (1024 * 1024)).toFixed(2))} MiB` : formatBytes(bytes);
  }

  async function loadLimits() {
    try {
      const response = await fetch("/api/v1/config", { headers: { Accept: "application/json" } });
      if (!response.ok) return;
      const config = await response.json();
      for (const [field, key] of [["max_files", "maxFiles"], ["max_image_bytes", "maxFileBytes"], ["max_total_bytes", "maxTotalBytes"]]) {
        if (Number.isSafeInteger(config[field]) && config[field] > 0) limits[key] = config[field];
      }
      elements["drop-description"].textContent = `JPEG、PNG、WebP · 单张最多 ${formatLimit(limits.maxFileBytes)}`;
      let hint = `最多 ${limits.maxFiles} 张，合计不超过 ${formatLimit(limits.maxTotalBytes)}。`;
      if (Number.isSafeInteger(config.max_image_pixels) && config.max_image_pixels > 0) {
        hint += `单张解码后不超过 ${new Intl.NumberFormat("zh-CN").format(config.max_image_pixels)} 像素。`;
      }
      elements["upload-hint"].textContent = `${hint}手机 HEIC / HEIF 照片需先转换为 JPEG、PNG 或 WebP。`;
      if (typeof config.segmentation_enabled === "boolean") {
        elements["segmentation-config-note"].textContent = config.segmentation_enabled
          ? "框选分割已启用：展开每张照片并拖出目标范围，再建立档案。服务端仍会检查模型依赖和权重；物体名称不会自动定位目标。"
          : "分割当前未启用：照片与质量评估仍可正常保存。可先框选目标，但需要管理员配置并启用模型后，才会生成掩码。";
      }
      renderFiles();
    } catch {
      // The form remains usable with the default limits; the server validates every upload.
    }
  }

  function showError(message) {
    elements["form-error"].textContent = message;
    elements["form-error"].hidden = !message;
  }

  function setBusy(busy) {
    isBusy = busy;
    elements["upload-fields"].disabled = busy || isReadingFiles;
    elements["submit-button"].disabled = busy || isReadingFiles;
    elements["submit-label"].textContent = busy ? "正在建立档案…" : "建立数字档案";
    elements["submit-arrow"].hidden = busy;
    elements["submit-spinner"].hidden = !busy;
    elements["upload-form"].setAttribute("aria-busy", String(busy || isReadingFiles));
  }

  function makeTargetSelector(entry) {
    const details = createNode("details", "target-selector");
    const summary = createNode("summary", "", "框选分割目标（可选）");
    details.append(summary);
    if (!entry.objectUrl) {
      details.append(createNode("p", "target-help", "当前照片无法预览，不能框选目标；服务器仍会验证照片。"));
      return details;
    }
    details.append(createNode("p", "target-help", "在下图拖出一个矩形，包住要分割的物体。矩形只提供大致位置，最终掩码由模型生成；不会按名称自动定位。"));
    const surface = createNode("div", "target-surface");
    const image = createNode("img", "target-image");
    image.src = entry.objectUrl;
    image.alt = `框选目标预览：${entry.file.name}`;
    image.draggable = false;
    const rectangle = createNode("span", "target-rectangle");
    rectangle.setAttribute("aria-hidden", "true");
    const status = createNode("p", "target-status");
    status.setAttribute("role", "status");
    const clear = createNode("button", "text-button target-clear", "清除目标框");
    clear.type = "button";
    clear.setAttribute("aria-label", `清除目标框：${entry.file.name}`);
    let gesture = null;

    function showBox(box) {
      summary.textContent = entry.segmentationBox ? "已框选分割目标（展开可重画）" : "框选分割目标（可选）";
      rectangle.hidden = !box;
      if (box) {
        rectangle.setAttribute("style", `left:${box[0] * 100}%;top:${box[1] * 100}%;width:${(box[2] - box[0]) * 100}%;height:${(box[3] - box[1]) * 100}%`);
      }
      status.textContent = entry.segmentationBox
        ? `已框选目标：左上 ${Math.round(entry.segmentationBox[0] * 100)}%、${Math.round(entry.segmentationBox[1] * 100)}%，右下 ${Math.round(entry.segmentationBox[2] * 100)}%、${Math.round(entry.segmentationBox[3] * 100)}%。再次拖动可重画。`
        : "尚未框选；启用模型后，需要先框选才能分割这张图片。";
      clear.disabled = !entry.segmentationBox;
    }
    function position(event) {
      const bounds = image.getBoundingClientRect();
      if (!(bounds.width > 0 && bounds.height > 0) || !Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return null;
      return [Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
        Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))];
    }
    function boxBetween(first, second) {
      return [Math.min(first[0], second[0]), Math.min(first[1], second[1]),
        Math.max(first[0], second[0]), Math.max(first[1], second[1])];
    }
    surface.addEventListener("pointerdown", (event) => {
      if (isBusy || isReadingFiles || event.isPrimary === false || (event.button !== undefined && event.button !== 0)) return;
      const point = position(event);
      if (!point) return;
      event.preventDefault();
      gesture = { pointerId: event.pointerId, start: point, previous: entry.segmentationBox };
      surface.setPointerCapture?.(event.pointerId);
      showBox(boxBetween(point, point));
    });
    surface.addEventListener("pointermove", (event) => {
      if (!gesture || event.pointerId !== gesture.pointerId || isBusy || isReadingFiles) return;
      const point = position(event);
      if (point) { event.preventDefault(); showBox(boxBetween(gesture.start, point)); }
    });
    surface.addEventListener("pointerup", (event) => {
      if (!gesture || event.pointerId !== gesture.pointerId) return;
      event.preventDefault();
      const point = position(event);
      const candidate = point ? boxBetween(gesture.start, point) : null;
      const bounds = image.getBoundingClientRect();
      if (!isBusy && !isReadingFiles && candidate && (candidate[2] - candidate[0]) * bounds.width >= 4 && (candidate[3] - candidate[1]) * bounds.height >= 4) {
        entry.segmentationBox = candidate;
      } else {
        entry.segmentationBox = gesture.previous;
      }
      gesture = null;
      if (surface.hasPointerCapture?.(event.pointerId)) surface.releasePointerCapture(event.pointerId);
      showBox(entry.segmentationBox);
    });
    surface.addEventListener("pointercancel", () => {
      if (gesture) entry.segmentationBox = gesture.previous;
      gesture = null;
      showBox(entry.segmentationBox);
    });
    clear.addEventListener("click", () => {
      if (isBusy || isReadingFiles) return;
      gesture = null;
      entry.segmentationBox = null;
      showBox(null);
    });
    surface.append(image, rectangle);
    details.append(surface, status, clear);
    showBox(entry.segmentationBox);
    return details;
  }

  function renderFiles() {
    elements["file-list"].replaceChildren();
    selectedFiles.forEach((entry) => {
      const item = createNode("div", "file-item");
      const preview = entry.objectUrl
        ? createNode("img", "file-thumbnail")
        : createNode("span", "file-thumbnail thumbnail-placeholder", "图片");
      if (entry.objectUrl) {
        preview.src = entry.objectUrl;
        preview.alt = "";
      }
      const copy = createNode("div", "file-copy");
      const name = createNode("span", "file-name", entry.file.name);
      name.title = entry.file.name;
      const dimensions = entry.width && entry.height
        ? `${entry.width} × ${entry.height} px`
        : "由服务器验证图片内容";
      copy.append(name, createNode("span", "file-meta", `${formatBytes(entry.file.size)} · ${dimensions}`));
      const remove = createNode("button", "remove-file", "×");
      remove.id = `remove-file-${entry.id}`;
      remove.type = "button";
      remove.setAttribute("aria-label", `移除照片：${entry.file.name}`);
      remove.addEventListener("click", () => {
        if (isBusy || isReadingFiles) return;
        const index = selectedFiles.findIndex((file) => file.id === entry.id);
        const nextEntry = selectedFiles[index + 1] || selectedFiles[index - 1];
        if (entry.objectUrl) URL.revokeObjectURL(entry.objectUrl);
        selectedFiles = selectedFiles.filter((file) => file.id !== entry.id);
        renderFiles();
        showError("");
        if (nextEntry) {
          document.getElementById(`remove-file-${nextEntry.id}`)?.focus();
        } else {
          elements["image-input"].focus();
        }
      });
      item.append(preview, copy, remove, makeTargetSelector(entry));
      elements["file-list"].append(item);
    });
    const totalBytes = selectedFiles.reduce((total, entry) => total + entry.file.size, 0);
    elements["selection-summary"].textContent = selectedFiles.length
      ? `${selectedFiles.length} / ${limits.maxFiles} 张 · ${formatBytes(totalBytes)}`
      : "尚未选择";
    elements["selection-actions"].hidden = selectedFiles.length === 0;
  }

  function inspectImage(file) {
    return new Promise((resolve) => {
      const objectUrl = URL.createObjectURL(file);
      const image = new Image();
      const timer = window.setTimeout(() => finish(null), 8000);
      let finished = false;
      function finish(dimensions) {
        if (finished) return;
        finished = true;
        window.clearTimeout(timer);
        image.onload = null;
        image.onerror = null;
        if (!dimensions) URL.revokeObjectURL(objectUrl);
        resolve({
          id: ++fileSequence,
          file,
          objectUrl: dimensions ? objectUrl : null,
          width: dimensions?.width,
          height: dimensions?.height,
          segmentationBox: null,
        });
      }
      image.onload = () => finish({ width: image.naturalWidth, height: image.naturalHeight });
      image.onerror = () => finish(null);
      image.src = objectUrl;
    });
  }

  async function addFiles(fileList) {
    if (isBusy || isReadingFiles) return;
    const incoming = Array.from(fileList || []);
    if (!incoming.length) return;
    showError("");
    const errors = [];
    const candidates = [];
    let totalBytes = selectedFiles.reduce((total, entry) => total + entry.file.size, 0);
    for (const file of incoming) {
      if (!ALLOWED_EXTENSION.test(file.name) || (file.type && !ALLOWED_TYPES.has(file.type))) {
        errors.push(`“${file.name}”格式不支持。请选择 JPEG、PNG 或 WebP；HEIC / HEIF 请先转换格式。`);
        continue;
      }
      if (file.size === 0) {
        errors.push(`“${file.name}”是空文件，请重新选择。`);
        continue;
      }
      if (file.size > limits.maxFileBytes) {
        errors.push(`“${file.name}”超过单张 ${formatLimit(limits.maxFileBytes)} 的限制。`);
        continue;
      }
      const exists = [...selectedFiles.map((entry) => entry.file), ...candidates].some(
        (existing) => existing.name === file.name && existing.size === file.size && existing.lastModified === file.lastModified,
      );
      if (exists) {
        errors.push(`“${file.name}”已经选中，已跳过重复照片。`);
        continue;
      }
      if (selectedFiles.length + candidates.length >= limits.maxFiles) {
        errors.push(`最多可选择 ${limits.maxFiles} 张照片，多出的照片未添加。`);
        break;
      }
      if (totalBytes + file.size > limits.maxTotalBytes) {
        errors.push(`添加“${file.name}”将超过整批 ${formatLimit(limits.maxTotalBytes)} 的限制，未添加此照片。`);
        continue;
      }
      candidates.push(file);
      totalBytes += file.size;
    }
    if (candidates.length) {
      isReadingFiles = true;
      setBusy(false);
      elements["upload-status"].textContent = "正在读取照片预览…";
      try {
        const entries = await Promise.all(candidates.map(inspectImage));
        selectedFiles.push(...entries);
        renderFiles();
      } catch {
        errors.push("读取照片预览失败，请重新选择照片。");
      } finally {
        isReadingFiles = false;
        setBusy(false);
        elements["upload-status"].textContent = "";
      }
    }
    showError([...new Set(errors)].join("\n"));
    elements["image-input"].value = "";
  }

  function getErrorMessage(payload, httpStatus) {
    const detail = payload?.detail ?? payload;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail)) {
      const messages = detail.map((item) => typeof item === "string" ? item : item?.msg || item?.message).filter(Boolean);
      if (messages.length) return messages.join("\n");
    }
    if (detail && typeof detail === "object" && typeof detail.message === "string") return detail.message;
    if (httpStatus === 413) return `上传内容过大。单张最多 ${formatLimit(limits.maxFileBytes)}，整批最多 ${formatLimit(limits.maxTotalBytes)}。`;
    if (httpStatus === 422) return "提交内容未通过验证，请检查物体名称和图片格式。";
    if (httpStatus >= 500) return "服务暂时无法完成处理，请稍后重试。";
    return `建立档案失败（HTTP ${httpStatus}），请检查图片后重试。`;
  }

  function renderFact(label, value, unknown = false) {
    const item = createNode("div", "fact");
    item.append(createNode("dt", "", label), createNode("dd", unknown ? "is-unknown" : "", value));
    elements["profile-facts"].append(item);
  }

  function describeMissing(value) {
    const fieldLabels = {
      category: "物体类别：等待识别与确认",
      materials: "物体材质：等待识别与确认",
      measurements: "真实尺寸：需要测量或经过标定的估计",
      geometry: "三维几何：尚未生成模型，比例尚未标定",
      components: "组成部件：等待检测与确认",
    };
    if (typeof value === "string") return fieldLabels[value] || value;
    if (!value || typeof value !== "object") return "存在待补充的信息";
    const label = value.message || value.description || value.reason || value.field || value.name;
    return typeof label === "string" ? fieldLabels[label] || label : JSON.stringify(value);
  }

  function renderQuality(profile, images) {
    const results = Array.isArray(profile.processing_results) ? profile.processing_results : [];
    // A profile may retain several processing runs; use the latest quality assessment.
    const result = [...results].reverse().find((item) => item?.module_name === "image_quality");
    const records = ["success", "partial"].includes(result?.status) && Array.isArray(result?.data?.images) ? result.data.images : [];
    const byId = new Map(records.filter((item) => typeof item?.asset_id === "string").map((item) => [item.asset_id, item]));
    const errors = Array.isArray(result?.errors) ? result.errors : [];
    const errorById = new Map(errors.filter((item) => typeof item?.asset_id === "string").map((item) => [item.asset_id, item]));
    const checked = images.filter((asset) => ["good", "review"].includes(byId.get(asset.asset_id)?.status));
    const reviewCount = checked.filter((asset) => byId.get(asset.asset_id).status === "review").length;
    elements["quality-summary"].textContent = result
      ? `已评估 ${checked.length} / ${images.length} 张图片，${reviewCount} 张建议复查。${checked.length < images.length ? "未完成评估的图片见下方。" : ""}`
      : "本份档案未执行图像质量评估，不能据此判断照片质量。";

    const issueLabels = {
      low_resolution: "分辨率偏低",
      too_dark: "画面整体偏暗",
      too_bright: "画面整体偏亮",
      low_contrast: "整体对比度偏低",
      low_detail: "细节响应较弱，建议人工复查",
    };
    const formatMetric = (value) => Number.isFinite(value) ? value.toFixed(2) : "未取得";
    elements["quality-images"].replaceChildren();
    images.forEach((asset, index) => {
      const assessment = byId.get(asset.asset_id);
      const isChecked = ["good", "review"].includes(assessment?.status);
      const needsReview = isChecked && assessment.status === "review";
      const error = errorById.get(asset.asset_id);
      const item = createNode("li", `quality-image${needsReview || !isChecked ? " needs-review" : ""}`);
      const heading = createNode("div", "quality-image-heading");
      const label = isChecked ? (needsReview ? "建议复查" : "未发现明显问题") : (result ? "评估未完成" : "未评估");
      heading.append(
        createNode("h5", "quality-image-name", `${String(index + 1).padStart(2, "0")}  ${asset.filename || "未命名图片"}`),
        createNode("span", `quality-badge${needsReview || !isChecked ? " is-review" : ""}`, label),
      );
      item.append(heading);
      if (isChecked) {
        item.append(createNode("p", "quality-metrics", `平均亮度 ${formatMetric(assessment.mean_luminance)} / 255 · 对比度 ${formatMetric(assessment.contrast_stddev)} · 边缘响应 ${formatMetric(assessment.edge_strength)}`));
        if (Number.isFinite(assessment.analysis_width) && Number.isFinite(assessment.analysis_height)) {
          item.append(createNode("p", "quality-analysis-size", `分析尺寸 ${assessment.analysis_width} × ${assessment.analysis_height} px；指标用于启发式复查。`));
        }
        const issues = Array.isArray(assessment.issues) ? assessment.issues.filter((value) => typeof value === "string") : [];
        if (issues.length) item.append(createNode("p", "quality-issues", `复查项：${issues.map((code) => issueLabels[code] || code).join("、")}`));
        const suggestions = Array.isArray(assessment.suggestions) ? assessment.suggestions.filter((value) => typeof value === "string") : [];
        if (suggestions.length) {
          const list = createNode("ul", "quality-suggestions");
          suggestions.forEach((suggestion) => list.append(createNode("li", "", suggestion)));
          item.append(list);
        }
      } else {
        const message = typeof error?.message === "string" ? error.message
          : result ? "这张图片未完成质量评估，请稍后重新处理；原始图片与档案仍已保留。" : "本次处理没有这张图片的质量评估结果。";
        item.append(createNode("p", "quality-issues", message));
      }
      elements["quality-images"].append(item);
    });
    if (!images.length) elements["quality-images"].append(createNode("li", "quality-note", "暂无可评估的图片记录。"));
  }

  function renderSegmentation(profile, images, objectId) {
    const results = Array.isArray(profile.processing_results) ? profile.processing_results : [];
    const result = [...results].reverse().find((item) => item?.module_name === "object_segmentation");
    const records = Array.isArray(result?.data?.images) ? result.data.images : [];
    const byId = new Map(records.filter((item) => typeof item?.asset_id === "string").map((item) => [item.asset_id, item]));
    const maskStatuses = new Set(["success", "needs_confirmation"]);
    const canShowMask = ["success", "partial", "needs_input"].includes(result?.status);
    const labels = { success: "分割成功", needs_confirmation: "需要人工确认", not_found: "未找到有效目标",
      failed: "分割失败", disabled: "未启用", unavailable: "模型不可用", needs_input: "需要框选目标" };
    const known = images.map((asset) => byId.get(asset.asset_id)).filter(Boolean);
    const succeeded = canShowMask ? known.filter((item) => item.status === "success").length : 0;
    const review = canShowMask ? known.filter((item) => item.status === "needs_confirmation").length : 0;
    const failed = known.filter((item) => ["failed", "not_found"].includes(item.status)
      || (!canShowMask && maskStatuses.has(item.status))).length;
    elements["segmentation-summary"].textContent = result
      ? `共 ${images.length} 张图片，分割成功 ${succeeded} 张，待人工确认 ${review} 张，失败或未找到目标 ${failed} 张。未执行或失败的原因见各图说明。`
      : "本份档案未执行物体分割；没有掩码不代表没有物体。";
    elements["segmentation-images"].replaceChildren();
    images.forEach((asset, index) => {
      const record = byId.get(asset.asset_id);
      const accepted = record && (canShowMask || !maskStatuses.has(record.status));
      const status = accepted ? record.status : result ? "failed" : null;
      const item = createNode("li", "segmentation-image");
      const header = createNode("div", "quality-image-heading");
      header.append(createNode("h5", "quality-image-name", `${String(index + 1).padStart(2, "0")}  ${asset.filename || "未命名图片"}`),
        createNode("span", "quality-badge is-review", labels[status] || (result ? "分割未完成" : "未执行")));
      item.append(header);
      if (accepted && typeof record.model_name === "string") {
        const prompt = record.prompt_type === "box" ? "人工矩形框" : "未提供目标框";
        item.append(createNode("p", "segmentation-metrics", `模型：${record.model_name} · 提示方式：${prompt}；目标位置不由名称自动识别。`));
      }
      if (accepted && Number.isFinite(record.width) && Number.isFinite(record.height)) {
        item.append(createNode("p", "segmentation-metrics", `分割坐标图 ${record.width} × ${record.height} px（按 EXIF 转正）。`));
      }
      if (accepted && Array.isArray(record.bbox) && record.bbox.length === 4 && record.bbox.every(Number.isFinite)) {
        item.append(createNode("p", "segmentation-metrics", `边界框 [${record.bbox.join(", ")}] px（左上包含、右下不包含；EXIF 转正坐标）。`));
      }
      if (accepted && Number.isFinite(record.area_ratio)) {
        item.append(createNode("p", "segmentation-metrics", `前景面积占比 ${(record.area_ratio * 100).toFixed(2)}%。`));
      }
      if (accepted && Number.isFinite(record.confidence)) {
        item.append(createNode("p", "segmentation-metrics", `模型候选分数 ${record.confidence.toFixed(3)}；分数不是目标正确的概率。`));
      }
      const warnings = accepted && Array.isArray(record.warnings) ? record.warnings.filter((value) => typeof value === "string") : [];
      if (warnings.length) {
        const list = createNode("ul", "quality-suggestions");
        warnings.forEach((warning) => list.append(createNode("li", "", warning)));
        item.append(list);
      }
      if (!accepted) item.append(createNode("p", "quality-issues", result
        ? "本次分割结果不可用；原图与其他有效处理信息仍已保存。" : "本次处理没有此图片的分割结果。"));
      if (accepted && status === "needs_input") item.append(createNode("p", "quality-issues", "请在上传前展开照片，在目标四周拖出矩形，再建立档案。"));
      if (accepted && status === "disabled") item.append(createNode("p", "quality-issues", "后台未启用分割；不会生成掩码，图片采集和质量评估不受影响。"));
      if (accepted && status === "unavailable") item.append(createNode("p", "quality-issues", "分割模型不可用，请管理员检查可选依赖、设备和本地权重配置。"));
      if (accepted && typeof record.error_id === "string") item.append(createNode("p", "segmentation-metrics", `错误编号：${record.error_id}`));

      if (typeof objectId === "string" && typeof asset.asset_id === "string") {
        const base = `/api/v1/objects/${encodeURIComponent(objectId)}/images/${encodeURIComponent(asset.asset_id)}`;
        const previews = createNode("div", "segmentation-previews");
        const original = createNode("figure");
        const image = createNode("img", "segmentation-preview");
        image.src = `${base}/original`;
        image.alt = `原图：${asset.filename || "未命名图片"}`;
        image.loading = "lazy";
        original.append(image, createNode("figcaption", "", "原图（按 EXIF 方向显示）"));
        previews.append(original);
        if (canShowMask && accepted && maskStatuses.has(status) && typeof record.mask_path === "string") {
          const maskFigure = createNode("figure");
          const mask = createNode("img", "segmentation-preview segmentation-mask");
          mask.src = `${base}/mask`;
          mask.alt = `分割掩码：${asset.filename || "未命名图片"}`;
          mask.loading = "lazy";
          const link = createNode("a", "", "查看掩码 PNG");
          link.href = `${base}/mask`;
          link.target = "_blank";
          link.rel = "noopener";
          const caption = createNode("figcaption", "", "白色为前景，黑色为背景。 ");
          caption.append(link);
          maskFigure.append(mask, caption);
          previews.append(maskFigure);
        }
        item.append(previews);
      }
      elements["segmentation-images"].append(item);
    });
    if (!images.length) elements["segmentation-images"].append(createNode("li", "quality-note", "暂无可分割的图片记录。"));
  }

  function renderResult(payload) {
    const profile = payload.profile;
    const objectId = payload.object_id;
    const images = Array.isArray(profile.images) ? profile.images : [];
    const warnings = [];
    if (Array.isArray(profile.warnings)) warnings.push(...profile.warnings.filter((item) => typeof item === "string"));
    if (Array.isArray(profile.processing_results)) {
      profile.processing_results.forEach((result) => {
        if (Array.isArray(result.warnings)) warnings.push(...result.warnings.filter((item) => typeof item === "string"));
        if (Array.isArray(result.errors)) {
          result.errors.forEach((error) => {
            if (typeof error?.message === "string") warnings.push(`${result.module_name || "处理模块"}：${error.message}`);
          });
        }
      });
    }
    const uniqueWarnings = [...new Set(warnings)];
    const hasWarnings = payload.status === "completed_with_warnings" || uniqueWarnings.length > 0;
    elements["profile-name"].textContent = profile.name || elements["object-name"].value.trim();
    elements["profile-id"].textContent = `档案编号 ${objectId}`;
    elements["result-status"].textContent = hasWarnings ? "已保存 · 含处理提示" : "已保存";
    elements["result-status"].classList.toggle("has-warnings", hasWarnings);
    elements["profile-facts"].replaceChildren();
    renderFact("图像记录", `${images.length} 张照片`);
    const category = typeof profile.category === "string" ? profile.category : profile.category?.name;
    renderFact("物体类别", category || "待识别", !category);
    const measurements = Array.isArray(profile.measurements) ? profile.measurements.filter((item) => item?.value !== null && item?.value !== undefined) : [];
    renderFact("真实尺寸", measurements.length ? `已记录 ${measurements.length} 项` : "待测量", !measurements.length);
    const modelStatus = profile.geometry?.model_status;
    const modelLabels = { not_started: "待生成", not_generated: "待生成", pending: "待处理", partial: "部分生成", ready: "已生成", failed: "生成失败" };
    const modelKnown = ["partial", "ready"].includes(modelStatus);
    renderFact("三维模型", modelLabels[modelStatus] || "待生成", !modelKnown);

    elements["result-images"].replaceChildren();
    images.forEach((asset, index) => {
      const item = createNode("li");
      const dimensions = Number.isFinite(asset.width) && Number.isFinite(asset.height)
        ? `${asset.width} × ${asset.height} px` : "像素尺寸未知";
      const type = typeof asset.mime_type === "string" ? asset.mime_type.replace("image/", "").toUpperCase() : "图片";
      item.append(
        createNode("span", "result-image-name", `${String(index + 1).padStart(2, "0")}  ${asset.filename || "未命名图片"}`),
        createNode("span", "result-image-meta", `${type} · ${dimensions} · ${formatBytes(asset.size_bytes)}`),
      );
      elements["result-images"].append(item);
    });
    if (!images.length) elements["result-images"].append(createNode("li", "result-image-meta", "暂无可用图片记录"));
    renderQuality(profile, images);
    renderSegmentation(profile, images, objectId);

    elements["missing-information"].replaceChildren();
    let missing = profile.missing_information;
    if (missing && !Array.isArray(missing) && typeof missing === "object") {
      missing = Object.entries(missing).map(([field, value]) => typeof value === "string" ? `${field}：${value}` : { field, ...value });
    }
    const missingItems = Array.isArray(missing) ? missing : missing ? [missing] : [];
    if (!missingItems.length) missingItems.push("暂无额外待补信息；空值仍表示尚未确认。");
    missingItems.forEach((item) => elements["missing-information"].append(createNode("li", "", describeMissing(item))));

    elements["warnings-list"].replaceChildren();
    uniqueWarnings.forEach((warning) => elements["warnings-list"].append(createNode("li", "", warning)));
    elements["warnings-section"].hidden = uniqueWarnings.length === 0;
    elements["profile-json"].textContent = JSON.stringify(profile, null, 2);
    const encodedId = encodeURIComponent(objectId);
    elements["view-json"].href = `/api/v1/objects/${encodedId}`;
    elements["download-json"].href = `/api/v1/objects/${encodedId}/profile.json`;
    elements["result-section"].hidden = false;
    elements["result-announcement"].textContent = hasWarnings
      ? `档案已保存，共 ${images.length} 张图片。请查看处理提示和待补充的信息。`
      : `档案已保存，共 ${images.length} 张图片。`;
    elements["result-section"].scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    elements["profile-name"].focus({ preventScroll: true });
  }

  elements["image-input"].addEventListener("change", (event) => addFiles(event.target.files));
  elements["clear-files"].addEventListener("click", () => {
    if (isBusy || isReadingFiles) return;
    selectedFiles.forEach((entry) => { if (entry.objectUrl) URL.revokeObjectURL(entry.objectUrl); });
    selectedFiles = [];
    elements["image-input"].value = "";
    renderFiles();
    showError("");
    elements["image-input"].focus();
  });

  let dragDepth = 0;
  elements["drop-zone"].addEventListener("dragenter", (event) => {
    event.preventDefault();
    if (isBusy || isReadingFiles) return;
    dragDepth += 1;
    elements["drop-zone"].classList.add("is-dragging");
  });
  elements["drop-zone"].addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = isBusy || isReadingFiles ? "none" : "copy";
  });
  elements["drop-zone"].addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) elements["drop-zone"].classList.remove("is-dragging");
  });
  elements["drop-zone"].addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    elements["drop-zone"].classList.remove("is-dragging");
    addFiles(event.dataTransfer?.files);
  });

  elements["upload-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    if (isBusy || isReadingFiles) return;
    showError("");
    const name = elements["object-name"].value.trim();
    if (!name) {
      showError("请为物体填写一个名称。");
      elements["object-name"].focus();
      return;
    }
    if ([...name].length > 120) {
      showError("物体名称不能超过 120 个字符。");
      elements["object-name"].focus();
      return;
    }
    if (!selectedFiles.length) {
      showError("请至少选择一张 JPEG、PNG 或 WebP 照片。");
      elements["image-input"].focus();
      return;
    }
    if (selectedFiles.length > limits.maxFiles || selectedFiles.some((entry) => entry.file.size > limits.maxFileBytes) || selectedFiles.reduce((total, entry) => total + entry.file.size, 0) > limits.maxTotalBytes) {
      showError(`所选照片超过服务当前限制：最多 ${limits.maxFiles} 张，单张 ${formatLimit(limits.maxFileBytes)}，合计 ${formatLimit(limits.maxTotalBytes)}。请移除超限照片。`);
      return;
    }
    const formData = new FormData();
    formData.append("name", name);
    selectedFiles.forEach((entry) => formData.append("images", entry.file, entry.file.name));
    if (selectedFiles.some((entry) => entry.segmentationBox)) {
      formData.append("segmentation_boxes", JSON.stringify(selectedFiles.map((entry) => entry.segmentationBox)));
    }
    setBusy(true);
    elements["upload-status"].textContent = "正在上传、评估图像质量并处理已配置的分割任务，请保持页面打开…";
    try {
      const response = await fetch("/api/v1/objects", { method: "POST", body: formData, headers: { Accept: "application/json" } });
      let payload;
      try { payload = await response.json(); } catch { payload = null; }
      if (!response.ok) throw new Error(getErrorMessage(payload, response.status));
      if (!payload || typeof payload.object_id !== "string" || !payload.profile || typeof payload.profile !== "object") {
        throw new Error("服务返回的档案格式不完整，无法展示结果。请查看服务日志后重试。");
      }
      renderResult(payload);
      elements["upload-status"].textContent = "档案已保存。你可以下载 JSON，或继续创建下一份档案。";
    } catch (error) {
      showError(error instanceof TypeError ? "无法连接到服务，请检查网络或确认服务仍在运行，再重试。" : error.message || "建立档案失败，请稍后重试。");
      elements["upload-status"].textContent = "本次请求未确认成功，已保留你的选择。";
    } finally {
      setBusy(false);
    }
  });
  loadLimits();
})();
