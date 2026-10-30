"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.value = "";
    this.hidden = false;
    this._text = "";
    this.classes = new Set();
    this.bounds = { left: 100, top: 200, width: 400, height: 300 };
    this.classList = {
      add: (value) => this.classes.add(value),
      remove: (value) => this.classes.delete(value),
      toggle: (value, enabled) => enabled ? this.classes.add(value) : this.classes.delete(value),
    };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((item) => item.textContent).join(""); }
  set innerHTML(_) { throw new Error("Unsafe dynamic HTML insertion"); }
  set outerHTML(_) { throw new Error("Unsafe dynamic HTML insertion"); }
  insertAdjacentHTML() { throw new Error("Unsafe dynamic HTML insertion"); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this._text = ""; this.children = nodes; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getBoundingClientRect() { return this.bounds; }
  setPointerCapture(id) { this.capture = id; }
  hasPointerCapture(id) { return this.capture === id; }
  releasePointerCapture() { this.capture = null; }
  focus() {}
  scrollIntoView() {}
  querySelectorAll(tagName) { return descendants(this).filter((item) => item.tagName === tagName); }
}

function descendants(node) {
  return node.children.flatMap((child) => [child, ...descendants(child)]);
}
function byClass(node, value) {
  return descendants(node).find((item) => (item.className || "").split(" ").includes(value));
}
function pointer(node, name, x, y, options = {}) {
  const event = { pointerId: 1, isPrimary: true, button: 0, clientX: x, clientY: y,
    preventDefault() {}, ...options };
  node.listeners[name](event);
}
function drag(node, start, end, options = {}) {
  pointer(node, "pointerdown", ...start, options);
  pointer(node, "pointermove", ...end, options);
  pointer(node, "pointerup", ...end, options);
}

const malicious = '<img src=x onerror="globalThis.pwned=true">';
const objectId = "11111111-1111-4111-8111-111111111111";
const asset = (asset_id, filename = "same.png") => ({
  asset_id, filename, width: 800, height: 600, size_bytes: 300, mime_type: "image/png",
});
const record = (asset_id, status, extra = {}) => ({
  asset_id, filename: "A module filename must not select the displayed image", status,
  model_name: "sam2.1_hiera_tiny", prompt_type: "box", width: 800, height: 600,
  coordinate_system: "exif_oriented_pixels", mask_path: ["success", "needs_confirmation"].includes(status) ? "../../ignored.png" : null,
  bbox: [80, 120, 720, 480], area_ratio: 0.35, confidence: 0.85, warnings: [], error_id: null,
  ...extra,
});
const processingResult = (images, status = "partial") => ({
  module_name: "object_segmentation", status, data: { input_count: images.length, images }, warnings: [], errors: [],
});
const file = (size = 300, name = "same.png") => ({ name, size, type: "image/png", lastModified: size });

async function setup(profile, enabled = true) {
  const html = fs.readFileSync(path.join(__dirname, "../app/web/index.html"), "utf8");
  const nodes = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => [match[1], new Element("div")]));
  let submission;
  let urlSequence = 0;
  const context = {
    document: {
      getElementById: (id) => nodes[id] || Object.values(nodes).flatMap(descendants).find((item) => item.id === id),
      createElement: (tagName) => new Element(tagName),
    },
    window: { setTimeout, clearTimeout, matchMedia: () => ({ matches: true }) },
    URL: { createObjectURL: () => `blob:local-${++urlSequence}`, revokeObjectURL() {} },
    Image: class {
      constructor() { this.naturalWidth = 800; this.naturalHeight = 600; }
      set src(_) { queueMicrotask(() => this.onload?.()); }
    },
    FormData: class {
      constructor() { this.entries = []; }
      append(name, value, filename) { this.entries.push({ name, value, filename }); }
    },
    fetch: async (url, options) => {
      if (url === "/api/v1/config") return { ok: true, json: async () => ({ segmentation_enabled: enabled }) };
      submission = options.body;
      return { ok: true, status: 201, json: async () => ({ object_id: objectId, status: "completed_with_warnings", profile }) };
    },
    console,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../app/web/app.js"), "utf8"), context);
  await new Promise(setImmediate);
  nodes["object-name"].value = "DOM 分割验收";
  return {
    nodes, context,
    add: (files) => nodes["image-input"].listeners.change({ target: { files } }),
    async submit() {
      await nodes["upload-form"].listeners.submit({ preventDefault() {} });
      assert.equal(nodes["form-error"].textContent, "");
      assert.equal(context.pwned, undefined);
      return submission.entries;
    },
  };
}

async function main() {
  const scenario = process.argv[2];
  const profile = { name: malicious, images: [asset("a"), asset("b")], warnings: [], processing_results: [] };
  if (scenario === "box_coordinates") {
    const app = await setup(profile);
    await app.add([file()]);
    const card = app.nodes["file-list"].children[0];
    const surface = byClass(card, "target-surface");
    drag(surface, [140, 260], [460, 440]);
    assert.match(byClass(card, "target-status").textContent, /已框选目标/);
    const entries = await app.submit();
    assert.deepEqual(JSON.parse(entries.find((item) => item.name === "segmentation_boxes").value), [[0.1, 0.2, 0.9, 0.8]]);
    assert.match(app.nodes["segmentation-config-note"].textContent, /名称不会自动定位/);
  } else if (scenario === "pointer_cancel_and_clear") {
    const app = await setup(profile);
    await app.add([file()]);
    const card = app.nodes["file-list"].children[0];
    const surface = byClass(card, "target-surface");
    drag(surface, [420, 440], [20, 100], { pointerType: "touch" });
    pointer(surface, "pointerdown", 200, 250);
    pointer(surface, "pointermove", 300, 300);
    surface.listeners.pointercancel();
    const entries = await app.submit();
    assert.deepEqual(JSON.parse(entries.find((item) => item.name === "segmentation_boxes").value), [[0, 0, 0.8, 0.8]]);
    byClass(card, "target-clear").listeners.click();
    assert.match(byClass(card, "target-status").textContent, /尚未框选/);
    const cleared = await app.submit();
    assert.equal(cleared.some((item) => item.name === "segmentation_boxes"), false);
  } else if (scenario === "selection_identity") {
    const app = await setup(profile);
    await app.add([file(300), file(301)]);
    const [first, second] = app.nodes["file-list"].children;
    drag(byClass(first, "target-surface"), [140, 260], [460, 440]);
    drag(byClass(second, "target-surface"), [180, 290], [420, 410]);
    byClass(first, "remove-file").listeners.click();
    await app.add([file(302, "third.png")]);
    const entries = await app.submit();
    assert.deepEqual(entries.filter((item) => item.name === "images").map((item) => item.value.size), [301, 302]);
    assert.deepEqual(JSON.parse(entries.find((item) => item.name === "segmentation_boxes").value), [[0.2, 0.3, 0.8, 0.7], null]);
  } else if (scenario === "oriented_scaled_image") {
    const app = await setup(profile);
    await app.add([file()]);
    const card = app.nodes["file-list"].children[0];
    byClass(card, "target-image").bounds = { left: 50, top: 80, width: 300, height: 400 };
    drag(byClass(card, "target-surface"), [110, 180], [290, 380]);
    const entries = await app.submit();
    assert.deepEqual(JSON.parse(entries.find((item) => item.name === "segmentation_boxes").value), [[0.2, 0.25, 0.8, 0.75]]);
  } else if (scenario === "mixed_results") {
    profile.images = [asset("a", malicious), asset("b", malicious), asset("c"), asset("d"), asset("e"), asset("f"), asset("g")];
    profile.processing_results = [processingResult([
      record("b", "needs_confirmation", { warnings: [malicious] }), record("a", "success", { model_name: malicious }),
      record("c", "disabled"), record("d", "unavailable"), record("e", "needs_input", { prompt_type: "none" }), record("f", "not_found"),
      record("g", "failed", { error_id: malicious }), record("foreign", "success"),
    ])];
    const app = await setup(profile);
    await app.add([file()]);
    await app.submit();
    const items = app.nodes["segmentation-images"].children;
    assert.equal(items.length, 7);
    assert.match(app.nodes["segmentation-summary"].textContent, /成功 1 张，待人工确认 1 张/);
    assert.match(items[0].textContent, /分割成功/);
    assert.match(items[1].textContent, /需要人工确认/);
    assert.match(items[2].textContent, /未启用/);
    assert.match(items[3].textContent, /模型不可用/);
    assert.match(items[4].textContent, /需要框选目标/);
    assert.match(items[4].textContent, /未提供目标框/);
    assert.match(items[5].textContent, /未找到有效目标/);
    assert.match(items[6].textContent, /分割失败/);
    assert.ok(items[0].textContent.includes(malicious));
    assert.ok(items[1].textContent.includes(malicious));
    assert.ok(items[6].textContent.includes(malicious));
    assert.match(items[0].textContent, /35\.00%/);
    const images = app.nodes["segmentation-images"].querySelectorAll("img");
    assert.equal(images.filter((image) => image.src.endsWith("/mask")).length, 2);
    assert.equal(images.filter((image) => image.src.endsWith("/original")).length, 7);
    assert.equal(images[0].src, `/api/v1/objects/${objectId}/images/a/original`);
    assert.ok(images.every((image) => image.src.startsWith(`/api/v1/objects/${objectId}/images/`)));
    assert.ok(images.every((image) => !image.src.includes("ignored.png")));
    assert.doesNotMatch(app.nodes["segmentation-images"].textContent, /A module filename/);
  } else if (scenario === "disabled_and_old_archive") {
    const app = await setup(profile, false);
    await app.add([file()]);
    await app.submit();
    assert.match(app.nodes["segmentation-config-note"].textContent, /当前未启用/);
    assert.match(app.nodes["segmentation-summary"].textContent, /未执行物体分割/);
    assert.equal(app.nodes["segmentation-images"].querySelectorAll("img").some((image) => image.src.endsWith("/mask")), false);
  } else if (scenario === "failed_diagnostic_data") {
    profile.processing_results = [processingResult([record("a", "success")], "success"),
      processingResult([record("a", "success")], "failed")];
    const app = await setup(profile);
    await app.add([file()]);
    await app.submit();
    assert.match(app.nodes["segmentation-summary"].textContent, /成功 0 张/);
    assert.equal(app.nodes["segmentation-images"].querySelectorAll("img").some((image) => image.src.endsWith("/mask")), false);
    assert.match(app.nodes["segmentation-images"].textContent, /原图与其他有效处理信息仍已保存/);
  } else {
    throw new Error(`Unknown scenario: ${scenario}`);
  }
  console.log(`segmentation DOM checks passed: ${scenario}`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
