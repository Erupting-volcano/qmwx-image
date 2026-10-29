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
    this.classList = {
      add: (value) => this.classes.add(value),
      remove: (value) => this.classes.delete(value),
      toggle: (value, on) => on ? this.classes.add(value) : this.classes.delete(value),
    };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((item) => item.textContent).join(""); }
  set innerHTML(_) { throw new Error("Unsafe dynamic innerHTML write"); }
  set outerHTML(_) { throw new Error("Unsafe dynamic outerHTML write"); }
  insertAdjacentHTML() { throw new Error("Unsafe dynamic HTML insertion"); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this._text = ""; this.children = nodes; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  focus() {}
  scrollIntoView() {}
  querySelectorAll(tagName) {
    return this.children.flatMap((item) => [
      ...(item.tagName === tagName ? [item] : []), ...item.querySelectorAll(tagName),
    ]);
  }
}

const malicious = '<img src=x onerror="globalThis.pwned=true">';
const asset = (asset_id, filename = malicious) => ({
  asset_id, filename, width: 800, height: 600, mime_type: "image/png", size_bytes: 300,
});
const quality = (asset_id, status = "review") => ({
  asset_id, filename: "Filename from the module must not select the displayed asset",
  width: 800, height: 600, analysis_width: 800, analysis_height: 600,
  mean_luminance: 12.25, contrast_stddev: 3.5, edge_strength: 0.2,
  status, issues: status === "review" ? ["too_dark", "low_detail", malicious] : [],
  suggestions: status === "review" ? ["增加均匀照明后重拍。", malicious] : [],
});
const result = (images, extra = {}) => ({
  module_name: "image_quality", status: "success", warnings: [], errors: [],
  data: { input_count: 2, checked_count: images.length, review_count: images.filter((item) => item.status === "review").length, images },
  ...extra,
});

async function render(profile) {
  const html = fs.readFileSync(path.join(__dirname, "../app/web/index.html"), "utf8");
  const nodes = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => [match[1], new Element("div")]));
  const context = {
    document: { getElementById: (id) => nodes[id], createElement: (name) => new Element(name) },
    window: { setTimeout, clearTimeout, matchMedia: () => ({ matches: true }) },
    URL: { createObjectURL: () => "blob:local-preview", revokeObjectURL: () => {} },
    Image: class {
      constructor() { this.naturalWidth = 800; this.naturalHeight = 600; }
      set src(_) { queueMicrotask(() => this.onload?.()); }
    },
    FormData: class { append() {} },
    fetch: async (url) => url === "/api/v1/config"
      ? { ok: false }
      : { ok: true, status: 201, json: async () => ({ object_id: "object-test", status: profile.status || "completed_with_warnings", profile }) },
    console,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../app/web/app.js"), "utf8"), context);
  nodes["object-name"].value = "DOM 测试档案";
  await nodes["image-input"].listeners.change({ target: { files: [{ name: "input.png", type: "image/png", size: 300, lastModified: 1 }] } });
  await nodes["upload-form"].listeners.submit({ preventDefault() {} });
  assert.equal(nodes["form-error"].textContent, "");
  assert.equal(nodes["result-section"].hidden, false);
  assert.equal(context.pwned, undefined);
  assert.equal(nodes["download-json"].href, "/api/v1/objects/object-test/profile.json");
  return nodes;
}

async function main() {
  const scenario = process.argv[2];
  const profile = { name: malicious, images: [asset("a"), asset("b")], warnings: [], processing_results: [] };
  if (scenario === "mixed") {
    profile.images.push(asset("c", "同批评估失败.png"));
    profile.processing_results.push(result([quality("b", "good"), quality("a")], {
      status: "partial", errors: [{ asset_id: "c", message: `无法读取 ${malicious}` }],
    }));
    const nodes = await render(profile);
    const items = nodes["quality-images"].children;
    assert.equal(items.length, 3);
    assert.match(nodes["quality-summary"].textContent, /已评估 2 \/ 3 张图片，1 张建议复查/);
    assert.match(items[0].textContent, /建议复查/);
    assert.match(items[0].textContent, /画面整体偏暗/);
    assert.match(items[0].textContent, /细节响应较弱/);
    assert.match(items[0].textContent, /12\.25 \/ 255/);
    assert.match(items[0].textContent, /增加均匀照明后重拍/);
    assert.ok(items[0].textContent.includes(malicious));
    assert.match(items[1].textContent, /未发现明显问题/);
    assert.doesNotMatch(items[1].textContent, /画面整体偏暗/);
    assert.match(items[2].textContent, /评估未完成/);
    assert.ok(items[2].textContent.includes(`无法读取 ${malicious}`));
    assert.ok(nodes["profile-name"].textContent.includes(malicious));
    assert.doesNotMatch(items.map((item) => item.textContent).join(""), /Filename from the module/);
  } else if (scenario === "unassessed") {
    const nodes = await render(profile);
    assert.match(nodes["quality-summary"].textContent, /未执行图像质量评估/);
    nodes["quality-images"].children.forEach((item) => {
      assert.match(item.textContent, /未评估/);
      assert.doesNotMatch(item.textContent, /未发现明显问题/);
    });
  } else if (scenario === "all_failed") {
    // Failed adapter validation can retain its original data for diagnostics.
    // That data must not produce a false positive quality badge.
    profile.processing_results.push(result([quality("a", "good")], { status: "failed", errors: [{ message: "模块执行失败" }] }));
    const nodes = await render(profile);
    assert.match(nodes["quality-summary"].textContent, /已评估 0 \/ 2/);
    nodes["quality-images"].children.forEach((item) => {
      assert.match(item.textContent, /评估未完成/);
      assert.match(item.textContent, /原始图片与档案仍已保留/);
      assert.doesNotMatch(item.textContent, /未发现明显问题/);
    });
  } else if (scenario === "latest_run") {
    profile.processing_results.push(result([quality("a", "good"), quality("b", "good")]), result([quality("a")]));
    const nodes = await render(profile);
    assert.match(nodes["quality-images"].children[0].textContent, /建议复查/);
    assert.match(nodes["quality-images"].children[1].textContent, /评估未完成/);
  } else if (scenario === "empty_images") {
    profile.images = [];
    const nodes = await render(profile);
    assert.match(nodes["quality-images"].textContent, /暂无可评估/);
  } else if (scenario === "unknown_asset") {
    profile.processing_results.push(result([quality("foreign", "good")]));
    const nodes = await render(profile);
    assert.match(nodes["quality-summary"].textContent, /已评估 0 \/ 2/);
    assert.doesNotMatch(nodes["quality-images"].textContent, /未发现明显问题/);
  } else {
    throw new Error(`Unknown scenario: ${scenario}`);
  }
  console.log(`quality DOM checks passed: ${scenario}`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
