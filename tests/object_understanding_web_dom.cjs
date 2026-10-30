"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor(tagName) {
    this.tagName = tagName; this.children = []; this.listeners = {}; this.attributes = {};
    this.value = ""; this.hidden = false; this._text = ""; this.classes = new Set();
    this.classList = { add: (v) => this.classes.add(v), remove: (v) => this.classes.delete(v),
      toggle: (v, enabled) => enabled ? this.classes.add(v) : this.classes.delete(v) };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((item) => item.textContent).join(""); }
  set innerHTML(_) { throw new Error("Unsafe dynamic HTML"); }
  set outerHTML(_) { throw new Error("Unsafe dynamic HTML"); }
  insertAdjacentHTML() { throw new Error("Unsafe dynamic HTML"); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this._text = ""; this.children = nodes; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  focus() {}
  scrollIntoView() {}
}

const malicious = '<img src=x onerror="globalThis.pwned=true">';
const asset = (asset_id, filename = malicious) => ({ asset_id, filename, width: 800, height: 600,
  mime_type: "image/png", size_bytes: 300 });
const fact = (text, asset_ids = ["a"], status = "observed") => ({ text, evidence: `可见依据 ${malicious}`, asset_ids, status });
const material = (name, ids = ["a"]) => ({ material_name: name, evidence: "只能根据表面外观提出候选", asset_ids: ids, status: "candidate", confidence: 0.95 });
const observation = (id) => ({ asset_id: id, object_name: fact("杯形物体", [id], "candidate"),
  category_candidates: [{ category: "杯具", asset_ids: [id], evidence: "可见开口与容器形状", status: "candidate" }],
  possible_uses: [fact("可能用于盛装物品", [id], "candidate")],
  appearance: { colors: [fact("白色", [id])], shape_description: fact("近圆柱形", [id]),
    surface_description: fact("有反光", [id]), visible_features: [fact("可见把手", [id])] },
  material_candidates: [material("陶瓷候选", [id])], visible_components: [{ component_name: "把手", description: "弯曲凸起",
    evidence: "侧边可见", asset_ids: [id], status: "observed" }],
  viewpoint: { viewpoint: "perspective", evidence: "同时可见侧面与顶部", asset_ids: [id], status: "candidate" },
  observations: [], unknown_information: ["内部材料无法由照片确认"], warnings: [malicious],
});
const result = (data, status = "partial") => ({ module_name: "object_understanding", status, data, warnings: [], errors: [] });

async function render(profile, config = {}) {
  const html = fs.readFileSync(path.join(__dirname, "../app/web/index.html"), "utf8");
  const nodes = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => [match[1], new Element("div")]));
  const context = {
    document: { getElementById: (id) => nodes[id], createElement: (tag) => new Element(tag) },
    window: { setTimeout, clearTimeout, matchMedia: () => ({ matches: true }) },
    URL: { createObjectURL: () => "blob:test-only", revokeObjectURL() {} },
    Image: class { constructor() { this.naturalWidth = 800; this.naturalHeight = 600; }
      set src(_) { queueMicrotask(() => this.onload?.()); } },
    FormData: class { append() {} },
    fetch: async (url) => url === "/api/v1/config" ? { ok: true, json: async () => config }
      : { ok: true, status: 201, json: async () => ({ object_id: "test-object", status: "completed_with_warnings", profile }) },
    console,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../app/web/app.js"), "utf8"), context);
  await new Promise(setImmediate);
  nodes["object-name"].value = "明确为测试用档案";
  await nodes["image-input"].listeners.change({ target: { files: [{ name: "test.png", type: "image/png", size: 300, lastModified: 1 }] } });
  await nodes["upload-form"].listeners.submit({ preventDefault() {} });
  assert.equal(nodes["form-error"].textContent, "");
  assert.equal(context.pwned, undefined);
  return nodes;
}

async function main() {
  const scenario = process.argv[2];
  const profile = { name: "仅测试结构", images: [asset("a"), asset("b", "另一视角.png")], processing_results: [], warnings: [] };
  const data = { image_observations: [observation("a"), observation("b")],
    image_states: [{ asset_id: "a", status: "success", message: "" }, { asset_id: "b", status: "success", message: "" }],
    fusion_status: "success", object_category: { category: "杯具候选", asset_ids: ["a", "b"], evidence: "两图可见容器形状", status: "candidate" },
    appearance_summary: [fact("白色带把手容器", ["a", "b"])], material_candidates: [material("陶瓷候选", ["a", "b"])],
    components: [], cross_view_consensus: [fact("可见把手", ["a", "b"])], conflicting_observations: [], unknown_information: ["真实尺寸未知"] };
  if (scenario === "sourced_candidates") {
    profile.processing_results.push(result(data));
    const nodes = await render(profile);
    const text = nodes["understanding-results"].textContent;
    assert.match(nodes["understanding-summary"].textContent, /2 \/ 2/);
    assert.match(text, /候选：杯具候选/);
    assert.match(text, /材料候选（未经材质验证）/);
    assert.match(text, /来源：/);
    assert.match(text, /另一视角\.png \[b\]/);
    assert.ok(text.includes(malicious));
    assert.doesNotMatch(text, /已确认|confirmed/);
    assert.match(text, /真实尺寸未知/);
  } else if (scenario === "conflicts_and_partial") {
    data.fusion_status = "failed";
    data.image_states[1] = { asset_id: "b", status: "partial", message: "材料任务失败，其他观察保留" };
    data.conflicting_observations = [{ field: "颜色", description: "可能受到不同照明影响，无法选定",
      claims: [fact("白色", ["a"]), fact("灰色", ["b"])] }];
    profile.processing_results.push(result(data));
    const nodes = await render(profile);
    assert.match(nodes["understanding-summary"].textContent, /融合未完成，单图结果仍然保留/);
    assert.match(nodes["understanding-results"].textContent, /待核对冲突：颜色/);
    assert.match(nodes["understanding-results"].textContent, /无法选定/);
    assert.match(nodes["understanding-results"].textContent, /部分任务完成/);
    assert.match(nodes["understanding-results"].textContent, /材料任务失败/);
  } else if (scenario === "old_and_disabled") {
    const old = await render(profile, { recognition_enabled: false });
    assert.match(old["understanding-summary"].textContent, /未执行物体理解/);
    assert.match(old["recognition-config-note"].textContent, /当前未启用/);
    profile.processing_results.push(result({ image_states: [{ asset_id: "a", status: "disabled" }] }, "skipped"));
    const disabled = await render(profile);
    assert.match(disabled["understanding-summary"].textContent, /未完成或未启用/);
    assert.doesNotMatch(disabled["understanding-results"].textContent, /观察已完成|杯具候选/);
  } else if (scenario === "failed_latest") {
    profile.processing_results.push(result(data), result({ ...data, object_category: { category: "不可信失败数据", asset_ids: ["a"] } }, "failed"));
    const nodes = await render(profile);
    assert.match(nodes["understanding-summary"].textContent, /未完成或未启用/);
    assert.doesNotMatch(nodes["understanding-results"].textContent, /不可信失败数据|杯具候选|白色带把手容器/);
  } else if (scenario === "foreign_sources") {
    data.object_category.asset_ids = ["foreign"];
    data.image_observations = [observation("foreign")];
    data.material_candidates = [material("外部材料不应显示", ["foreign"])];
    data.appearance_summary = []; data.cross_view_consensus = [];
    profile.processing_results.push(result(data));
    const nodes = await render(profile);
    assert.match(nodes["understanding-summary"].textContent, /0 \/ 2/);
    assert.doesNotMatch(nodes["understanding-results"].textContent, /杯具候选|外部材料不应显示|可见把手/);
  } else if (scenario === "privacy_config") {
    const nodes = await render(profile, { recognition_enabled: true, recognition_model: malicious, recognition_max_images: 3 });
    const text = nodes["recognition-config-note"].textContent;
    assert.ok(text.includes(malicious));
    assert.match(text, /照片及已校验的观察信息会发送/);
    assert.match(text, /最多分析 3 张/);
    assert.match(text, /费用/);
  } else throw new Error(`Unknown test scenario: ${scenario}`);
  console.log(`understanding DOM checks passed: ${scenario}`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
