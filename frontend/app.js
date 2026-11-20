"use strict";
const $ = id => document.getElementById(id);
let messages = {};
const localeReady = fetch("/assets/messages.json").then(response => response.json()).then(data => { messages = data; }).catch(() => {});
function translatedText(text) {
  if (document.documentElement.lang !== "en" || typeof text !== "string") return text;
  if (Object.hasOwn(messages,text)) return messages[text];
  const prefix = Object.keys(messages).find(key => key.endsWith(" ") && text.startsWith(key));
  return prefix ? messages[prefix] + text.slice(prefix.length) : text;
}
const translatable = new WeakMap();
function putText(node, text) {
  translatable.set(node,text);
  node.textContent = translatedText(text);
}
const labels = {glioma:"Glioma",meningioma:"Meningioma",notumor:"Tümör yok",pituitary:"Hipofiz tümörü"};
function pct(value, {prediction=false, digits=2, minimum=0}={}) {
  if (prediction && value >= 0.99995) return translatedText("%99,99 ve üzeri");
  if (prediction && value < 0.00005) return translatedText("%0,01 altı");
  const lang=document.documentElement.lang;
  const number=(value*100).toLocaleString(lang,{maximumFractionDigits:digits,minimumFractionDigits:minimum});
  return lang==="en" ? number+"%" : "%"+number;
}
const predictionPct = value => pct(value,{prediction:true});
const status = (id, message) => { putText($(id + "-status"),message); };
const element = (tag, value, parent, className) => {
  const node = document.createElement(tag);
  if (value !== null) putText(node,value);
  if (className) node.className = className;
  parent.append(node);
  return node;
};
async function request(url, options) {
  await localeReady;
  const headers = new Headers(options?.headers);
  headers.set("Accept-Language",document.documentElement.lang);
  const response = await fetch(url,{...options,headers});
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw Error(typeof error.detail === "string" ? error.detail : error.detail?.message || "İstek tamamlanamadı.");
  }
  return response;
}
const postImage = async (url, image) => (await request(url,{
  method:"POST",headers:{"Content-Type":image.type},body:image
})).json();

let generation = 0, searchVersion = 0, image, preview, analysis, box, anchor, articles = [];
let visual = {};
let prompts = [];
let candidateBox = null;
const performanceByCheckpoint = new Map();
const canvas = $("drawing"), ctx = canvas.getContext("2d");
function paint() {
  ctx.clearRect(0,0,224,224);
  if (preview?.complete && preview.naturalWidth) ctx.drawImage(preview,0,0,224,224);
  $("manual-empty").hidden = Boolean(box && box[2]-box[0]>=4 && box[3]-box[1]>=4);
  if (!box) return;
  ctx.strokeStyle="#ffdb72"; ctx.fillStyle="#ffdb7233"; ctx.lineWidth=2;
  ctx.fillRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);
  ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);
  for (const [x,y,label] of prompts) {
    ctx.fillStyle = label ? "#40f5b4" : "#ffb499";
    ctx.beginPath(); ctx.arc(x,y,5,0,2*Math.PI); ctx.fill();
    ctx.fillStyle="#101522"; ctx.font="bold 8px sans-serif";
    ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(label ? "+" : "−",x,y);
  }
}
function clear() {
  generation++; searchVersion++;
  $("stage").disabled = false;
  $("open-viewer").disabled = true;
  showCamStages();
  prompts = []; $("prompt-mode").value = "box";
  candidateBox = null; $("use-candidate").hidden = true;
  image = preview = analysis = box = anchor = null; articles = []; visual = {};
  $("cnn-cam").disabled = true;
  $("cnn-cam-panel").hidden = true;
  $("cnn-cam-image").removeAttribute("src");
  $("test-context").hidden = true;
  $("scores").replaceChildren(); $("articles").replaceChildren();
  for (const key of ["analysis","cam","auto","manual","research","pdf"]) status(key,"");
  ["cam-base","cam-image","cam4-base","cam4-image","auto-base","manual-base"].forEach(id => $(id).removeAttribute("src"));
  ["auto","manual"].forEach(key => {
    $(key+"-image").classList.remove("ready"); $(key+"-box").hidden=true;
    $(key+"-image").getContext("2d").clearRect(0,0,224,224);
  });
  ["cam","auto","manual","pdf","intensity"].forEach(id => { $(id).disabled=true; delete $(id).dataset.busy; });
  updateOpacity();
  paint();
  resetExplanation();
}
function setPreview(data) {
  preview = new Image();
  preview.onload = paint;
  preview.src = data;
  ["cam","cam4","auto","manual"].forEach(key => {$(key+"-base").src=data;});
  $("cnn-cam-base").src = data;
  visual.original = {image:data};
  $("open-viewer").disabled = false;
  resetExplanation();
}
function showPredictions(data) {
  $("scores").replaceChildren();
  for (const p of data.predictions) {
    const card = element("article",null,$("scores"),"panel");
    element("h3",p.model,card);
    element("small",(p.profile || "") + " · " + p.checkpoint,card);
    element("p","En olası sınıf · " + translatedText(labels[p.class]),card,"winner");
    const facts = element("div",null,card,"model-facts");
    const predicted = p.probabilities[p.class];
    element("span","Bu görüntü için model skoru " + predictionPct(predicted),facts);
    const test = element("span","Test doğruluğu yükleniyor…",facts);
    test.dataset.checkpoint = p.checkpoint;
    test.dataset.model = p.model;
    test.dataset.profile = p.profile || "";
    updateAccuracy(test);
    element("small","İşlem süresi " + p.inference_ms.toFixed(1) + " ms · " + (p.device === "cuda" ? "GPU" : "CPU"),card);
    element("h4","Model skorları (softmax)",card);
    element("small","Softmax çıktısıdır; kalibre edilmiş klinik olasılık değildir.",card);
    for (const [name, score] of Object.entries(p.probabilities)) {
      const line = element("div",null,card,"score");
      element("span",labels[name],line); element("strong",predictionPct(score),line);
      const bar = element("div",null,card,"bar");
      element("span",null,bar).style.width = (score * 100) + "%";
    }
  }
  refreshAccuracies();
}
async function analyze(file) {
  clear();
  const current = generation;
  status("analysis","Modeller çalışıyor…");
  try {
    if (current !== generation) return;
    if (!file || !["image/png","image/jpeg"].includes(file.type) || file.size > 10*1024*1024) {
      throw Error("En fazla 10 MiB boyutunda bir PNG veya JPG seçin.");
    }
    const data = await postImage("/predict",file);
    if (current !== generation) return;
    image = file; analysis = data;
    setPreview(data.preview); showPredictions(data);
    recordHistory(data);
    $("cnn-cam").disabled = !data.predictions.some(item => item.model === "CNN");
    const messages = Object.values(data.errors);
    if (data.predictions.length === 2) {
      messages.unshift(data.predictions[0].class === data.predictions[1].class ?
        "Modeller aynı sınıfı seçti." : "Modeller farklı sınıflandırma sonuçları üretmiştir.");
    }
    if (data.predictions.some(model => model.class === "notumor")) {
      messages.push("«Tümör yok» bir model tahminidir. Görüntüde bir bölgeyi incelemek için kontrastla aday arayabilir veya manuel kutu çizebilirsiniz.");
    }
    status("analysis",messages.map(translatedText).join(" "));
    const swin = data.predictions.find(p => p.model === "Swin-Tiny");
    $("pdf").disabled=false;
    $("intensity").disabled = !swin;
    if (swin) {
      $("cam").disabled = $("auto").disabled = false;
      void generate("auto");
    }
    $("topic").value = (swin || data.predictions[0]).class;
    const disagree = data.predictions.length === 2 && data.predictions[0].class !== data.predictions[1].class;
    if (disagree) status("research","Modeller farklı sınıf seçti. Sınıfı belirleyip Ara düğmesine basın.");
    else $("research").requestSubmit();
  } catch(error) { if (current === generation) status("analysis",error.message); }
}
$("file").addEventListener("change", () => {
  const file = $("file").files[0];
  if (file) void analyze(file);
});
function locationOnImage(event) {
  const bounds = canvas.getBoundingClientRect();
  const bounded = (n,size) => Math.min(223,Math.max(0,Math.round(224*n/size)));
  return [bounded(event.clientX-bounds.left,bounds.width),bounded(event.clientY-bounds.top,bounds.height)];
}
canvas.addEventListener("pointerdown", e => {
  if (!image || !e.isPrimary || e.button !== 0 || $("manual").dataset.busy) return;
  const mode = $("prompt-mode").value;
  if (mode !== "box") {
    const [x,y] = locationOnImage(e);
    if (!box || $("manual").disabled) {
      status("manual","Önce bir kutu çizin.");
      return;
    }
    if (mode === "1" && (x < box[0] || x > box[2] || y < box[1] || y > box[3])) {
      status("manual","Dahil etme noktasını kutunun içine koyun.");
      return;
    }
    prompts = prompts.filter(([px,py]) => px !== x || py !== y);
    if (prompts.length >= 12) {
      status("manual","En fazla 12 nokta eklenebilir.");
      return;
    }
    prompts.push([x,y,Number(mode)]);
    status("manual","İşaretler güncellendi. Maskeyi yeniden üretin.");
    paint();
    return;
  }
  prompts = [];
  canvas.setPointerCapture(e.pointerId);
  anchor = locationOnImage(e); box = null; $("manual").disabled=true; paint();
});
canvas.addEventListener("pointermove", e => {
  if (!anchor || !e.isPrimary) return;
  const [x,y] = locationOnImage(e);
  box = [Math.min(x,anchor[0]),Math.min(y,anchor[1]),Math.max(x,anchor[0]),Math.max(y,anchor[1])]; paint();
});
function finishBox(e) {
  if (!e.isPrimary) return;
  if (!anchor) return;
  anchor = null;
  $("manual").disabled = !box || box[2]-box[0]<4 || box[3]-box[1]<4;
  status("manual",$("manual").disabled ? "Kutu geçerli değil." : "Kutuyla maskeyi üretmeye hazır.");
}
$("clear-points").addEventListener("click", () => {
  prompts = []; paint();
});
$("clear-box").addEventListener("click", () => {
  if ($("manual").dataset.busy) return;
  box = null; prompts = []; $("manual").disabled = true; paint();
  status("manual","Yeni kutu çizebilirsiniz.");
});
canvas.addEventListener("keydown", event => {
  const direction = {ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[event.key];
  if (!direction || !box || $("manual").disabled) return;
  event.preventDefault();
  const [dx,dy] = direction, [x,y,X,Y] = box, step = 3;
  const next = event.shiftKey ? [x,y,X+dx*step,Y+dy*step] :
    [x+dx*step,y+dy*step,X+dx*step,Y+dy*step];
  if (next[0] < 0 || next[1] < 0 || next[2] > 223 || next[3] > 223 ||
      next[2]-next[0] < 4 || next[3]-next[1] < 4) return;
  box = next;
  prompts = prompts.filter(([px,py,label]) => !label ||
    (px >= box[0] && px <= box[2] && py >= box[1] && py <= box[3]));
  paint();
  status("manual","Kutu düzenlendi. Maskeyi yeniden üretin.");
});
canvas.addEventListener("pointerup",finishBox);
canvas.addEventListener("pointercancel",finishBox);
function camResult(data, stage) {
  const imageId = Number(stage) === 4 ? "cam4-image" : "cam-image";
  $(imageId).src = data.gradcam_layer || data.gradcam || data.overlay;
  const note =  "Grad-CAM · " + translatedText("Aşama") + " " + stage + " · " + translatedText(labels[data.classification]) + " · " + (data.gradcam_device || data.device);
  visual[Number(stage) === 4 ? "gradcam4" : "gradcam"] = {image:data.gradcam || data.overlay,layer:data.gradcam_layer,note};
  status("cam",note);
  updateOpacity();
}
let nextCam = 0;
function showCamStages(compare = false) {
  const stage = $("stage").value;
  $("cam-stage-3").hidden = !compare && stage !== "3";
  $("cam-stage-4").hidden = !compare && stage !== "4";
  $("cam-pair").classList.toggle("compare", compare);
  $("cam").setAttribute("aria-pressed", String(compare));
  $("cam").textContent = compare ? translatedText("Tek aşamaya dön") :
    translatedText("Aşama 3 ve 4 karşılaştırması");
  updateOpacity();
}
$("use-candidate").addEventListener("click", () => {
  if (!candidateBox || $("manual").dataset.busy) return;
  box = [...candidateBox]; prompts = [];
  $("prompt-mode").value = "box";
  $("manual").disabled = false;
  status("manual","Aday kutu aktarıldı. Noktalarla düzenleyebilir veya yeni bir kutu çizebilirsiniz.");
  paint();
});
$("intensity").addEventListener("click", () => void generate("auto","intensity"));
async function generate(kind, method = "gradcam", compare = false) {
  if (!image || (kind === "manual" && !box)) return;
  const current = generation, stage = $("stage").value;
  const camTicket = kind !== "manual" ? ++nextCam : nextCam;
  const button = $(kind);
  button.disabled=true;
  if (kind !== "manual") $("stage").disabled = true;
  if (kind === "cam") $("auto").disabled = $("intensity").disabled = true;
  if (kind === "auto") {
    showCamStages();
    $("cam").disabled = true;
    $("intensity").disabled = true;
    candidateBox = null;
    $("use-candidate").hidden = true;
  }
  if (kind === "manual") button.dataset.busy = "true";
  status(kind,"İşleniyor…");
  if (kind !== "cam") {
    $(kind+"-image").classList.remove("ready"); $(kind+"-box").hidden=true;
    delete visual[kind];
    updateOpacity();
  }
  const params = new URLSearchParams({mode:kind,stage,method});
  if (kind === "auto") params.set("allow_fallback",String($("allow-fallback").checked));
  if (kind === "manual") {
    params.set("box",box.join(","));
    if (prompts.length) params.set("points",prompts.map(point => point.join(",")).join(";"));
  }
  const endpoint = "/segment?" + params;
  try {
    if (kind === "cam") {
      showCamStages(compare);
      for (const layer of compare ? [3,4] : [Number(stage)]) {
        if (current !== generation) return;
        const map = await postImage("/explain?stage="+layer,image);
        if (current === generation) camResult(map,layer);
      }
      if (current === generation && compare) status("cam","Grad-CAM · Aşama 3 ve 4 hazır.");
      return;
    }
    const data = await postImage(endpoint,image);
    if (current !== generation) return;
    if (camTicket === nextCam && data.gradcam) camResult(data,stage);
    if (kind !== "cam") {
      const summary = data.message || (method === "intensity" ?
        "Kontrastla bulunan aday bölge işaretlendi. Bu yöntem parlak dokuları da seçebilir." :
        "Aday bölge işaretlendi." + (kind === "auto" ? " Grad-CAM aşama " + stage + " kullanıldı." : ""));
      const source = {
        gradcam_selected: "Kutu kaynağı · Grad-CAM aşama " + data.box_stage + ".",
        gradcam_agreement: "Kutu kaynağı · Grad-CAM aşama 3 ve 4’ün ortak odağı.",
        gradcam_other_stage: "Kutu kaynağı · Grad-CAM aşama " + data.box_stage +
          "."
      }[data.box_source];
      const note = [source ? data.message || "Aday bölge işaretlendi." : summary, data.box_reason, source, data.box_warning, data.warning].filter(Boolean).map(translatedText).join(" ");
      if (kind === "auto" && data.box) {
        candidateBox = data.box;
        $("use-candidate").hidden = false;
      }
      visual[kind] = {note,image:data.overlay,box:data.box};
      if (data.box) {
        const bounds = $(kind+"-box");
        const [x,y,X,Y] = data.box;
        Object.assign(bounds.style,{left:x/2.24+"%",top:y/2.24+"%",width:(X-x)/2.24+"%",height:(Y-y)/2.24+"%"});
        bounds.hidden = false;
      }
      if (data.mask) {
        const layer = $(kind+"-image");
        const mask = new Image();
        mask.src = data.mask;
        await mask.decode();
        if (current !== generation) return;
        const context = layer.getContext("2d");
        context.clearRect(0,0,224,224);
        context.drawImage(mask,0,0,224,224);
        const pixels = context.getImageData(0,0,224,224);
        for (let i=0; i<pixels.data.length; i+=4) {
          const alpha = pixels.data[i];
          pixels.data.set([64,245,180,alpha],i);
        }
        context.putImageData(pixels,0,0);
        visual[kind].layer = layer.toDataURL("image/png");
        layer.classList.add("ready");
        updateOpacity();
      }
      status(kind,kind==="manual" && !data.mask && !data.message ? "Maske üretilemedi." : note);
    }
  } catch(error) { if (current === generation) status(kind,error.message); }
  finally {
    if (current === generation) {
      button.disabled=false; delete button.dataset.busy;
      if ($("explanation-lab").open) renderPipeline();
      if (kind !== "manual") $("stage").disabled = false;
      if (kind !== "manual") {
        $("auto").disabled = false;
        $("intensity").disabled = false;
        $("cam").disabled = false;
      }
    }
  }
}
["auto","manual"].forEach(kind => $(kind).addEventListener("click", () => void generate(kind)));
$("cam").addEventListener("click", () => {
  if ($("cam").getAttribute("aria-pressed") === "true") {
    showCamStages();
    const chosen = $("stage").value === "4" ? visual.gradcam4 : visual.gradcam;
    status("cam",chosen?.note || "");
    return;
  }
  void generate("cam","gradcam",true);
});
$("stage").addEventListener("change", () => {
  showCamStages();
  if (image && !$("auto").disabled) void generate("auto");
});
function updateOpacity(){
  const camIds = ["cam-image","cam4-image"];
  const samIds = ["auto-image","manual-image"];
  for (const [kind, ids, ready] of [
    ["cam",camIds,id => $(id).hasAttribute("src") && !$(id === "cam-image" ? "cam-stage-3" : "cam-stage-4").hidden],
    ["sam",samIds,id => $(id).classList.contains("ready")]
  ]) {
    const input = $(kind+"-opacity");
    input.disabled = !ids.some(ready) && !(kind === "cam" && $("cnn-cam-image").hasAttribute("src"));
    $(kind+"-level").textContent = "%" + input.value;
    for (const id of ids) $(id).style.opacity = input.value / 100;
    if (kind === "cam") $("cnn-cam-image").style.opacity = input.value / 100;
  }
}
for (const kind of ["cam","sam"]) $(kind+"-opacity").addEventListener("input",updateOpacity);
updateOpacity();

$("research").addEventListener("submit",async e => {
  e.preventDefault();
  const version = ++searchVersion;
  articles = []; $("articles").replaceChildren(); status("research","PubMed taranıyor…");
  const params = new URLSearchParams({label:$("topic").value,q:$("query").value,years:$("years").value,sort:$("sort").value});
  try {
    const data = await (await request("/research?"+params)).json();
    if (version !== searchVersion) return;
    articles = data.articles;
    status("research",articles.length ? articles.length + " PubMed kaydı" : "Bu aramada yayın bulunamadı.");
    for (const article of articles) {
      const row = element("li",null,$("articles"));
      const link = element("a",article.title,row); link.href=article.url; link.target="_blank"; link.rel="noopener noreferrer";
      element("div",[article.date,article.journal,article.authors.join(", "), "PMID " + article.id, article.doi ? "DOI " + article.doi : ""].filter(Boolean).join(" · "),row,"muted");
      const detail=element("details",null,row);
      element("summary","Özet / Abstract",detail);
      detail.addEventListener("toggle",async () => {
        if (!detail.open || detail.dataset.loaded) return;
        detail.dataset.loaded = "true";
        const text=element("p","PubMed özeti yükleniyor…",detail);
        try {
          const data=await (await request("/research/"+encodeURIComponent(article.id))).json();
          text.textContent=data.sections.length ? data.sections.map(section=>[section.label,section.text].filter(Boolean).join(" · ")).join("\n\n") :
            "Bu kayıt için PubMed özeti yok.";
        } catch(error) { text.textContent=error.message; }
      });
    }
  } catch(error) { if (version === searchVersion) status("research",error.message); }
});
function table(parent, headers, rows, diagonal = false) {
  const html = element("table",null,element("div",null,parent,"metric-table"));
  const head = element("tr",null,element("thead",null,html));
  headers.forEach(label=>{element("th",label,head).scope="col";});
  const body = element("tbody",null,html);
  rows.forEach(([title,...cells],i) => {
    const row=element("tr",null,body); element("th",title,row).scope="row";
    cells.forEach((value,j) => element("td",value,row,diagonal && i===j ? "hit" : ""));
  });
}
const swinAccuracy = {
  legacy: {checkpoint:"swin_tiny_best.pth",accuracy:1492/1600,label:"Test kümesi doğruluğu · İlk test (1.600 görüntü) "},
  v3: {checkpoint:"swin_tiny_v3_best.pth"},
  "v3-finetuned": {checkpoint:"swin_tiny_v3_finetuned_best.pth",accuracy:1485/1579,
                  label:"Test kümesi doğruluğu · Yerel V3 ince ayar "}
};
let performanceLoaded = false;
function updateAccuracy(node) {
  const {model, profile, checkpoint} = node.dataset;
  const known = model === "Swin-Tiny" ? swinAccuracy[profile] : null;
  const matches = Boolean(known && known.checkpoint === checkpoint);
  const accuracy = matches && known.accuracy !== undefined ? known.accuracy :
    (matches || model === "CNN" && profile === "v2") ? performanceByCheckpoint.get(checkpoint) : undefined;
  putText(node, accuracy !== undefined ? (known?.label || "Test kümesi doğruluğu ") + pct(accuracy) :
    performanceLoaded ? "Bu profil için doğrulanmış metrik yok" : "Test doğruluğu yükleniyor…");
}
function refreshAccuracies() {
  const cards = [...$("scores").querySelectorAll("[data-checkpoint]")];
  cards.forEach(updateAccuracy);
  $("test-context").hidden = !performanceLoaded ||
    !cards.some(({dataset:d}) => d.model === "CNN" && d.profile === "v2" && performanceByCheckpoint.has(d.checkpoint)) ||
    !cards.some(({dataset:d}) => d.model === "Swin-Tiny" && ["v3","v3-finetuned"].includes(d.profile) &&
      swinAccuracy[d.profile].checkpoint === d.checkpoint &&
      (swinAccuracy[d.profile].accuracy !== undefined || performanceByCheckpoint.has(d.checkpoint)));
}
let performanceData, performanceHealth;
async function init() {
  const health = await (await request("/health")).json().catch(() => null);
  if (!health) throw Error("Model durumu okunamadı.");
  $("health").textContent = health.models.filter(model=>model.loaded).length + "/" + health.models.length +
    " model hazır · " + health.device.toUpperCase();
  putText($("stage3-target"), "Aşama 3 · " + health.swin_stage3_layer);
  putText($("stage3-explanation"), health.swin_stage3_target === "stage_output" ?
    "Varsayılan hedef aşamanın tüm çıkışıdır; not defteri grafikleri son bloğun norm1 çıkışını kullandığı için haritalar farklı olabilir." :
    "Aşama 3 Grad-CAM şu anda son bloğun norm1 çıktısını kullanıyor. Notebook grafikleri de bu katmanı kullanır. Uygulama haritaya çıkarım sırasında yumuşatma uygular.");
  const activeSwin = health.models.find(model => model.name === "Swin-Tiny" && model.loaded);
  if (activeSwin) $("health").textContent += " · Swin: " + health.swin_profile + " · " + activeSwin.checkpoint;
  const cnn = health.models.find(model => model.name === "CNN");
  if (cnn?.error?.includes("CNN paketi bulunamadı")) {
    $("health").textContent += " · " + translatedText("CNN paketi bulunamadı");
  }
  if (health.cnn_integrity?.source === "pinned_local") {
    $("health").textContent += " · " + translatedText("CNN: SHA-256 doğrulamalı yerel ağırlık; dağıtım paketi yok.");
  }
  $("health").title = health.models.filter(model=>model.error).map(model=>model.error).join(" ");
  const data = await (await request("/performance")).json();
  performanceLoaded = true;
  for (const model of data.models) performanceByCheckpoint.set(model.checkpoint,model.metrics.accuracy);
  refreshAccuracies();
  status("performance",data.counts.join(" / ") + " eğitim / doğrulama / test · V3 · Hasta düzeyinde bağımsızlık doğrulanamadı.");
  performanceData=data; performanceHealth=health;
  renderPerformance();
}
function renderPerformance() {
  if (!performanceData) return;
  const data=performanceData, health=performanceHealth;
  $("comparison").replaceChildren();
  for (const m of data.models) {
    const card=element("article",null,$("comparison"),"panel"), metric=m.metrics;
    element("h3",m.name,card);
    const active = health.models.find(model=>model.name===m.name)?.checkpoint;
    element("p",active === m.checkpoint ? "Canlı analizde seçili" : "Kayıtlı deney",
      card,"muted");
    element("p",m.parameters.toLocaleString("tr-TR") + " parametre · " +
      (m.checkpoint_bytes/1048576).toFixed(1) + " MiB checkpoint",card,"muted");
    const macro=metric["macro avg"];
    for (const [name,value] of [["Doğruluk",metric.accuracy],["Ortalama kesinlik",macro.precision],
      ["Ortalama sınıf yakalama oranı",macro.recall],["Makro F1",macro["f1-score"]],["Ağırlıklı F1",metric["weighted avg"]["f1-score"]]]) {
      const line=element("div",null,card,"score");
      element("span",name,line);element("strong",pct(value),line);
      element("span",null,element("div",null,card,"bar")).style.width=(value*100)+"%";
    }
    table(card,["Sınıf","Kesinlik","Yakalama oranı","F1"],data.classes.map(key => {
      const values=metric[key];return [labels[key],pct(values.precision),pct(values.recall),pct(values["f1-score"])];
    }));
    const noTumor=data.classes.indexOf("notumor");
    const tumorRows=["glioma","meningioma","pituitary"].map(label=>m.matrix[data.classes.indexOf(label)]);
    const missed=tumorRows.reduce((sum,row)=>sum+row[noTumor],0);
    const total=tumorRows.reduce((sum,row)=>sum+row.reduce((a,b)=>a+b,0),0);
    const miss=element("div",null,card,"panel missed-tumor");
    element("h4","Tümör olup ‘tümör yok’ denen oran",miss);
    element("strong",`${missed}/${total} · ${pct(missed/total,{digits:1,minimum:1})}`,miss);
    element("p","Matristen türetilmiştir; aynı 1.579 görüntülük test kümesi; hasta düzeyinde bağımsızlık doğrulanmamıştır.",miss,"muted");
    element("p","Karışıklık matrisi · Satır gerçek sınıf / sütun tahmin",card);
    table(card,["Sınıf",...data.classes.map(key=>labels[key])],m.matrix.map((row,i)=>[labels[data.classes[i]],...row]),true);
  }
}
void init().catch(error => { performanceLoaded = true; refreshAccuracies(); status("performance",error.message); });
$("pdf").addEventListener("click",async () => {
  const current = generation;
  $("pdf").disabled=true; status("pdf","PDF hazırlanıyor…");
  try {
    const summary = {timestamp:analysis.timestamp, predictions:analysis.predictions, errors:analysis.errors};
    const reportImages = $("pdf-images").checked ? Object.fromEntries(
      Object.entries(visual).map(([key,entry]) => [key,{image:entry.image,note:entry.note}])) : {};
    const result = await request("/report",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({analysis:summary,visuals:reportImages,references:articles})});
    const blob = await result.blob();
    if (current !== generation) return;
    const url=URL.createObjectURL(blob), link=document.createElement("a");
    link.href=url;link.download="NEURO_AI_LAB_rapor.pdf";link.click();
    setTimeout(()=>URL.revokeObjectURL(url),30000);status("pdf","PDF oluşturuldu.");
  } catch(error) { if(current===generation) status("pdf",error.message); }
  finally { if(current===generation) $("pdf").disabled=false; }
});

$("open-viewer").addEventListener("click", () => {
  if (!visual.original) return;
  const compare = $("cam").getAttribute("aria-pressed") === "true";
  openViewer([["Orijinal MR", visual.original], ["Grad-CAM · Aşama 3", (compare || $("stage").value === "3") && visual.gradcam],
    ["Grad-CAM · Aşama 4", (compare || $("stage").value === "4") && visual.gradcam4],
    ["CNN Grad-CAM",visual.cnn_cam], ["Otomatik SAM",visual.auto], ["Manuel SAM",visual.manual]].filter(([,entry])=>entry?.image));
});

$("cnn-cam").addEventListener("click", async () => {
  if (!image) return;
  const current = generation;
  $("cnn-cam").disabled = true;
  status("cam","CNN Grad-CAM hesaplanıyor…");
  try {
    const map = await postImage("/explain?model=cnn", image);
    if (current !== generation) return;
    $("cnn-cam-image").src = map.gradcam_layer;
    $("cnn-cam-panel").hidden = false;
    visual.cnn_cam = {image:map.overlay,layer:map.gradcam_layer,note:"CNN · " + labels[map.classification]};
    status("cam",visual.cnn_cam.note + " · features.3.block.3");
    updateOpacity();
  } catch (error) {
    if (current === generation) status("cam",error.message);
  } finally {
    if (current === generation) $("cnn-cam").disabled = false;
  }
});

function recordHistory(data) {
  const text = data.predictions.map(item => item.model + " · " + labels[item.class]).join(" / ");
  const entry = document.createElement("li");
  entry.textContent = new Date(data.timestamp).toLocaleString(document.documentElement.lang) + " — " + text;
  const history = $("history");
  history.prepend(entry);
  if (history.children.length > 20) history.lastElementChild.remove();
}
$("clear-history").addEventListener("click", () => $("history").replaceChildren());

const galleryGroups = {
  cnn: ["CNN_V2 sonuçlar", [
    ["03_model_architecture_analysis.png","Model mimarisi"],
    ["04_accuracy_loss_curves.png","Eğitim ve doğrulama eğrileri"],
    ["07_confusion_matrix.png","Karışıklık matrisi"],
    ["09_class_performance_metrics.png","Sınıf performansı"],
    ["12_roc_curve_analysis.png","ROC eğrileri"],
    ["14_auc_ap_comparison.png","AUC ve ortalama kesinlik karşılaştırması"],
    ["23_gradcam_class_comparison.png","Sınıf bazında Grad-CAM"],
    ["28_final_performance_dashboard.png","Değerlendirme panosu"]
  ], "CNN V2 · Notebook deney çıktıları"],
  swin: ["swin_v3", [
    ["colab_t4_ek_analiz/03_swin_v3_model_architecture_analysis.png","Model mimarisi","Swin-Tiny V3 · Colab T4 mimari kaydı"],
    ["finetuning/local_run/training_curves.png","Eğitim ve doğrulama eğrileri","Swin-Tiny V3 · Yerel RTX 3050 ince ayar kaydı"],
    ["swin_sam_grafikleri/04_karisiklik_matrisi_ve_fark.png","Karışıklık matrisleri"],
    ["swin_sam_grafikleri/03_sinif_bazinda_precision_recall_f1.png","Sınıf performansı"],
    ["swin_sam_grafikleri/06_roc_auc.png","ROC eğrileri ve AUC"],
    ["swin_sam_grafikleri/07_precision_recall.png","Kesinlik–duyarlılık eğrileri ve ortalama kesinlik"],
    ["swin_sam_grafikleri/10_sam_ve_gradcam_dort_sinif.png","Sınıf bazında Grad-CAM","Başlangıç Swin V3 · Dört sınıftan kayıtlı MR örnekleri ve SAM çıktıları"],
    ["swin_sam_grafikleri/01_swin_v3_deney_karsilastirmasi.png","Değerlendirme panosu","Swin V3 · Başlangıç, yerel RTX 3050 ve Colab T4 deney özeti"],
    ["swin_sam_grafikleri/02_glioma_hatalari.png","Glioma hataları","Swin V3 · Üç deneyde 383 glioma görüntüsünün tahmin dağılımı"]
  ], "1.579 görüntülük V3 test kümesi · Başlangıç Swin ve yerel ince ayar"],
  cam: ["swin_v3/swin_sam_grafikleri", [
    ["10_sam_ve_gradcam_dort_sinif.png","Dört sınıfta Grad-CAM ve SAM"]
  ], "Swin V3 ve SAM ViT-B · Kayıtlı örnekler"],
  sam: ["swin_v3/swin_sam_grafikleri", [
    ["11_sam_otomatik_manuel_kutu.png","Otomatik ve manuel SAM"],
    ["12_sam_maske_alani_ve_kalite_skoru.png","Maske alanı ve model kalite tahmini"]
  ], "SAM ViT-B · Referans maske karşılaştırması yapılmadı"]
};
function gallerySelection() {
  const [path, figures, description] = galleryGroups[$("gallery-filter").value];
  const figure = figures.find(([file]) => file === $("figure-select").value) || figures[0];
  return {title:figure[1], url:"/results/"+path.split("/").concat(figure[0].split("/")).map(encodeURIComponent).join("/"), description:figure[2] || description};
}
function changeGalleryFilter() {
  const select = $("figure-select");
  select.replaceChildren();
  for (const [file,title] of galleryGroups[$("gallery-filter").value][1]) {
    const option = element("option",title,select);
    option.value = file;
  }
  updateFigureLink();
}
function updateFigureLink() {
  const {url,description,title} = gallerySelection();
  $("figure-download").href = url;
  putText($("figure-description"),description);
  const image=$("figure-preview");
  image.alt=translatedText(title);
  const current=new URL(url,location.href).href;
  if (image.src===current) return;
  image.hidden=true;
  putText($("figure-state"),"Grafik yükleniyor…");
  image.onload=()=>{
    image.hidden=false;
    putText($("figure-state"),"");
  };
  image.onerror=()=>putText($("figure-state"),"Grafik yüklenemedi.");
  image.src=url;
}
$("gallery-filter").addEventListener("change",changeGalleryFilter);
$("figure-select").addEventListener("change",updateFigureLink);
$("figure-open").addEventListener("click", () => {
  const {url,title} = gallerySelection();
  openViewer([["", {image:url}]],title);
});
changeGalleryFilter();

for (const node of document.querySelectorAll("[data-i18n]")) translatable.set(node,node.textContent.trim());
for (const id of ["language", "theme"]) {
  try {
    const saved = localStorage.getItem("neuro-" + id);
    if (saved && [...$(id).options].some(option => option.value === saved)) $(id).value = saved;
  } catch {}
  $(id).addEventListener("change",applyPreferences);
}
function applyPreferences() {
  const language = $("language").value;
  document.documentElement.lang = language;
  document.documentElement.dataset.theme = $("theme").value;
  for (const node of document.querySelectorAll("[data-i18n], [role=status], #scores *, #comparison *, #figure-select option, #figure-description, #figure-state")) {
    if (translatable.has(node)) node.textContent = translatedText(translatable.get(node));
  }
  if (analysis) showPredictions(analysis);
  renderPerformance();
  refreshAccuracies();
  updateFigureLink();
  showCamStages($("cam").getAttribute("aria-pressed") === "true");
  try {
    localStorage.setItem("neuro-language",language);
    localStorage.setItem("neuro-theme",$("theme").value);
  } catch {}
}
void localeReady.then(applyPreferences);
