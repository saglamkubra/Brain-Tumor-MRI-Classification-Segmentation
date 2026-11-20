"use strict";
const panes = $("viewer-images");
const dialog = $("mr-viewer");
const activePanes = () => [...panes.querySelectorAll(".viewer-viewport")];
const viewerStates = new WeakMap();
let selectedPane, drag, magnifier;

function redrawViewer() {
  const split = $("viewer-split").checked && !$("viewer-split").disabled;
  const division = Number($("viewer-divider").value);
  $("viewer-divider").disabled = !split;
  for (const pane of activePanes()) {
    const {x,y,zoom} = viewerStates.get(pane);
    pane.querySelector(".viewer-content").style.transform =
      `translate(${x * 100}%, ${y * 100}%) scale(${zoom})`;
    const after = pane.querySelector(".viewer-after");
    const line = pane.querySelector(".viewer-line");
    if (after) {
      after.style.clipPath = split ? `inset(0 0 0 ${division}%)` : "";
      line.hidden = !split;
      line.style.left = division + "%";
    }
  }
}

function updateViewer(change) {
  const state = viewerStates.get(selectedPane);
  if (!state) return;
  change(state);
  if ($("viewer-sync").checked)
    for (const pane of activePanes()) Object.assign(viewerStates.get(pane),state);
  redrawViewer();
}

function openViewer(images,title = "") {
  putText($("viewer-title"),title || "MR karşılaştırması");
  status("viewer","");
  panes.replaceChildren();
  panes.classList.toggle("single-view",images.length === 1);
  const aligned = images.length > 1 && visual.original;
  $("viewer-split").disabled = !aligned;
  $("viewer-sync").disabled = !aligned;
  $("viewer-opacity").disabled = !images.some(([,entry]) => entry?.layer);
  for (const [name,result] of images) {
    if (!result?.image) continue;
    const figure = element("figure",null,panes);
    const viewport = element("div",null,figure,"viewer-viewport");
    viewport.tabIndex = 0;
    viewport.setAttribute("aria-label",translatedText(name || title));
    const content = element("div",null,viewport,"viewer-content");
    const insertImage = (parent,src,cls="") => {
      const img = element("img",null,parent,cls);
      img.src = src;
      img.alt = "";
      img.draggable = false;
      img.onerror = () => status("viewer","Görüntü açılamadı.");
      return img;
    };
    if (aligned && result !== visual.original) insertImage(content,visual.original.image);
    const layered = aligned && result !== visual.original;
    const after = layered ? element("div",null,content,"viewer-after") : content;
    insertImage(after,result.layer ? visual.original.image : result.image);
    if (result.layer) {
      const overlay = insertImage(after,result.layer,"viewer-layer");
      overlay.style.opacity = $("viewer-opacity").value / 100;
    }
    if (result.box && result.layer) {
      const [x,y,X,Y] = result.box;
      const bounds = element("div",null,after,"bounds");
      Object.assign(bounds.style,{left:x/2.24+"%",top:y/2.24+"%",width:(X-x)/2.24+"%",height:(Y-y)/2.24+"%"});
    }
    if (layered) {
      const line=element("div",null,content,"viewer-line");
      line.hidden=true;
      line.setAttribute("aria-hidden","true");
    }
    viewerStates.set(viewport,{zoom:1,x:0,y:0});
    viewport.addEventListener("focus",()=>{selectedPane=viewport;});
    if (name) element("figcaption",name,figure);
  }
  selectedPane=activePanes()[0];
  redrawViewer();
  dialog.showModal();
  selectedPane?.focus();
}

function changeZoom(amount) {
  updateViewer(s=>{
    s.zoom=Math.max(1,Math.min(4,s.zoom+amount));
    if (s.zoom === 1) s.x=s.y=0;
  });
}
$("zoom-in").onclick=()=>changeZoom(.25);
$("zoom-out").onclick=()=>changeZoom(-.25);
$("zoom-reset").onclick=()=>{
  for (const pane of activePanes()) Object.assign(viewerStates.get(pane),{x:0,y:0,zoom:1});
  $("viewer-divider").value=50;
  redrawViewer();
  magnifier?.remove();
};
$("viewer-sync").onchange=()=>updateViewer(()=>{});
$("viewer-split").onchange=redrawViewer;
$("viewer-divider").oninput=redrawViewer;
$("viewer-opacity").oninput=()=>{
  for (const overlay of panes.querySelectorAll(".viewer-layer")) overlay.style.opacity=$("viewer-opacity").value/100;
};
$("viewer-full").onclick=async()=>{
  if (document.fullscreenElement===dialog) await document.exitFullscreen();
  else if (dialog.classList.contains("viewport-full")) dialog.classList.remove("viewport-full");
  else {
    try { await dialog.requestFullscreen(); }
    catch { dialog.classList.add("viewport-full"); }
  }
  updateFullscreen();
};
function updateFullscreen() {
  $("viewer-full").setAttribute("aria-pressed",String(document.fullscreenElement===dialog || dialog.classList.contains("viewport-full")));
}
document.addEventListener("fullscreenchange",updateFullscreen);
$("viewer-lens").onclick=()=>{
  const button=$("viewer-lens");
  button.setAttribute("aria-pressed",String(button.getAttribute("aria-pressed")!=="true"));
  magnifier?.remove();
  if (selectedPane && button.getAttribute("aria-pressed")==="true") {
    const rect=selectedPane.getBoundingClientRect();
    showLens(selectedPane,rect.left+rect.width/2,rect.top+rect.height/2);
  }
};
function showLens(pane,clientX,clientY) {
  if ($("viewer-lens").getAttribute("aria-pressed")!=="true") return;
  const source=pane.querySelector(".viewer-content");
  const box=source.getBoundingClientRect();
  if (!box.width || !box.height) return;
  magnifier?.remove();
  magnifier=element("div",null,pane,"viewer-magnifier");
  const view=pane.getBoundingClientRect();
  Object.assign(magnifier.style,{left:(clientX-view.left)+"px",top:(clientY-view.top)+"px"});
  const copy=source.cloneNode(true);
  copy.style.width=2*box.width+"px";
  copy.style.transform=`translate(${59-2*(clientX-box.left)}px,${59-2*(clientY-box.top)}px)`;
  magnifier.append(copy);
}
panes.addEventListener("keydown",evt=>{
  if (evt.key==="+" || evt.key==="=" || evt.key==="-") {
    evt.preventDefault();
    changeZoom(evt.key==="-" ? -.25 : .25);
    return;
  }
  const d={ArrowLeft:[-.08,0],ArrowRight:[.08,0],ArrowUp:[0,-.08],ArrowDown:[0,.08]}[evt.key];
  if (!d || !selectedPane || viewerStates.get(selectedPane).zoom<=1) return;
  evt.preventDefault();
  updateViewer(s=>{s.x+=d[0];s.y+=d[1];});
});
panes.addEventListener("pointerdown",evt=>{
  const pane=evt.target.closest(".viewer-viewport");
  if (!pane || evt.button!==0) return;
  selectedPane=pane;
  pane.focus({preventScroll:true});
  if (!evt.target.closest(".viewer-line") && viewerStates.get(pane).zoom<=1) return;
  drag={x:evt.clientX,y:evt.clientY,split:Boolean(evt.target.closest(".viewer-line"))};
  panes.setPointerCapture(evt.pointerId);
});
panes.addEventListener("pointermove",evt=>{
  const pane=selectedPane;
  if (drag && pane) {
    const frame=pane.querySelector(".viewer-content").getBoundingClientRect();
    if (drag.split) {
      $("viewer-divider").value=Math.round(100*(evt.clientX-frame.left)/frame.width);
      redrawViewer();
    } else {
      updateViewer(s=>{s.x+=(evt.clientX-drag.x)/pane.clientWidth; s.y+=(evt.clientY-drag.y)/pane.clientHeight;});
      drag.x=evt.clientX;
      drag.y=evt.clientY;
    }
  }
  const hovered=evt.target.closest(".viewer-viewport") || (drag && pane);
  if (hovered) showLens(hovered,evt.clientX,evt.clientY);
  else magnifier?.remove();
});
panes.addEventListener("pointerleave",()=>magnifier?.remove());
for(const name of ["pointerup","pointercancel","lostpointercapture"])
  panes.addEventListener(name,()=>{drag=null;});
dialog.addEventListener("close",()=>{
  if (document.fullscreenElement===dialog) void document.exitFullscreen();
  dialog.classList.remove("viewport-full");
  updateFullscreen();
  panes.replaceChildren();
  magnifier?.remove();
});

const lab=$("explanation-lab");
const labCanvas=$("lab-canvas");
const labContext=labCanvas.getContext("2d");
const labButtons=[...lab.querySelectorAll("[data-mode]")];
const labEnable = ids => { for (const button of ids) button.disabled=!image || (button.dataset.mode==="random" && !interventionQuery); };
let inputImages=[], labAnchor, labBox, lastIntervention, randomControl, interventionQuery;
function paintLab() {
  if (!preview?.complete || !preview.naturalWidth) return;
  labContext.drawImage(preview,0,0,224,224);
  if (labBox) {
    const [x,y,X,Y]=labBox;
    labContext.strokeStyle="#ffe278";
    labContext.lineWidth=2;
    labContext.strokeRect(x,y,X-x,Y-y);
  }
}
function resetExplanation() {
  inputImages=[];
  lastIntervention=randomControl=interventionQuery=null;
  labAnchor=labBox=null;
  $("lab-box").value="";
  $("lab-results").replaceChildren();
  $("lab-altered").removeAttribute("src");
  $("lab-altered").hidden=true;
  labCanvas.hidden=!image;
  labEnable([$("lab-inputs"),...labButtons]);
  for (const key of ["lab-input","lab-result"]) status(key,"");
  if (image) {
    const current=generation;
    void preview.decode().then(()=>{if(current===generation) paintLab();}).catch(()=>{});
  }
  renderPipeline();
}
function renderPipeline() {
  const grid=$("lab-pipeline");
  grid.replaceChildren();
  if (!analysis) {
    element("p","Önce bir MR görüntüsü yükleyin.",grid);
    return;
  }
  const step=(title,description,source)=> {
    const card=element("article",null,grid,"panel lab-step");
    element("h4",title,card);
    if (source) {
      const img=element("img",null,card);
      img.src=source;
      img.alt=translatedText(title);
    }
    element("p",description,card);
    return card;
  };
  step("Orijinal MR","Önizleme, görüntünün en-boy oranını korur.",visual.original?.image);
  step("CNN girişi","224 × 224 gri tensör, gösterim için 0–255 aralığına ölçeklendi.",inputImages.find(i=>i.model==="CNN")?.image);
  step("Swin girişi","224 × 224 RGB tensör, gösterim için 0–255 aralığına ölçeklendi.",inputImages.find(i=>i.model==="Swin-Tiny")?.image);
  const scores=step("Model skorları","Her modelin özgün MR üzerinde seçtiği sınıf ve softmax skoru gösterilir.");
  for (const p of analysis.predictions)
    element("p",p.model+" · "+translatedText(labels[p.class])+" · "+pct(p.probabilities[p.class]),scores);
  const cam=$("stage").value==="4" ? visual.gradcam4 : visual.gradcam;
  step("Grad-CAM","Seçili aşamanın sınıf kararına katkı haritası.",cam?.image);
  const boxStep=step("Kutu","Mevcut otomatik SAM kutusu önizleme koordinatlarındadır.",candidateBox ? visual.original.image : null);
  if (candidateBox) {
    const frame=element("div",null,boxStep,"frame");
    frame.append(boxStep.querySelector("img"));
    const line=element("div",null,frame,"bounds");
    const [x,y,X,Y]=candidateBox;
    Object.assign(line.style,{left:x/2.24+"%",top:y/2.24+"%",width:(X-x)/2.24+"%",height:(Y-y)/2.24+"%"});
    element("p",candidateBox.join(", "),boxStep);
  }
  step("SAM","Son otomatik SAM çıktısı, varsa aday maskeyi gösterir.",visual.auto?.layer ? visual.auto.image : null);
}
lab.addEventListener("toggle",()=>{if(lab.open) renderPipeline();});
$("stage").addEventListener("change",()=>{if(lab.open) renderPipeline();});
$("language").addEventListener("change",()=>{if(lab.open) renderPipeline(); if(lastIntervention) showIntervention(lastIntervention);});
$("lab-inputs").onclick=async()=>{
  const current=generation;
  $("lab-inputs").disabled=true;
  status("lab-input","İşleniyor…");
  try {
    const result=await postImage("/model-inputs",image);
    if (current!==generation) return;
    inputImages=result.inputs;
    renderPipeline();
    status("lab-input","");
  } catch(error) {
    if(current===generation) status("lab-input",error.message);
  } finally {
    if(current===generation) labEnable([$("lab-inputs")]);
  }
};
function labPoint(evt) {
  const rect=labCanvas.getBoundingClientRect();
  return [Math.max(0,Math.min(224,Math.round(224*(evt.clientX-rect.left)/rect.width))),
    Math.max(0,Math.min(224,Math.round(224*(evt.clientY-rect.top)/rect.height)))];
}
labCanvas.addEventListener("pointerdown",evt=>{
  if(!image || evt.button!==0 || !evt.isPrimary) return;
  labCanvas.setPointerCapture(evt.pointerId);
  labAnchor=labPoint(evt);
  labBox=null;
  $("lab-box").value="";
  paintLab();
});
labCanvas.addEventListener("pointermove",evt=>{
  if(!labAnchor) return;
  const [x,y]=labPoint(evt);
  labBox=[Math.min(x,labAnchor[0]),Math.min(y,labAnchor[1]),Math.max(x,labAnchor[0]),Math.max(y,labAnchor[1])];
  $("lab-box").value=labBox.join(",");
  paintLab();
});
for(const event of ["pointerup","pointercancel","lostpointercapture"])
  labCanvas.addEventListener(event,()=>{labAnchor=null;});
$("lab-box").addEventListener("input",()=>{
  const list=$("lab-box").value.split(",").map(Number);
  labBox=list.length===4 && list.every(Number.isFinite) ? list : null;
  paintLab();
});
function showIntervention(data) {
  $("lab-altered").src=data.preview;
  $("lab-altered").hidden=false;
  $("lab-altered").alt=translatedText("Dolgu uygulanmış MR");
  const grid=$("lab-results");
  grid.replaceChildren();
  const fraction=element("p",null,grid,"lab-area");
  fraction.textContent=translatedText("Gizlenen alan")+ " · "+pct(data.pixels/data.total);
  const cards=element("div",null,grid,"lab-grid");
  for(const p of data.results) {
    const card=element("article",null,cards,"panel");
    element("h4",p.model,card);
    const target=element("p",null,card);
    target.textContent=translatedText("Özgün hedef sınıf")+" · "+translatedText(labels[p.target]);
    const control=randomControl?.results.find(m=>m.model===p.model);
    const columns=element("div",null,card,"lab-compare");
    for (const [entry,heading] of [[p,"Asıl müdahale"],[control,"Rastgele kontrol"]]) {
    if (!entry) continue;
    const column=element("div",null,columns);
    element("h4",heading,column);
    const outcome=element("p",null,column);
    outcome.textContent=translatedText("Müdahaleden sonra en yüksek sınıf")+" · "+translatedText(labels[entry.after_class]);
    for(const [title,values,format,limit] of [
      ["Softmax skoru",entry.scores,predictionPct,1],
      ["Logit marjı",entry.margins,n=>n.toLocaleString(document.documentElement.lang,{maximumFractionDigits:2}),Math.max(1,...p.margins.map(Math.abs),...(control?.margins || []).map(Math.abs))]
    ]) {
      element("h4",title,column);
      values.forEach((value,index)=>{
        const row=element("div",null,column,"lab-bar-row");
        element("span",index ? "Sonra" : "Önce",row);
        const meter=element("meter",null,row);
        meter.min=title==="Logit marjı" ? -limit : 0;
        meter.max=limit;
        meter.value=value;
        meter.setAttribute("aria-label",translatedText(title)+" · "+translatedText(index ? "Sonra" : "Önce"));
        element("strong",format(value),row);
      });
    }
    }
  }
}
for(const button of labButtons) button.onclick=async()=>{
  const current=generation;
  let query=new URLSearchParams({mode:button.dataset.mode,fill:$("lab-fill").value,stage:$("stage").value});
  if(button.dataset.mode==="box") query.set("box",$("lab-box").value);
  if(button.dataset.mode==="random") {
    if(!interventionQuery) return;
    query=new URLSearchParams(interventionQuery);
    query.set("mode","random");
  }
  labButtons.forEach(b=>b.disabled=true);
  status("lab-result","İşleniyor…");
  try {
    const data=await postImage("/intervene?"+query,image);
    if(current!==generation) return;
    if(button.dataset.mode==="random") {
      if(data.pixels!==lastIntervention.pixels) throw Error(translatedText("Kontrol alanı müdahaleyle eşleşmedi."));
      randomControl=data;
    } else {
      lastIntervention=data;
      interventionQuery=query.toString();
      randomControl=null;
    }
    showIntervention(lastIntervention);
    status("lab-result","");
  } catch(error) {
    if(current===generation) status("lab-result",error.message);
  } finally {
    if(current===generation) labEnable(labButtons);
  }
};
