const fileInput = document.querySelector('#file');
const predict = document.querySelector('#predict');
const status = document.querySelector('#status');
const results = document.querySelector('#predictions');
const visuals = document.querySelector('#visuals');
const labels = {glioma: 'Glioma', meningioma: 'Meningioma', notumor: 'Tümör yok', pituitary: 'Hipofiz tümörü'};
const percent = new Intl.NumberFormat('tr-TR', {style: 'percent', maximumFractionDigits: 3});
const deviceLabel = device => device === 'cuda' ? 'CUDA GPU' : 'CPU';
let selection = 0;

async function send(path, file) {
  const mime = file.type || (/\.png$/i.test(file.name) ? 'image/png' : 'image/jpeg');
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type': mime}, body: file});
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail :
      detail?.message ? [detail.message, ...Object.values(detail.errors || {})].join(' ') : 'İstek tamamlanamadı.');
  }
  return data;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showPredictions(data) {
  for (const item of data.predictions) {
    const card = element('article', undefined, 'panel');
    card.append(element('h3', item.model), element('p', labels[item.class], 'top'),
      element('p', 'Cihaz · ' + deviceLabel(item.device)));
    for (const key of data.classes) {
      const row = element('div', undefined, 'score');
      row.append(element('span', labels[key]), element('span', percent.format(item.probabilities[key])));
      const bar = element('progress');
      bar.max = 1;
      bar.value = item.probabilities[key];
      bar.setAttribute('aria-label', labels[key]);
      card.append(row, bar);
    }
    results.append(card);
  }
  for (const [name, error] of Object.entries(data.errors)) {
    results.append(element('p', name + ' · ' + error, 'panel error'));
  }
}

fileInput.addEventListener('change', () => {
  selection++;
  results.replaceChildren();
  visuals.replaceChildren();
  status.textContent = '';
  predict.disabled = !fileInput.files.length;
});

predict.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const current = ++selection;
  predict.disabled = true;
  results.replaceChildren();
  visuals.replaceChildren();
  status.textContent = 'Modeller görüntüyü inceliyor…';
  try {
    const data = await send('/predict', file);
    if (current !== selection) return;
    showPredictions(data);
    showVisuals(file, current, data.preview);
    const fallback = data.predictions.some(item => item.device === 'cpu');
    status.textContent = fallback ? 'Tahminler hazır. CPU kullanan modeller kartlarda belirtiliyor.' : 'Tahminler hazır.';
  } catch (error) {
    if (current === selection) status.textContent = error.message;
  } finally {
    if (current === selection) predict.disabled = false;
  }
});

fetch('/health').then(response => response.json()).then(data => {
  const loaded = data.models.filter(model => model.loaded).length;
  document.querySelector('#health').textContent = deviceLabel(data.device) + ' · ' + loaded + '/' + data.models.length + ' model hazır';
}).catch(() => {
  document.querySelector('#health').textContent = 'Sunucuya erişilemedi';
});

function showVisuals(file, current, preview) {
  const grid = element('div', undefined, 'visual-grid');
  const panels = ['Orijinal MR', 'Swin Grad-CAM', 'Otomatik SAM', 'Manuel SAM'].map(title => {
    const panel = element('article', undefined, 'panel');
    panel.append(element('h3', title));
    grid.append(panel);
    return panel;
  });
  visuals.append(grid, element('p', 'SAM maskeleri kutuya bağlı model çıktılarıdır. Klinik tümör sınırları olarak kullanılmamalıdır.'));
  const [original, cam, auto, manual] = panels;
  const canvas = element('canvas');
  canvas.width = canvas.height = 224;
  canvas.setAttribute('aria-label', 'Manuel SAM kutusu çizim alanı');
  const context = canvas.getContext('2d');
  const image = new Image();

  let start = null;
  let box = null;
  let pointer = null;
  let pending = false;
  const button = element('button', 'Manuel maskeyi üret');
  button.disabled = true;
  const instructions = element('p', 'Kutu çizmek için MR üzerinde sürükleyin.');
  original.append(canvas, instructions, button);
  const camBody = element('div');
  const autoBody = element('div');
  const manualBody = element('div');
  cam.append(camBody);
  auto.append(autoBody);
  manual.append(manualBody);
  manualBody.append(element('p', 'Manuel kutu bekleniyor.'));
  camBody.append(element('p', 'Grad-CAM hesaplanıyor…'));
  autoBody.append(element('p', 'Otomatik kutu hesaplanıyor…'));

  function draw() {
    context.clearRect(0, 0, 224, 224);
    context.drawImage(image, 0, 0, 224, 224);
    if (box) {
      context.strokeStyle = '#71ead8';
      context.lineWidth = 2;
      context.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    }
  }

  image.onload = () => { draw(); };
  image.onerror = () => { instructions.textContent = 'Görüntü önizlemesi açılamadı.'; };
  image.src = preview;

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    const fit = value => Math.max(0, Math.min(223, Math.round(value)));
    return [fit((event.clientX - rect.left) * 224 / rect.width),
      fit((event.clientY - rect.top) * 224 / rect.height)];
  }

  canvas.addEventListener('pointerdown', event => {
    if (pointer !== null || event.button !== 0 || !image.naturalWidth || pending) return;
    pointer = event.pointerId;
    canvas.setPointerCapture(pointer);
    start = point(event);
    box = [...start, ...start];
    button.disabled = true;
    draw();
  });
  function resizeBox(event) {
    const end = point(event);
    box = [Math.min(start[0], end[0]), Math.min(start[1], end[1]),
      Math.max(start[0], end[0]), Math.max(start[1], end[1])];
    draw();
  }
  canvas.addEventListener('pointermove', event => {
    if (pointer === event.pointerId) resizeBox(event);
  });
  canvas.addEventListener('pointerup', event => {
    if (pointer !== event.pointerId) return;
    resizeBox(event);
    pointer = null;
    button.disabled = box[0] === box[2] || box[1] === box[3];
  });
  canvas.addEventListener('pointercancel', event => {
    if (pointer !== event.pointerId) return;
    pointer = null;
    box = null;
    button.disabled = true;
    draw();
  });

  function picture(src, alt) {
    const img = element('img');
    img.src = src;
    img.alt = alt;
    return img;
  }

  function showMask(body, result, title) {
    if (result.status !== 'hazır') {
      body.replaceChildren(element('p', result.message, result.status === 'atlandı' ? '' : 'error'));
      return;
    }
    const maskLink = element('a', 'Maskeyi ayrı aç');
    maskLink.href = result.mask;
    maskLink.download = 'sam-maskesi.png';
    body.replaceChildren(picture(result.overlay, title),
      element('p', 'SAM kalite skoru · ' + result.score.toLocaleString('tr-TR', {maximumFractionDigits: 4})),
      element('p', 'Cihaz · ' + deviceLabel(result.device)), maskLink);
  }

  function choice(title, options) {
    const label = element('label', title + ' ');
    const select = element('select');
    for (const [value, text] of options) {
      const option = element('option', text);
      option.value = value;
      select.append(option);
    }
    label.append(select);
    controls.append(label);
    return select;
  }
  const controls = element('div', undefined, 'segment-options');
  visuals.prepend(controls);
  const method = choice('Otomatik kutu', [['intensity', 'Görüntü işleme'], ['gradcam', 'Grad-CAM']]);
  const stage = choice('Grad-CAM', [['3', 'Stage 3'], ['4', 'Stage 4']]);
  let sequence = 0;
  function runAuto() {
    const request = ++sequence;
    camBody.replaceChildren(element('p', 'Grad-CAM hesaplanıyor…'));
    autoBody.replaceChildren(element('p', 'Otomatik kutu hesaplanıyor…'));
    send('/segment?mode=auto&method=' + method.value + '&stage=' + stage.value, file).then(result => {
        if (current !== selection || request !== sequence) return;
        camBody.replaceChildren(picture(result.gradcam, 'Orijinal MR üzerinde Swin Grad-CAM ısı haritası'),
          element('p', 'Stage ' + result.stage + ' · ' + labels[result.classification] + ' · ' + deviceLabel(result.gradcam_device)));
        showMask(autoBody, result, 'Otomatik SAM maskesi ve kutusu');
        if (result.warning) autoBody.prepend(element('p', result.warning, 'error'));
      }).catch(error => {
        if (current !== selection || request !== sequence) return;
        camBody.replaceChildren(element('p', error.message, 'error'));
        autoBody.replaceChildren(element('p', 'Otomatik segmentasyon çalışmadı. Manuel kutu çizebilirsiniz.'));
      });
  }
  method.addEventListener('change', runAuto);
  stage.addEventListener('change', runAuto);
  runAuto();

  button.addEventListener('click', async () => {
    if (!box || pending) return;
    pending = true;
    button.disabled = true;
    manualBody.replaceChildren(element('p', 'SAM maskesi hesaplanıyor…'));
    try {
      const result = await send('/segment?mode=manual&box=' + box.join(','), file);
      if (current === selection) showMask(manualBody, result, 'Manuel SAM maskesi ve kutusu');
    } catch (error) {
      if (current === selection) manualBody.replaceChildren(element('p', error.message, 'error'));
    } finally {
      pending = false;
      button.disabled = false;
    }
  });
}
