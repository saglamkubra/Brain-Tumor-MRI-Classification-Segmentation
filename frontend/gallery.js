(() => {
  const controls = document.querySelector('.gallery-controls');
  const display = document.querySelector('#gallery-case');
  const names = {meningioma: 'Meningioma', pituitary: 'Hipofiz tümörü', notumor: 'Tümör yok', glioma: 'Glioma'};
  const percent = new Intl.NumberFormat('tr-TR', {style: 'percent', maximumFractionDigits: 3});
  const decimal = new Intl.NumberFormat('tr-TR', {maximumFractionDigits: 4});
  const formatScore = score => score > 0 && score < 0.00001 ? '< %0,001' : percent.format(score);

  function make(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function picture(title, path, caption) {
    const figure = make('figure', undefined, 'gallery-figure');
    figure.append(make('h4', title));
    if (path) {
      const img = make('img');
      img.src = path;
      img.alt = title;
      img.decoding = 'async';
      figure.append(img);
    }
    if (caption) figure.append(make('figcaption', caption));
    return figure;
  }

  function predictionCard(item, example) {
    const card = make('article', undefined, 'gallery-prediction');
    const heading = make('div', undefined, 'gallery-prediction-head');
    const correct = item.class === example.label;
    heading.append(make('h4', item.model),
      make('span', correct ? 'Doğru' : 'Yanlış', correct ? 'gallery-match' : 'gallery-miss'));
    card.append(heading, make('p', names[item.class] + ' · ' + formatScore(item.probabilities[item.class]), 'gallery-top'));
    const details = make('details');
    details.append(make('summary', 'Dört sınıfın olasılıkları'));
    const scores = make('dl');
    for (const key of example.class_order) {
      const row = make('div');
      row.append(make('dt', names[key]), make('dd', formatScore(item.probabilities[key])));
      scores.append(row);
    }
    details.append(scores);
    card.append(details);
    return card;
  }

  function segmentPanel(title, result, message) {
    if (!result?.image) {
      const figure = picture(title, null, result?.message || message);
      figure.classList.add('gallery-unavailable');
      return figure;
    }
    const figure = picture(title, result.image, 'SAM kalite skoru · ' + decimal.format(result.score));
    const link = make('a', 'İkili maskeyi aç');
    link.href = result.mask;
    figure.append(link);
    return figure;
  }

  function show(example) {
    const head = make('div', undefined, 'gallery-case-head');
    head.append(make('h3', names[example.label]), make('p', 'Test etiketi · ' + names[example.label]),
      make('small', example.source));
    const predictions = make('div', undefined, 'gallery-predictions');
    predictions.append(...example.predictions.map(item => predictionCard(item, example)));
    const visuals = make('div', undefined, 'gallery-visuals');
    visuals.append(picture('Orijinal MR', example.original),
      picture('Swin Grad-CAM', example.gradcam, 'Hedef sınıf · ' + names[example.gradcam_class]),
      segmentPanel('Otomatik SAM', example.auto, 'Otomatik SAM sonucu bulunmuyor.'),
      segmentPanel('Manuel SAM', example.manual, 'Bu örnek için kayıtlı manuel kutu bulunmuyor.'));
    display.replaceChildren(head, predictions);
    if (example.label === 'glioma') {
      display.append(make('p', 'Bu glioma örneğinde CNN ve Swin-Tiny test etiketinden farklı sınıflar seçti. Karşılaştırma yalnızca bu tek görüntüye aittir.', 'gallery-note'));
    }
    display.append(visuals, make('p', 'SAM maskeleri uzman tarafından doğrulanmış tümör sınırları değildir.', 'gallery-foot'));
  }

  fetch('/assets/gallery/cases.json').then(response => {
    if (!response.ok) throw new Error();
    return response.json();
  }).then(examples => {
    const buttons = examples.map(example => {
      const button = make('button', names[example.label]);
      button.type = 'button';
      button.setAttribute('aria-controls', display.id);
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => {
        for (const other of buttons) other.setAttribute('aria-pressed', String(other === button));
        show(example);
      });
      controls.append(button);
      return button;
    });
    buttons[0].click();
  }).catch(() => display.replaceChildren(make('p', 'Deney sonuçları yüklenemedi.', 'error')));
})();
