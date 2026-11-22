from base64 import b64decode
from binascii import Error
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
import re

from .language import message

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

FONT_PATHS = (
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)
LABELS = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "Tümör yok", "pituitary": "Hipofiz tümörü"}
PREFIX = "data:image/png;base64,"


def format_probability(value, language):
    if value >= 0.99995 and value != 1:
        return message("%99,99 ve üzeri", language)
    if value < 0.00005:
        return message("%0,01 altı", language)
    number = "100" if value == 1 else f"{value*100:.2f}"
    return number + "%" if language == "en" else "%" + number.replace(".", ",")


def report(data, language="tr", swin_stage3_layer=None):
    analysis = data.get("analysis")
    if not isinstance(analysis, dict) or not isinstance(analysis.get("predictions"), list) or not analysis["predictions"]:
        raise ValueError("PDF için bir MR analizi gerekli.")
    if "NeuroSans" not in pdfmetrics.getRegisteredFontNames():
        font = next((path for path in FONT_PATHS if path.is_file()), None)
        if font is None:
            raise ValueError("PDF için Türkçe karakter destekli sistem yazı tipi bulunamadı.")
        pdfmetrics.registerFont(TTFont("NeuroSans", str(font)))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "NeuroSans"
        style.textColor = colors.HexColor("#19253a")
    content = []

    def add(value, style="BodyText"):
        content.append(Paragraph(escape(message(str(value), language)[:1800]), styles[style]))

    add("NEURO AI LAB", "Title")
    add("Beyin Tümörü Analiz Sistemi · Beyin MR analiz raporu", "Heading1")
    add(analysis.get("timestamp", ""))
    add("Araştırma ve eğitim amaçlıdır. Klinik teşhis için kullanılamaz.")
    add("Model tahminleri", "Heading2")
    add("Model skorları (softmax)")
    add("Softmax çıktısıdır; kalibre edilmiş klinik olasılık değildir.")
    for item in analysis["predictions"][:2]:
        add(f'{item["model"]} · {message(LABELS.get(item["class"], item["class"]), language)} · {item["device"]} · {item["inference_ms"]:.1f} ms')
        add(f'{item.get("profile", "")} · {item["checkpoint"]}')
        add("   |   ".join(f"{message(name, language)} {format_probability(float(item['probabilities'][key]), language)}" for key, name in LABELS.items()))
    predictions = analysis["predictions"]
    if len(predictions) == 2:
        agreement = "Aynı sınıf tahmin edildi." if predictions[0]["class"] == predictions[1]["class"] else "Modeller farklı sınıflandırma sonuçları üretmiştir."
        add(message("Model uyumu · ", language) + message(agreement, language))
    for name, error_text in analysis.get("errors", {}).items():
        add(f"{name} · {message(error_text, language)}")
    add("Görsel analiz", "Heading2")
    if swin_stage3_layer:
        add(f"{message('Aşama 3', language)} Grad-CAM · {swin_stage3_layer}")
    prefixes = {"original": "Orijinal MR", "gradcam": "Grad-CAM aşama 3", "gradcam4": "Grad-CAM aşama 4", "cnn_cam": "CNN Grad-CAM", "auto": "Otomatik SAM", "manual": "Manuel SAM"}
    for key, title in prefixes.items():
        item = data.get("visuals", {}).get(key)
        if not item:
            continue
        note = item.get("note")
        translated_title = message(title, language)
        add(f"{translated_title} · {note}" if note else translated_title, "Heading3")
        raw = item.get("image")
        if raw:
            if not isinstance(raw, str) or not raw.startswith(PREFIX) or len(raw) > 800_000:
                raise ValueError("Rapor görseli formatı veya boyutu geçersiz.")
            try:
                source = BytesIO(b64decode(raw[len(PREFIX):], validate=True))
                with Image.open(source) as image:
                    if image.format != "PNG" or max(image.size) > 512:
                        raise ValueError("Rapor görseli en fazla 512 piksel PNG olmalı.")
                    image.verify()
                source.seek(0)
            except (OSError, Error) as exc:
                raise ValueError("Rapor görseli okunamadı.") from exc
            heading = content.pop()
            content.append(KeepTogether([heading, PDFImage(source, width=140, height=140, hAlign="LEFT"), Spacer(1, 8)]))
    add("SAM maskeleri kutuya bağlı model çıktısıdır. Tümör sınırı doğrulaması yapılmamıştır.")
    add("PubMed kaynakları", "Heading2")
    references = data.get("references", [])[:12]
    if not references:
        add("Bu analize PubMed kaynağı eklenmedi.")
    for item in references:
        uid = str(item.get("id", ""))
        if not re.fullmatch(r"[0-9]{1,12}", uid):
            continue
        add(f'{item.get("title", "")} · {item.get("journal", "")} · {item.get("date", "")}')
        if item.get("doi"):
            add(f'DOI {item["doi"]}')
        link = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
        content.append(Paragraph(f'<link href="{link}">{link}</link>', styles["BodyText"]))
    buffer = BytesIO()
    SimpleDocTemplate(buffer, title="NEURO AI LAB | Beyin MR analiz raporu").build(content)
    return buffer.getvalue()
