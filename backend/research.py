"""Retrieve real bibliographic records via NCBI E-utilities."""
from xml.etree import ElementTree
import json
from urllib.parse import urlencode
from urllib.request import urlopen

TOPICS = {
    "glioma": "glioma AND (MRI OR magnetic resonance imaging)",
    "meningioma": "meningioma AND (MRI OR magnetic resonance imaging)",
    "notumor": "(normal brain OR healthy controls) AND (brain MRI OR neuroimaging)",
    "pituitary": "(pituitary adenoma OR pituitary tumor) AND (MRI OR magnetic resonance imaging)",
}
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def request(endpoint, **params):
    url = BASE + endpoint + "?" + urlencode(dict(db="pubmed", retmode="json", **params))
    with urlopen(url, timeout=12) as response:
        return json.load(response)


def search_term(label, query):
    term = "(" + TOPICS[label] + ") AND (intracranial OR brain OR sellar)"
    if query:
        term += " AND (" + query + ")"
    return "(" + term + ") NOT (dog OR canine OR feline OR veterinary OR spinal)"


def research(label, query, years, order):
    params = {"term": search_term(label, query), "sort": order, "retmax": 12}
    if years:
        params.update(datetype="pdat", reldate=years * 365)
    ids = request("esearch.fcgi", **params)["esearchresult"]["idlist"]
    if not ids:
        return []
    summaries = request("esummary.fcgi", id=",".join(ids))["result"]
    return [
        {"id": uid, "title": row["title"], "date": row.get("pubdate", ""),
         "doi": next((item["value"] for item in row.get("articleids", []) if item.get("idtype") == "doi"), ""),
         "journal": row.get("fulljournalname", ""), "authors": [a["name"] for a in row.get("authors", [])],
         "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"}
        for uid in ids if (row := summaries.get(uid)) and "title" in row
    ]




def abstract(pmid):
    params = urlencode({"db": "pubmed", "id": pmid, "retmode": "xml"})
    with urlopen(BASE + "efetch.fcgi?" + params, timeout=12) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("PubMed yanıtı çok büyük.")
    root = ElementTree.fromstring(payload)
    return [{"label": part.get("Label", ""), "text": "".join(part.itertext()).strip()}
            for part in root.findall(".//Article/Abstract/AbstractText") if "".join(part.itertext()).strip()]
