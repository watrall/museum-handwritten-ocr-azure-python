# Handwritten Museum Collections Records — OCR → Structured Data → Data Analysis and Insights (Microsoft Azure & Python)

This tutorial shows you how to convert handwritten museum collection records (scans or PDFs) into structured data and then carry out **data analysis and insights** using **Microsoft Azure**. It is designed for **scholars, museum professionals, graduate students, and advanced undergraduates**. You do **not** need prior coding experience. We proceed carefully and conversationally.

By the end you will:
- Set up and run a **Python notebook** in **Azure Machine Learning**.
- Store your documents securely in **Azure Blob Storage**.
- Convert PDFs to page images and run OCR using **Azure AI Vision (Image Analysis)**.
- Extract structured fields (Accession Number, Object Name, Provenance, etc.).
- Export your dataset to CSV and JSON.
- Perform **data analysis and insights** with tables and charts.

---

## Table of Contents
1. [ What you’ll need ](#what-youll-need)  
2. [ Why Azure ](#why-azure)  
3. [ Meet Azure Notebooks ](#meet-azure-notebooks)  
4. [ Ethics, rights, and responsible handling ](#ethics-rights-and-responsible-handling)  
5. [Create your Azure account and resources](#1-create-your-azure-account-and-resources)  
6. [Upload your documents to Azure Blob Storage](#2-upload-your-documents-to-azure-blob-storage)  
7. [Connect your Notebook to Azure services](#3-connect-your-notebook-to-azure-services)  
8. [Process PDFs and run OCR with Azure AI Vision](#4-process-pdfs-and-run-ocr-with-azure-ai-vision)  
9. [Turn raw text into structured fields](#5-turn-raw-text-into-structured-fields)  
10. [Flag items for human review](#6-flag-items-for-human-review)  
11. [Clean, standardize, and export](#7-clean-standardize-and-export)  
12. [Data analysis and insights](#8-data-analysis-and-insights)  
13. [ Contributing to this tutorial ](#contributing-to-this-tutorial)  
14. [ Troubleshooting ](#troubleshooting)  
15. [ Glossary (plain-English definitions) ](#glossary-plain-english-definitions)

---

## What you’ll need
A Microsoft account and access to the **Azure Portal**. You don’t need to install anything on your personal computer—everything runs in the browser via **Azure Machine Learning** notebooks. You’ll also want a few sample PDFs or images of handwritten records that you have the rights to process.

---

## Why Azure
Think of **Microsoft Azure** as a massive, professional-grade digital workshop. Instead of buying individual tools and bringing them home, you get access to a fully equipped facility with specialized machines and services. For our project, we’ll use Azure for **storage** (Blob Storage), **AI** (Azure AI Vision: Image Analysis for OCR), and **compute** (Azure ML notebooks)—all designed to work smoothly together. This approach is powerful and **scalable**: it works for a handful of records or millions.

**Free options:** Azure offers a **Free Account** with credits and services suitable for this tutorial. Students can use **Azure for Students** (credits, no credit card) if eligible.

---

## Meet Azure Notebooks
Azure ML notebooks are your digital lab bench. You can place explanatory text right next to the code you run and see results—tables, logs, charts—inline. This **visual**, step-by-step environment is perfect for learners who want to understand *what* they’re doing and *why*.

---

## Ethics, rights, and responsible handling
Museum and historical records can include sensitive personal data and culturally restricted knowledge. Work with permission, follow institutional and community guidance, and consider: copyright and use, privacy and sensitivity, cultural patrimony, and bias/limitations in data and models.

---

## 1. Create your Azure account and resources

**What:** Set up the workspace pieces you’ll use together.  
**Why:** Keeps your work organized, secure, and easy to manage.  
**Validate:** You can see all resources in one **Resource Group**.

**Steps (Azure Portal):**
1) **Create an account** (Free or **Azure for Students**).  
2) **Resource Group** (e.g., `MuseumOCR_Project`).  
3) **Storage Account** (e.g., `museumocrstorage123`, region near you, redundancy: Standard_LRS).  
4) **Azure AI Vision – Image Analysis** (free **F0** tier if available).  
5) **Azure Machine Learning Workspace** (same region as above).

> Tip: Use consistent names and the same region to reduce surprises.

---

## 2. Upload your documents to Azure Blob Storage

**What:** Put PDFs/images in a durable, cloud location.  
**Why:** Notebooks’ ephemeral storage resets; Blob Storage is reliable and scalable.  
**Validate:** Files appear in your **container** (e.g., `museum-records`).

**Portal route:** Storage account → **Containers** → **+ Container** (private) → open it → **Upload** your PDFs / images.

---

## 3. Connect your Notebook to Azure services

**What:** Give your notebook secure “keys” to talk to Blob Storage and Image Analysis.  
**Why:** So your code can list files, download them, and call OCR.  
**Validate:** A quick sanity call returns no auth errors.

> In Azure ML, create a new Notebook and run the cells below. Store secrets in environment variables or use **Key Vault** in production.

```python
# Install SDKs
!pip -q install azure-storage-blob azure-ai-vision-imageanalysis pandas matplotlib wordcloud pillow PyMuPDF pdf2image

# Imports
import os, io, re, json
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
```

Authenticate with **keys** (simple for learning) or with **Entra ID** (enterprise). For this tutorial, we’ll use keys:

```python
# Set these securely in your environment (preferred) or paste temporarily (demo only)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "YOUR_BLOB_STORAGE_CONNECTION_STRING")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "https://YOUR-RESOURCE-NAME.cognitiveservices.azure.com")
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY", "YOUR_VISION_KEY")

blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
vision_client = ImageAnalysisClient(endpoint=AZURE_VISION_ENDPOINT, credential=AzureKeyCredential(AZURE_VISION_KEY))

# Quick check: list blobs to confirm access
container_name = "museum-records"  # change if you used a different name
container_client = blob_service.get_container_client(container_name)
print("Sample listing (first 5):", [b.name for _, b in zip(range(5), container_client.list_blobs())])
```

> **Note:** Azure recommends **azure-ai-vision-imageanalysis** for OCR via **VisualFeatures.READ**. For text-heavy PDFs, Microsoft suggests **Document Intelligence (Read)**—but here we’ll convert PDFs to images first and use Image Analysis, which keeps our workflow simple for teaching.

---

## 4. Process PDFs and run OCR with Azure AI Vision

**What:** Convert PDFs to images, then extract text (printed/handwritten).  
**Why:** Image Analysis expects image files; we’ll handle PDFs page-by-page.  
**Validate:** A DataFrame of page-level OCR text (filename, page, text).

We try **PyMuPDF** first (no external system dependency). If unavailable, we fall back to **pdf2image** (requires Poppler on the machine).

```python
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

def pdf_bytes_to_images(pdf_bytes, zoom=2.0):
    """Render each PDF page to a PNG bytes object using PyMuPDF."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        images.append(img_bytes)
    doc.close()
    return images

# OCR helper
def ocr_image_bytes(img_bytes: bytes) -> str:
    result = vision_client.analyze(image_data=img_bytes, visual_features=[VisualFeatures.READ])
    text = []
    if result.read and result.read.blocks:
        for line in result.read.blocks[0].lines:
            text.append(line.text)
    return " ".join(text).strip()

results = []
for blob in container_client.list_blobs():
    bclient = container_client.get_blob_client(blob)
    data = bclient.download_blob().readall()
    if blob.name.lower().endswith(".pdf"):
        pages = pdf_bytes_to_images(data)
        for i, img_b in enumerate(pages, start=1):
            print(f"OCR: {blob.name} page {i}")
            text = ocr_image_bytes(img_b)
            results.append({"source_file": blob.name, "page": i, "raw_ocr_text": text})
    elif blob.name.lower().endswith((".jpg",".jpeg",".png",".tif",".tiff",".bmp",".gif",".webp")):
        print(f"OCR: {blob.name}")
        text = ocr_image_bytes(data)
        results.append({"source_file": blob.name, "page": 1, "raw_ocr_text": text})

df_raw_ocr = pandas.DataFrame(results)
print("Pages OCRed:", len(df_raw_ocr))
df_raw_ocr.head()
```

---

## 5. Turn raw text into structured fields

**What:** Pull fields like Accession Number, Object Name, Provenance into columns.  
**Why:** Structured data can be validated, searched, analyzed, and shared.  
**Validate:** A DataFrame with the new columns filled where patterns match.

```python
import pandas as pd, re

df = df_raw_ocr.copy()
for c in ["accession_number","object_name","provenance","site_location","materials","dimensions","date","notes"]:
    if c not in df.columns: df[c] = ""

FIELD_PATTERNS = {
    "accession_number": r"(accession|acc\.?\s*no\.?|catalog\s*no\.?|cat\.?\s*no\.?)[:\s]*([A-Za-z0-9\-./]+)",
    "object_name":      r"(object\s*name|artifact|item)[:\s]+(.{1,60})",
    "provenance":       r"(provenance|provenience)[:\s]+(.{1,200})",
    "site_location":    r"(site|find\s*spot|location)[:\s]+(.{1,120})",
    "materials":        r"(materials?|medium)[:\s]+(.{1,120})",
    "dimensions":       r"(dimensions?|size)[:\s]+(.{1,120})",
    "date":             r"(date|dated)[:\s]+(.{1,60})",
    "notes":            r"(notes?|remarks?)[:\s]+(.{1,240})",
}

def extract_field(text: str, key: str):
    m = re.search(FIELD_PATTERNS[key], text, flags=re.IGNORECASE)
    return m.group(m.lastindex).strip(" .;:") if m else ""

for idx, row in df.iterrows():
    txt = (row.get("raw_ocr_text") or "").strip()
    for k in FIELD_PATTERNS:
        val = extract_field(txt, k)
        if val: df.at[idx, k] = val

df.head()
```

---

## 6. Flag items for human review

**What:** Mark likely-problem rows (very short text, missing key fields).  
**Why:** Enables a targeted **human-in-the-loop** review queue.  
**Validate:** A `needs_review` column and a filtered view.

```python
def needs_review(row):
    if not row.get("accession_number"): return True
    if isinstance(row.get("raw_ocr_text"), str) and len(row["raw_ocr_text"]) < 50: return True
    return False

df["needs_review"] = df.apply(needs_review, axis=1)
df[df["needs_review"]].head()
```

---

## 7. Clean, standardize, and export

**What:** Tidy whitespace and case; export to **CSV** (spreadsheets) and **JSON** (apps/APIs).  
**Why:** Clean, portable outputs make your data more usable elsewhere.  
**Validate:** Files appear in the notebook’s workspace and download successfully.

```python
import re
def tidy(s):
    if pd.isna(s): return s
    s = str(s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s

for col in ["accession_number","object_name","provenance","site_location","materials","dimensions","date","notes"]:
    if col in df.columns:
        df[col] = df[col].apply(tidy)

if "object_name" in df.columns:
    df["object_name"] = df["object_name"].str.title()

df.to_csv("museum_handwritten_records_azure.csv", index=False)
df.to_json("museum_handwritten_records_azure.json", orient="records", indent=2, force_ascii=False)
"Exported museum_handwritten_records_azure.csv and museum_handwritten_records_azure.json"
```

---

## 8. Data analysis and insights

**What:** Turn the dataset into understanding (counts, timelines, signals, gaps).  
**Why:** OCR is step one—the goal is **data analysis and insights**.  
**Validate:** You see charts/tables that match expectations.

```python
import matplotlib.pyplot as plt
from collections import Counter
import string

def top_counts(series, n=10, title="Top values"):
    vc = series.dropna().astype(str).str.strip().value_counts().head(n)
    display(vc)
    plt.figure()
    vc.sort_values().plot(kind="barh", edgecolor="black")
    plt.title(title); plt.xlabel("Count"); plt.ylabel(series.name or "Value")
    plt.tight_layout(); plt.show()

for col, label in [("materials","Top 10 Materials"),
                   ("object_name","Top 10 Object Names"),
                   ("provenance","Top 10 Provenance Phrases (rough)")]:  # rough without NLP
    if col in df.columns:
        top_counts(df[col], n=10, title=label)

if "date" in df.columns:
    df["year"] = df["date"].astype(str).str.extract(r"(\d{4})")
    year_counts = df["year"].value_counts().sort_index()
    if not year_counts.empty:
        plt.figure(); year_counts.plot(kind="line", marker="o")
        plt.title("Objects by Year"); plt.xlabel("Year"); plt.ylabel("Count")
        plt.tight_layout(); plt.show()

missing = df.isna().sum().sort_values(ascending=True)
plt.figure(); missing.plot(kind="barh", edgecolor="black")
plt.title("Missing Values by Field"); plt.xlabel("Missing count")
plt.tight_layout(); plt.show()
```

---

## Contributing to this tutorial
We welcome improvements: typos, clearer explanations, new analyses. **Fork** this repo, make changes, and open a **Pull Request**. For bigger ideas, start a **Discussion/Issue** to gather feedback first.

---

## Troubleshooting
- **ModuleNotFoundError for ImageAnalysis**: ensure `pip install azure-ai-vision-imageanalysis` and import `ImageAnalysisClient` from `azure.ai.vision.imageanalysis`.  
- **Auth errors**: verify endpoint & key match the **same region** as your Vision resource.  
- **PDF conversion errors**: prefer **PyMuPDF** (no system deps). If using `pdf2image`, install Poppler.  
- **Quota limits**: free tier may limit requests—pause and retry later or choose a paid tier.  
- **Slow OCR**: run on a **compute with GPU** (if available) or batch fewer pages per run.

---

## Glossary (beginner friendly definitions)
- **Blob Storage**: Azure’s durable object store for files like PDFs/images.  
- **Image Analysis (Azure AI Vision)**: Service that provides features like OCR (`VisualFeatures.READ`).  
- **Resource Group**: A folder-like container for related Azure resources.  
- **Notebook**: An interactive document mixing text, code, and results.  
- **DataFrame**: A table of data (rows/columns) in Python via pandas.  
- **CSV / JSON**: Portable file formats for tabular / structured data.  
