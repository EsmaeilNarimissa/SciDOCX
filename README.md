[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17665758.svg)](https://doi.org/10.5281/zenodo.17665758)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![Docker GPU Ready](https://img.shields.io/badge/docker-GPU%20Ready-blue.svg)
![CUDA](https://img.shields.io/badge/cuda-12.1+-brightgreen.svg)

**Version 1.0 (2025-11)**  
Tested on Python 3.10 / CUDA 12.1

# SciDOCX: Scientific Document Conversion and MM-RAG Pipeline

A high-fidelity, GPU-accelerated system for scientific PDFs that performs **two complementary tasks**:

1. **Document Conversion:** Converts PDFs — including equations, tables, and figures — into clean, editable **Markdown** and **Word (DOCX)** files with full layout and semantic preservation.
2. **Multimodal RAG Preparation:** Converts the same PDFs into structured **JSONL** datasets optimized for **multimodal Retrieval-Augmented Generation (MM-RAG)** pipelines, optionally enriched with **Qwen2-VL-2B-Instruct** for accurate, context-aware scientific figure descriptions.

### At a Glance

| Use Case                      | Output                                                                    | Typical Consumer                                |
| ----------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------- |
| Editing, review, or archiving | Markdown and DOCX with equations, tables, and figures                     | Researchers, authors, reviewers                 |
| Building RAG datasets         | Structured JSONL with figures, tables, and text (optionally VLM-enriched) | Data scientists, AI researchers, RAG developers |

### Dual Pipeline Overview

```
PDF
 ├──► DeepSeek-OCR
 │     ├──► Markdown → Pandoc → DOCX        (Document Conversion)
 │     └──► JSONL Elements → Qwen2-VL-2B   (MM-RAG Enrichment)
 └──► Figures, Tables, Equations preserved throughout
```

### Key Features

* **Accurate OCR:** Powered by **DeepSeek-OCR** with GPU acceleration for near human-level text recognition.
* **Scientific Awareness:** Preserves LaTeX-style equations, complex tables, and figure captions with full layout integrity.
* **Dual Output:**
  * Markdown and DOCX for editing and publishing.
  * JSONL elements (paragraphs, tables, figures) for multimodal retrieval.
* **VLM Enrichment:** Integrates **Qwen2-VL-2B-Instruct** for precise, factual, and watermark-free figure descriptions that improve RAG recall.
* **Embedded Figures:** Extracts, deduplicates, and embeds figures with correct captions in both DOCX and JSONL outputs.
* **End-to-End Automation:** Processes entire folders of PDFs with a single command.
* **Cross-Platform Execution:** Runs via **Docker** (GPU) or locally using **Python/UV**, with automatic GPU/CPU fallback.

> Need a retrieval dataset instead of a Word file?
> Run `pdf_to_mmrag_json.py` to produce JSONL output for your RAG system.


## Quick Start Guide

### Prerequisites

* **Docker** with NVIDIA Container Toolkit (for GPU execution)
  *or*
  **Python 3.8+** with **UV** installed locally
* **Git**
* (Optional) GPU with ≥6 GB VRAM for Qwen2-VL-2B-Instruct

### 1. Clone the Repository

```bash
git clone https://github.com/EsmaeilNarimissa/SciDOCX.git
cd SciDOCX
```

### 2. Download Model Weights

Models are downloaded automatically on first run.
To pre-download manually:

```bash
pip install huggingface_hub

# DeepSeek-OCR (text extraction)
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir models/deepseek-ai/DeepSeek-OCR

# Qwen2-VL-2B-Instruct (Vision-Language Model)
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen/Qwen2-VL-2B-Instruct
```

### 3. Choose Your Setup

#### Option A — Docker (Recommended for GPU)

```bash
docker compose up -d --build
curl http://localhost:8001/health
```

#### Option B — Local (Using UV)

```bash
uv pip install -r requirements.txt
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git DeepSeek-OCR

export PYTHONPATH="$PWD/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm:$PYTHONPATH"
export MODEL_PATH="$PWD/models/deepseek-ai/DeepSeek-OCR"

uv run uvicorn start_server:app --host 0.0.0.0 --port 8001
```

---

## Example Outputs

**Input:** Scientific PDF with equations, tables, and figures
**Output:**

* Markdown file: `your_paper-MD.md`
* DOCX file: `your_paper.docx` (editable equations, embedded figures)
* JSONL file: `your_paper.jsonl` (structured for RAG)

One-command run:

```bash
python pdf_to_docx.py --input data/input/yourfile.pdf --output data/output/
```

---

## MM-RAG Pipeline with VLM Enrichment

### Overview

`pdf_to_mmrag_json.py` converts scientific PDFs into structured **JSONL** files suitable for multimodal RAG ingestion.
It can optionally use **Qwen2-VL-2B-Instruct** to generate concise, factual figure descriptions.

### Features

* **Structured JSONL Output:** Each element (paragraph, figure, table) is formatted for RAG embedding.
* **Automatic Figure Detection:** Extracts and links figures with captions and context.
* **VLM Descriptions:** Generates short scientific figure summaries (axes, trends, variable relationships).
* **Near-Complete Success Rate:** Robust watermark filtering and accurate scientific plot parsing.
* **Hardware-Agnostic:** Runs locally with GPU acceleration or CPU fallback.
* **No Paid APIs:** All processing is local and offline.

### Usage

**Basic (no VLM):**

```bash
python pdf_to_mmrag_json.py
```

**With Qwen2-VL enrichment:**

```bash
python pdf_to_mmrag_json.py --use-vlm
```

**Single file:**

```bash
python pdf_to_mmrag_json.py --input data/input/paper.pdf --use-vlm
```

### Example Output (Excerpt)

```json
{
  "element_id": "page_1_fig_1",
  "type": "figure",
  "content": "Storage modulus G' and loss modulus G''...",
  "metadata": {
    "image_path": "images/fig1.png",
    "caption": "Storage modulus G' and loss modulus G''...",
    "vlm_description": "The plot shows storage and loss moduli of PnBA-AA copolymers decreasing with frequency, consistent with power-law behavior."
  }
}
```

---

## Project Structure

```
SciDOCX/
├── data/
│   ├── input/           # Input PDFs
│   ├── output/          # Markdown and DOCX outputs
│   ├── mmrag-output/    # JSONL outputs
│   └── images/          # Extracted figures
├── pdf_to_docx.py       # Markdown/DOCX pipeline
├── pdf_to_mmrag_json.py # MM-RAG pipeline
├── shared_utils.py      # Common utilities
├── start_server.py      # DeepSeek-OCR API server
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Configuration Notes

**Docker:** Adjust ports and volumes in `docker-compose.yml`.
**Local:**

```bash
export PYTHONPATH="$PWD/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm:$PYTHONPATH"
export MODEL_PATH="$PWD/models/deepseek-ai/DeepSeek-OCR"
export CUDA_VISIBLE_DEVICES="0"
```

---

## Hardware Requirements

| Component | Minimum   | Recommended                 |
| --------- | --------- | --------------------------- |
| GPU       | 6 GB VRAM | ≥ 12 GB for large PDFs      |
| CPU       | 8 cores   | 16 GB RAM                   |
| Disk      | 5 GB      | 10 GB + (for cached models) |

---

## Why SciDOCX

* Higher fidelity than traditional OCR (Tesseract/pdfminer).
* Accurately preserves scientific equations and multi-column tables.
* Extracts, deduplicates, and embeds figures with captions.
* Produces both human-readable and retrieval-ready outputs.
* Fully local and reproducible — no external APIs required.

---

## 📊 Evaluation Framework

For comprehensive benchmarking results and academic evaluation, see:

- **Evaluation Branch**: `git checkout evaluation-framework`
- **Evaluation README**: [test/README.md](evaluation-framework/test/README.md)  
- **Evaluation Record**: [PR #2](https://github.com/EsmaeilNarimissa/SciDOCX/pull/2)

The evaluation framework includes:
- Cross-domain testing across 5 scientific disciplines
- Baseline comparisons with Tesseract and PDFMiner
- Performance metrics (WER, coverage, retrieval accuracy)
- Complete reproducibility artifacts and datasets

**Key Results**: 183.4% figure coverage, 0.309 WER vs PDFMiner, 100% retrieval hit rate

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, suggestions, or collaborations, reach out at:
**esmaeil.narimissa@gmail.com**

---

## Acknowledgments

- **[DeepSeek-AI](https://github.com/deepseek-ai)** for the DeepSeek-OCR model
- **[Qwen Team (Alibaba)](https://github.com/QwenLM/Qwen2-VL)** for Qwen2-VL-2B-Instruct VLM
- **[Pandoc](https://pandoc.org/)** for document conversion
- **[Bogdanovich77](https://github.com/Bogdanovich77/DeekSeek-OCR---Dockerized-API)** for Dockerized API implementation inspiration
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** for PDF processing
- **[HuggingFace Transformers](https://huggingface.co/transformers)** for model infrastructure