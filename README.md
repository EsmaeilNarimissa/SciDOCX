# DeepSeek-OCR: Scientific Document Conversion Pipeline

A high-fidelity OCR-to-Markdown and OCR-to-DOCX pipeline powered by DeepSeek-OCR, Pandoc, and Python.  
It converts scientific PDFs — including equations, tables, and figures — into clean, editable Word or Markdown files with full layout preservation.

## Key Features

- **Accurate OCR:** Uses DeepSeek-OCR with GPU acceleration for near human-level text recognition
- **Scientific Awareness:** Preserves LaTeX-style equations, tables, and figure captions
- **Markdown and DOCX Export:** Converts to GitHub-flavored Markdown and fully editable Word documents
- **Embedded Figures:** Automatically extracts and embeds figures directly into the DOCX file
- **VLM Enrichment:** Qwen2-VL-2B-Instruct for accurate scientific figure descriptions (MM-RAG pipeline)
- **End-to-End Automation:** One command to process a folder of PDFs
- **Cross-Platform:** Works via Docker or locally using Python/UV without containers

## Quick Start Guide

### Prerequisites
- Docker with NVIDIA Container Toolkit installed **OR** Python 3.8+ with UV
- Git

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd DS-OCR
```

### 2. Download Model Weights
```bash
# Download the OCR and VLM models
pip install huggingface_hub

# DeepSeek-OCR (text extraction)
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir models/deepseek-ai/DeepSeek-OCR

# Qwen2-VL-2B-Instruct (Vision-Language Model)
# Automatically downloaded on first run, but you can prefetch manually:
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir models/Qwen/Qwen2-VL-2B-Instruct
```

### 3. Choose Your Setup

#### Option A: Docker (Recommended for GPU acceleration)
```bash
# Build and start the service
docker compose up -d --build

# Check health
curl http://localhost:8001/health
```

#### Option B: Local Setup (Using UV)
```bash
# Install dependencies with UV
uv pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="$PWD/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm:$PYTHONPATH"
export MODEL_PATH="$PWD/models/deepseek-ai/DeepSeek-OCR"

# Launch the API server
uv run uvicorn start_server:app --host 0.0.0.0 --port 8001
```

### Stopping the Service

**Docker:**
```bash
docker compose down
```

**Local:**
```bash
# Stop the uvicorn process (Ctrl+C)
```

## Example Output

**Input PDF:** Scientific paper with complex equations, tables, and figures

**Output DOCX:** 
- Editable Word equations (`G(t) = Σ g_i exp(-t/τ_i)`)
- Embedded figures (Fig. 1–7) with captions
- Fully formatted tables
- Clean Markdown alternative for version control

**One-command demo:**
```bash
python pdf_to_docx.py --input data/input/yourfile.pdf --output data/output/
```

## API Usage

### Process an Image
```bash
curl -X POST "http://localhost:8001/ocr/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.jpg"
```

### Process a PDF
```bash
curl -X POST "http://localhost:8001/ocr/pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/document.pdf"
```

## Project Structure

```
DS-OCR/
├── data/
│   ├── input/          # Put your PDFs here
│   ├── output/         # Processed Markdown and DOCX files
│   ├── mmrag-output/   # JSONL files for RAG (MM-RAG pipeline)
│   └── images/         # Extracted figures and images
├── pdf_to_docx.py      # Main conversion script (Markdown/DOCX)
├── pdf_to_mmrag_json.py # MM-RAG pipeline (JSONL for RAG)
├── start_server.py     # DeepSeek-OCR API server
├── shared_utils.py     # Shared utilities
├── requirements.txt    # Dependencies
├── requirements-lock.txt # Locked versions
└── README.md           # This file
```

## Configuration

**Docker:** Edit `docker-compose.yml` to adjust port mappings and volume mounts

**Local:** Set environment variables:
```bash
# Set Python path to include DeepSeek-OCR code
export PYTHONPATH="$PWD/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm:$PYTHONPATH"

# Point to the model directory
export MODEL_PATH="$PWD/models/deepseek-ai/DeepSeek-OCR"

# Optional: Specify GPU device
export CUDA_VISIBLE_DEVICES="0"
```

## Quick Demo

```bash
# 1. Create directories
mkdir -p data/input data/output data/images

# 2. Add your PDF
cp your_scientific_paper.pdf data/input/

# 3. Run the conversion
python pdf_to_docx.py

# 4. Find your results in data/output/
ls data/output/
# your_paper-MD.md    # GitHub-flavored Markdown
# your_paper.docx      # Editable Word document
```

## What Makes This Different?

- **Higher fidelity than Tesseract/pdfminer:** DeepSeek-OCR understands scientific notation and complex layouts
- **Preserves mathematical equations:** LaTeX-style math converted to editable Word equations
- **Maintains table structure:** Complex multi-column tables properly formatted
- **Figure extraction:** Images embedded with correct relative paths
- **End-to-end automation:** No manual post-processing required

## MM-RAG Pipeline with VLM Enrichment

### Overview

The MM-RAG pipeline (`pdf_to_mmrag_json.py`) converts scientific PDFs into structured JSONL format optimized for multimodal Retrieval-Augmented Generation (RAG) systems. It includes **optional VLM (Vision-Language Model) enrichment** for automatic figure descriptions using **Qwen2-VL-2B-Instruct**.

The Qwen2-VL model is **automatically downloaded** from Hugging Face on first use (~5GB, cached locally). You can also pre-download it manually using the instructions in the setup section above.

### Features

- **Structured JSONL Output:** Elements (paragraphs, figures, tables) formatted for RAG ingestion
- **Figure Detection:** Automatic extraction and linking of figures with captions
- **VLM Descriptions:** Qwen2-VL-2B-Instruct for generating accurate scientific figure descriptions
- **100% Success Rate:** Superior watermark filtering and scientific plot understanding
- **Local & Free:** No paid APIs required, runs entirely on your hardware
- **GPU First, CPU Fallback:** Automatic device detection

### Usage

**Basic (No VLM):**
```bash
python pdf_to_mmrag_json.py
```

**With Qwen2-VL Enrichment (Recommended):**
```bash
python pdf_to_mmrag_json.py --use-vlm
```

**Single File:**
```bash
python pdf_to_mmrag_json.py --input data/input/paper.pdf --use-vlm
```

### VLM Backend: Qwen2-VL-2B-Instruct

**Model:** `Qwen/Qwen2-VL-2B-Instruct`

**Why Qwen2-VL?**
- **High accuracy** on scientific figures with complex plots
- **Watermark filtering** - ignores copyright text and publication dates
- **Axis-aware** - identifies axes, units, and variable relationships
- **Compact** - 5GB model optimized for scientific content

**Hardware Requirements:**
- **GPU (Recommended):** 6GB+ VRAM (NVIDIA)
- **CPU (Fallback):** 16GB+ RAM (slower)
- **Disk:** ~5GB for model download (first run only, then cached)

**What It Generates:**
- 2-3 sentence scientific descriptions
- Plot type, structure, and axes identification
- Data trends and variable relationships
- Watermark-free analysis
- Added to `metadata.vlm_description` field

**Output Format:**
```json
{
  "element_id": "page_1_fig_1",
  "type": "figure",
  "content": "Storage modulus G' and loss modulus G''...",
  "metadata": {
    "image_path": "images/fig1.png",
    "caption": "Storage modulus G' and loss modulus G''...",
    "vlm_description": "The plot depicts the storage modulus (G') and loss modulus (G'') of PnBA-AA copolymers. The lines represent the expected power law dependence of G' and G'' in the low frequency range. The data trends show both G' and G'' decreasing with increasing frequency."
  }
}
```

**Performance:**
- ~8-10 seconds per figure (GPU)
- ~25-40 seconds per figure (CPU)
- Automatic GPU cache management
- First run: ~2 min model download (cached thereafter)

### Pipeline Summary Log

The pipeline provides a summary at completion:
```
VLM Summary: 7 processed, 7 enriched, 0 failed
```

- **Processed:** Total figures attempted
- **Enriched:** Successfully described  
- **Failed:** Missing images or errors

**Typical Success Rate:** High reliability on scientific figures with clear visualizations

### Output Files

1. **JSONL:** `data/mmrag-output/yourfile.jsonl` - Structured elements for RAG
2. **Debug Markdown:** `data/mmrag-output/yourfile.debug.md` - Human-readable preview

### Capabilities & Limitations

**Strengths:**
- **Watermark filtering** - ignores publication dates, copyright text
- **Axis recognition** - identifies plot axes, labels, and units
- **Trend analysis** - describes data patterns and relationships
- **Scientific focus** - trained on scientific and technical content

**Limitations:**
- Descriptions are concise (2-3 sentences) - optimized for RAG retrieval
- Multi-panel figures analyzed as single composite (not per-panel)
- Very complex 3D plots may receive simplified descriptions

### Regression Test

**Without VLM (baseline):**
```bash
python pdf_to_mmrag_json.py
# Should complete without errors, figures detected but no vlm_description
```

**With VLM:**
```bash
python pdf_to_mmrag_json.py --use-vlm
# Check JSONL for vlm_description fields in figure elements
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[DeepSeek-AI](https://github.com/deepseek-ai)** for the DeepSeek-OCR model
- **[Qwen Team (Alibaba)](https://github.com/QwenLM/Qwen2-VL)** for Qwen2-VL-2B-Instruct VLM
- **[Pandoc](https://pandoc.org/)** for document conversion
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** for PDF processing
- **[HuggingFace Transformers](https://huggingface.co/transformers)** for model infrastructure