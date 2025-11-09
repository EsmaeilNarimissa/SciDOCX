# SciDOCX Evaluation Framework

This branch contains the complete evaluation and benchmarking results for the SciDOCX system.
It includes input PDFs, baseline OCR outputs, processed DOCX and Markdown files, JSONL structured outputs,
and performance metrics across five scientific domains.

**Contents:**
- `/data/input/` — source scientific PDFs used for evaluation  
- `/data/evaluation/baselines/` — Tesseract and PDFMiner text baselines  
- `/data/output/` — generated DOCX and Markdown documents  
- `/data/mmrag-output/` — multimodal RAG-compatible JSONL outputs  
- `/test/` — evaluation notebooks and CSV metric summaries

This branch is designed for **reproducibility and academic publication**, complementing the main branch
which remains focused on deployment and application development.
