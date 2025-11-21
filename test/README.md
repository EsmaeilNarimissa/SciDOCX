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

**Key Performance Results:**
- **Cross-domain validation**: 5 scientific disciplines (Biology, Chemistry, Physics, Polymer Physics, Computer Science)
- **Superior text quality**: 0.309 WER vs PDFMiner (37x better than Tesseract)
- **Excellent structural extraction**: 183.4% figure coverage, 168% equation coverage
- **Perfect retrieval utility**: 100% hit rate across all domains
- **Production-ready performance**: 550s average processing time with VLM enhancement

**Evaluation Artifacts:**
- `metrics_multi_runtime.csv` - Performance timing data
- `metrics_multi_coverage.csv` - Structural extraction metrics  
- `metrics_multi_wer.csv` - OCR baseline comparisons
- `metrics_multi_retrieval.csv` - RAG utility assessment
- `metrics_multi_summary.csv` - Complete aggregated results
- `accuracy_annotations_multi.csv` - Human verification template
- `final_evaluation_summary.md` - Publication-ready summary
- `pipeline_test.ipynb` - Complete evaluation notebook

**Repository Links:**
- **Main Repository**: https://github.com/EsmaeilNarimissa/SciDOCX
- **Evaluation Record**: https://github.com/EsmaeilNarimissa/SciDOCX/pull/2

This branch is designed for **reproducibility and academic publication**, complementing the main branch
which remains focused on deployment and application development.

**Citation**: When using this evaluation framework, please cite both the main SciDOCX implementation
and this evaluation record (PR #2) for complete reproducibility.
