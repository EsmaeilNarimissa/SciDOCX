---
title: "SciDOCX: A multimodal scientific document conversion and retrieval pipeline"
authors:
  - name: "Esmaeil Narimissa"
    orcid: "0000-0002-5665-4412"
    affiliation: 1
affiliations:
  - name: "Independent Researcher"
    index: 1
date: "2025-11-19"
bibliography: paper.bib
tags:
  - OCR
  - scientific document understanding
  - multimodal retrieval
  - open-source software
---

# Summary

SciDOCX is presented as an open-source pipeline for converting complex scientific PDF documents into high-fidelity Markdown and DOCX files, together with a structured JSONL format that is suitable for multimodal retrieval-augmented generation workflows. The pipeline integrates DeepSeek-OCR [@deepseek-ocr] for text extraction, Pandoc [@pandoc] for layout reconstruction and Qwen2-VL [@qwen2-vl] for the automatic production of scientific figure descriptions. This design preserves equations, tables, figures and captions within their original context and provides outputs suitable for both human editing and downstream retrieval across scientific corpora. Evaluation on representative scientific articles shows that SciDOCX reduces word-level errors by a factor of about five relative to a baseline using Tesseract [@tesseract-ocr], while cross-domain tests on a set of arXiv papers demonstrate high structural coverage, accurate figure extraction and practical runtimes.

# Statement of need

Scientific literature contains equations, tables and figures that convey essential information, yet these elements are often degraded or lost when processed by standard OCR systems. Existing tools tend to produce plain text or approximate layout reconstructions with limited preservation of multimodal elements, which restricts the reuse of scientific PDFs in retrieval, analysis and multimodal question answering. At the same time, research in retrieval-augmented generation and vision-language modelling requires structured representations that align textual and visual components with consistent metadata. A dedicated pipeline that preserves scientific structure while also generating retrieval-ready multimodal formats is therefore required.

SciDOCX addresses this need by combining a modern OCR backend with a vision-language model to create both editable artefacts and JSONL files suitable for multimodal indexing. The pipeline retains mathematical content, table structure and figure context, and supplements figure elements with concise descriptions. Evaluations across several scientific domains indicate that SciDOCX provides more complete structural recovery and lower transcription error rates than baselines based on Tesseract and pdfminer.six [@pdfminer]. The outputs therefore serve researchers who prepare multimodal corpora, develop question answering systems, or convert scientific PDFs into structured formats for downstream analysis.

# Software description

## Functional overview

SciDOCX contains two coordinated pipelines that operate on the same OCR output. The first produces human-readable documents. A scientific PDF is processed by DeepSeek-OCR to generate structured output containing text blocks, bounding boxes and raster crops of detected figures. This material is transformed into GitHub-flavoured Markdown through routines that normalise equation delimiters, reconstruct table layouts when the OCR output provides sufficient structure and associate figure captions with their corresponding images. Pandoc is then used to convert the Markdown into an editable DOCX file that reproduces the logical structure of the original article.

The second pipeline produces a JSONL representation for multimodal retrieval. The OCR output is segmented into document elements such as paragraphs, tables and figures. Each entry is assigned metadata including element type, page number, position and references to extracted images. When multimodal enrichment is enabled, Qwen2-VL produces factual descriptions of figure crops that are stored alongside these entries. This representation is intended for retrieval systems that operate jointly over textual and visual elements.

## Implementation details

The implementation is organised around Python scripts and supporting utilities. The script `pdf_to_docx.py` orchestrates the generation of Markdown and DOCX files, while `pdf_to_mmrag_json.py` constructs the JSONL representation. Shared modules handle file management, OCR communication, page-level parsing and figure extraction. A lightweight server launched by `start_server.py` exposes the DeepSeek-OCR model through a local HTTP endpoint and is compatible with GPU-enabled machines.

PyMuPDF [@pymupdf] is used for PDF page access and for extracting raster images corresponding to figure regions. Optional dependencies, such as Tesseract and pdfminer.six, support baseline comparisons and form part of the evaluation framework supplied in a dedicated branch of the repository. Installation is possible through a local Python environment or via the Docker configuration provided in the repository. In both cases, users supply model weights for DeepSeek-OCR and Qwen2-VL and ensure that Pandoc is available on the system path.

## Quality control

Quality control involved automated checks and empirical evaluation. A detailed rheology article was used to verify the correctness of Markdown and DOCX reconstruction and to examine the consistency of figure extraction. In this setting, a simple Tesseract baseline was constructed for comparison. The baseline produced a word-error count that was about 4.7 times higher than the corresponding SciDOCX output, which indicated an improvement in transcription accuracy and equation preservation.

Further evaluation was carried out on arXiv articles from Biology, Chemistry, Physics, Polymer Physics and Computer Science. All pages in these documents were processed without failure. Figures were extracted reliably, and their captions remained associated with the correct image. When the JSONL representation was analysed, the vast majority of figures present in the original documents appeared as entries with valid captions and Qwen2-VL descriptions. The DOCX pipeline processed typical articles in tens of seconds on a single GPU-equipped workstation, while the multimodal pipeline required additional time proportional to the number of figures. Simple retrieval experiments using TF-IDF over the JSONL entries showed that factual queries about figures or methods usually returned the correct element within the top three results. These procedures and results are documented in the evaluation-framework branch of the repository.

# Comparison with related software

Tools such as Tesseract, pdfminer.six and GROBID support OCR, text extraction or metadata parsing, but none of them provide multimodal, retrieval-ready outputs with integrated figure descriptions. SciDOCX complements these tools by supplying Markdown and DOCX files suitable for editing and JSONL files suitable for multimodal indexing. Evaluation indicates that SciDOCX recovers more equations, captions and table elements than these baselines while maintaining practical runtimes.

# Availability

SciDOCX is implemented in Python (version 3.8 or later) and supports Linux, macOS and Windows. GPU acceleration is recommended for DeepSeek-OCR and Qwen2-VL, although CPU-only execution remains possible for smaller workloads. The open-source repository includes documentation, example inputs and outputs and a dedicated evaluation-framework branch. A versioned release is archived on Zenodo and assigned the DOI: https://doi.org/10.5281/zenodo.17665758.

# Reuse potential

SciDOCX can be used to construct corpora for multimodal question answering, automate the processing of scientific literature or support RAG pipelines that require consistent alignment of text and figures. The Markdown and DOCX outputs allow researchers to edit and annotate scientific material, while the JSONL format can be integrated into retrieval or indexation workflows. The modular structure encourages extensions, including the addition of new enrichment models or support for further document types.

# Acknowledgements

Development has benefited from the communities surrounding DeepSeek-OCR, Qwen2-VL, Pandoc and PyMuPDF, and from feedback gathered during evaluation.

# References
