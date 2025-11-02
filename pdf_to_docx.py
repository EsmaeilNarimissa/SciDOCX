#!/usr/bin/env python3
import os
import sys
import re
import io
import json
import logging
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import requests
from PIL import Image
import hashlib
import fitz  # PyMuPDF

# Import shared utilities
import shared_utils

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*AutoModelForVision2Seq.*deprecated.*")
warnings.filterwarnings("ignore", message=".*generation flags.*not valid.*")
warnings.filterwarnings("ignore", message=".*fast processor.*breaking change.*")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_VERBOSITY.*")

# Set environment variable to suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


class PDFToDocxProcessor:
    def __init__(self, base_dir: str = "data", api_base_url: str = "http://localhost:8001"):
        # Set up directory structure
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.images_dir = self.base_dir / "images"
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_base_url = api_base_url
        if not self._test_api_connection():
            raise ConnectionError(f"Cannot connect to API at {api_base_url}")
            
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Images directory: {self.images_dir}")

    def _test_api_connection(self) -> bool:
        return shared_utils.test_api_connection(self.api_base_url)

    def _hash_bytes(self, b: bytes) -> str:
        return shared_utils.hash_bytes(b)

    def _extract_embedded_images(self, pdf_path: str, page_idx: int, seen_hashes: Optional[set] = None) -> Tuple[str, int]:
        return shared_utils.extract_embedded_images(
            pdf_path, page_idx, self.images_dir, seen_hashes, relative_path="../images"
        )

    def _pdf_to_images_bitmap(self, pdf_path: str, dpi: int = 144) -> List[Image.Image]:
        return shared_utils.pdf_to_images_bitmap(pdf_path, dpi)

    def _re_match_refs(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        return shared_utils.re_match_refs(text)

    def _extract_crops_from_boxes(self, pdf_path: str, content: str, page_idx: int, seen_hashes: Optional[set] = None) -> Tuple[str, int]:
        return shared_utils.extract_crops_from_boxes(
            pdf_path, content, page_idx, self.images_dir, seen_hashes, relative_path="../images"
        )

    def _html_table_to_markdown(self, html_table: str) -> str:
        """Convert a simple HTML table to Markdown table format."""
        return shared_utils.html_table_to_markdown(html_table)

    def _clean_content(self, content: str) -> str:
        return shared_utils.clean_content(content)

    def _call_ocr_api(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        return shared_utils.call_ocr_api(pdf_path, self.api_base_url)

    def _assemble_markdown(self, pdf_path: str, api_result: Dict[str, Any]) -> str:
        processed = ""
        if isinstance(api_result, dict) and "results" in api_result and isinstance(api_result["results"], list):
            logger.info("API returned %d page results", len(api_result["results"]))
            for array_idx, page_result in enumerate(api_result["results"]):
                # Handle None values explicitly
                if isinstance(page_result, dict):
                    page_content = page_result.get("result", "")
                    page_content = "" if page_content is None else page_content
                else:
                    page_content = "" if page_result is None else str(page_result)
                # Use actual PDF page index if available (from chunked processing), else use array index
                pdf_page_idx = page_result.get("_pdf_page_idx", array_idx) if isinstance(page_result, dict) else array_idx
                # deduplicate across both sources on a per-page basis
                seen_hashes: set = set()
                links_md, embed_count = self._extract_embedded_images(pdf_path, pdf_page_idx, seen_hashes)
                if links_md:
                    page_content = links_md + page_content
                # always also try grounding-based crops, but skip duplicates by hash
                page_content, _ = self._extract_crops_from_boxes(pdf_path, page_content, pdf_page_idx, seen_hashes)
                page_content = self._clean_content(page_content)
                processed += f"\n\n<!-- Page {pdf_page_idx+1} -->\n\n" + page_content + "\n\n"
        else:
            # single-block response fallback
            content = str(api_result)
            links_md, _ = self._extract_embedded_images(pdf_path, 0)
            page_content = links_md + content
            page_content = self._clean_content(page_content)
            processed += page_content + "\n\n"
        return processed.strip()

    def _split_pdf_into_chunks(self, pdf_path: str, chunk_size: int = 3) -> List[Tuple[Path, Tuple[int, int]]]:
        """Create temporary chunk PDFs of size chunk_size. Returns list of (temp_path, (start_idx, end_idx))."""
        return shared_utils.split_pdf_into_chunks(pdf_path, chunk_size)

    def _concat_api_results(self, results_list: List[Dict[str, Any]], chunk_ranges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Concatenate chunked results and preserve original page indices."""
        return shared_utils.concat_api_results(results_list, chunk_ranges)

    def _normalize_math_delimiters(self, md: str) -> str:
        """
        Normalize LaTeX math to Pandoc-friendly delimiters to improve DOCX rendering.
        - Replace \( ... \) with $...$
        - Replace \[ ... \] with $$...$$
        """
        # Display math: \[ ... \] -> $$ ... $$
        md = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", md, flags=re.DOTALL)
        # Inline math: \( ... \) -> $ ... $
        md = re.sub(r"\\\((.+?)\\\)", r"$\1$", md, flags=re.DOTALL)
        return md

    def _ensure_pandoc(self) -> bool:
        try:
            subprocess.run(["pandoc", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def _md_to_docx(self, md_path: Path, docx_path: Path) -> bool:
        if not self._ensure_pandoc():
            logger.error("Pandoc not found. Install with: conda install -c conda-forge pandoc")
            return False
        try:
            subprocess.run([
                "pandoc", str(md_path), "-o", str(docx_path), 
                "--from", "gfm+tex_math_dollars+raw_html",
                "--resource-path", str(md_path.parent)
            ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error("Pandoc failed: %s", e)
            return False

    def _save_results(self, pdf_path: str, md_content: str) -> Tuple[Path, Path]:
        """Save markdown content and convert to DOCX."""
        pdf_p = Path(pdf_path)
        md_path = self.output_dir / pdf_p.with_name(f"{pdf_p.stem}-MD.md").name
        md_path.write_text(self._normalize_math_delimiters(md_content), encoding="utf-8")
        logger.info("Wrote Markdown: %s", md_path)
        docx_path = self.output_dir / pdf_p.with_name(f"{pdf_p.stem}.docx").name
        if self._md_to_docx(md_path, docx_path):
            logger.info("Wrote DOCX: %s", docx_path)
        else:
            logger.warning("DOCX generation skipped (Pandoc missing)")
        return md_path, docx_path

    def convert_one(self, pdf_path: str) -> Optional[Tuple[Path, Path]]:
        logger.info("Processing PDF: %s", pdf_path)
        # inspect total pages
        try:
            src_doc = fitz.open(pdf_path)
            total_pages = src_doc.page_count
            src_doc.close()
        except Exception as e:
            logger.warning("Could not inspect PDF pages: %s", e)
            total_pages = None
        api_result = self._call_ocr_api(pdf_path)
        if api_result is None:
            return None
        # If API returned fewer pages than source, fallback to chunked processing
        returned_pages = None
        if isinstance(api_result, dict) and isinstance(api_result.get("results"), list):
            returned_pages = len(api_result["results"])
        if total_pages is not None and returned_pages is not None and returned_pages < total_pages:
            logger.warning(
                "API returned %d of %d pages. Falling back to chunked processing.",
                returned_pages, total_pages
            )
            chunks = self._split_pdf_into_chunks(pdf_path, chunk_size=3)
            combined: List[Dict[str, Any]] = []
            for temp_path, (start, end) in chunks:
                logger.info("Processing chunk pages %d-%d", start + 1, end)
                chunk_result = self._call_ocr_api(str(temp_path))
                if chunk_result is not None:
                    combined.append(chunk_result)
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass
            api_result = self._concat_api_results(combined, [(s, e) for _, (s, e) in chunks])
        md_content = self._assemble_markdown(pdf_path, api_result)
        return self._save_results(pdf_path, md_content)

    def scan_and_process_all_pdfs(self) -> List[Tuple[Path, Path]]:
        pdf_files = list(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in %s", self.input_dir)
            return []
        logger.info("Found %d PDF files to process", len(pdf_files))
        outputs: List[Tuple[Path, Path]] = []
        for pdf in pdf_files:
            r = self.convert_one(str(pdf))
            if r:
                outputs.append(r)
        return outputs


def main():
    print(f"{Colors.BLUE}PDF to DOCX Processor{Colors.RESET}")
    print(f"{Colors.YELLOW}Directory Structure:{Colors.RESET}")
    print(f"  - Input:  ./data/input/    # Put your PDFs here")
    print(f"  - Output: ./data/output/   # Processed files will be saved here")
    print(f"  - Images: ./data/images/   # Extracted images (if any)")
    
    # Get API URL from environment variable or use default
    api_url = os.environ.get('DEEPSEEK_OCR_API', 'http://localhost:8001')
    print(f"\n{Colors.YELLOW}Using API URL: {api_url}{Colors.RESET}")
    print(f"{Colors.YELLOW}Scanning for PDF files...{Colors.RESET}")
    
    try:
        proc = PDFToDocxProcessor(api_base_url=api_url)
        results = proc.scan_and_process_all_pdfs()
        if results:
            print(f"\n{Colors.GREEN}Successfully processed {len(results)} PDF files:{Colors.RESET}")
            for md_path, docx_path in results:
                print(f"  - MD: {md_path}")
                if docx_path.exists():
                    print(f"    DOCX: {docx_path}")
                else:
                    print(f"    DOCX: (skipped, install pandoc)")
        else:
            print(f"{Colors.YELLOW}No PDF files were processed.{Colors.RESET}")
    except requests.exceptions.ConnectionError as e:
        logger.error("Failed to connect to the API server. Is it running?")
        print(f"{Colors.RED}Error: Cannot connect to API at {api_url}. Make sure the server is running.{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        logger.error("Application error: %s", str(e))
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
