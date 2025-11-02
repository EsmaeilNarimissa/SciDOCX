#!/usr/bin/env python3
"""
Shared utilities for DeepSeek-OCR pipelines.
Provides common functions for OCR API calls, image extraction, and content cleaning.
"""
import os
import re
import io
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import requests
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def test_api_connection(api_base_url: str) -> bool:
    """Test connection to the DeepSeek-OCR API."""
    try:
        resp = requests.get(f"{api_base_url}/docs", timeout=5)
        if resp.status_code == 200:
            logger.info("API connection successful")
            return True
        logger.error("API returned status code: %s", resp.status_code)
        return False
    except requests.exceptions.RequestException as e:
        logger.error("API connection failed: %s", str(e))
        return False


def hash_bytes(b: bytes) -> str:
    """Generate SHA1 hash of bytes for deduplication."""
    return hashlib.sha1(b).hexdigest()


def call_ocr_api(pdf_path: str, api_base_url: str) -> Optional[Dict[str, Any]]:
    """
    Call DeepSeek-OCR API to process a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        api_base_url: Base URL of the OCR API
        
    Returns:
        API response as dict, or None if failed
    """
    try:
        url = f"{api_base_url}/ocr/pdf"
        logger.info("Processing PDF with API endpoint: %s", url)
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            resp = requests.post(url, files=files, timeout=300)
            if resp.status_code == 200:
                return resp.json()
            logger.error("API request failed %s: %s", resp.status_code, resp.text)
            return None
    except Exception as e:
        logger.error("Error calling API: %s", e)
        return None


def re_match_refs(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse DeepSeek-OCR grounding tags from text.
    
    Returns:
        Tuple of (all_matches, image_matches, other_matches)
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    m_img, m_other = [], []
    allm = [m[0] for m in matches]
    for a in allm:
        if '<|ref|>image<|/ref|>' in a:
            m_img.append(a)
        else:
            m_other.append(a)
    return allm, m_img, m_other


def extract_embedded_images(
    pdf_path: str,
    page_idx: int,
    images_dir: Path,
    seen_hashes: Optional[set] = None,
    relative_path: str = "../images"
) -> Tuple[str, int]:
    """
    Extract embedded images from a PDF page.
    
    Args:
        pdf_path: Path to the PDF file
        page_idx: Page index (0-based)
        images_dir: Directory to save extracted images
        seen_hashes: Set of already seen image hashes (for deduplication)
        relative_path: Relative path prefix for markdown links
        
    Returns:
        Tuple of (markdown_links, image_count)
    """
    links = []
    count = 0
    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= doc.page_count:
            doc.close()
            return "", 0
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        stem_safe = Path(pdf_path).stem.replace(" ", "_")
        for i, img in enumerate(image_list, start=1):
            try:
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base.get("image")
                if not img_bytes:
                    continue
                # dedup by content hash before saving
                img_hash = hash_bytes(img_bytes)
                if seen_hashes is not None and img_hash in seen_hashes:
                    continue
                if seen_hashes is not None:
                    seen_hashes.add(img_hash)
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                fname = f"{stem_safe}_p{page_idx+1}_img{i}.png"
                out_path = images_dir / fname
                pil.save(out_path, format="PNG")
                links.append(f"![Figure_{i}]({relative_path}/{fname})")
                count += 1
            except Exception as ie:
                logger.error("Error saving embedded image: %s", ie)
        doc.close()
    except Exception as e:
        logger.error("Error extracting embedded images: %s", e)
    return ("\n".join(links) + ("\n\n" if links else ""), count)


def pdf_to_images_bitmap(pdf_path: str, dpi: int = 144) -> List[Image.Image]:
    """
    Convert PDF pages to bitmap images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering
        
    Returns:
        List of PIL Image objects
    """
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        pdf_document.close()
    except Exception as e:
        logger.error("Error converting PDF to images: %s", e)
    return images


def extract_crops_from_boxes(
    pdf_path: str,
    content: str,
    page_idx: int,
    images_dir: Path,
    seen_hashes: Optional[set] = None,
    relative_path: str = "../images"
) -> Tuple[str, int]:
    """
    Extract cropped images from grounding box coordinates in content.
    
    Args:
        pdf_path: Path to the PDF file
        content: Text content with grounding tags
        page_idx: Page index (0-based)
        images_dir: Directory to save extracted images
        seen_hashes: Set of already seen image hashes (for deduplication)
        relative_path: Relative path prefix for markdown links
        
    Returns:
        Tuple of (processed_content, image_count)
    """
    pdf_bitmaps = pdf_to_images_bitmap(pdf_path)
    if page_idx >= len(pdf_bitmaps):
        return content, 0
    page_image = pdf_bitmaps[page_idx]
    w, h = page_image.size
    _, matches_images, _ = re_match_refs(content)
    img_idx = 0
    for a_match in matches_images:
        try:
            det_match = re.search(r'<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>', a_match)
            if not det_match:
                continue
            det_content = det_match.group(1)
            try:
                coords = eval(det_content)
                for points in coords:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * w); y1 = int(y1 / 999 * h)
                    x2 = int(x2 / 999 * w); y2 = int(y2 / 999 * h)
                    cropped = page_image.crop((x1, y1, x2, y2))
                    # hash cropped bytes to avoid duplicates
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    img_hash = hash_bytes(img_bytes)
                    if seen_hashes is not None and img_hash in seen_hashes:
                        # remove the tag but skip saving/linking duplicate
                        content = content.replace(a_match, "", 1)
                        break
                    if seen_hashes is not None:
                        seen_hashes.add(img_hash)
                    stem_safe = Path(pdf_path).stem.replace(" ", "_")
                    fname = f"{stem_safe}_p{page_idx+1}_img{img_idx+1}.png"
                    out_path = images_dir / fname
                    with open(out_path, "wb") as f_out:
                        f_out.write(img_bytes)
                    content = content.replace(a_match, f"![]({relative_path}/{fname})\n", 1)
                    img_idx += 1
                    break
            except Exception as ce:
                logger.error("Error processing box coords: %s", ce)
                content = content.replace(a_match, "", 1)
        except Exception as e:
            logger.error("Error extracting crop: %s", e)
            content = content.replace(a_match, "", 1)
    return content, img_idx


def html_table_to_markdown(html_table: str) -> str:
    """
    Convert a simple HTML table to Markdown table format.
    
    Args:
        html_table: HTML table string
        
    Returns:
        Markdown table string
    """
    try:
        # Extract rows
        rows = re.findall(r'<tr>(.*?)</tr>', html_table, re.DOTALL)
        if not rows:
            return html_table
        
        md_rows = []
        for row in rows:
            # Extract cells (td or th)
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
            if cells:
                # Clean cell content
                cells = [cell.strip().replace('\n', ' ') for cell in cells]
                md_rows.append('| ' + ' | '.join(cells) + ' |')
        
        if not md_rows:
            return html_table
        
        # Add header separator after first row
        if len(md_rows) > 1:
            num_cols = md_rows[0].count('|') - 1
            separator = '| ' + ' | '.join(['---'] * num_cols) + ' |'
            md_rows.insert(1, separator)
        
        return '\n'.join(md_rows)
    except Exception as e:
        logger.warning(f"Failed to convert HTML table: {e}")
        return html_table


def clean_content(content: str) -> str:
    """
    Clean OCR content by removing special tags and normalizing formatting.
    
    Args:
        content: Raw OCR content
        
    Returns:
        Cleaned content
    """
    if '<｜end▁of▁sentence｜>' in content:
        content = content.replace('<｜end▁of▁sentence｜>', '')
    
    # Remove grounding tags but keep the text content
    # Pattern: <|ref|>type<|/ref|><|det|>[[coords]]<|/det|>
    pattern = r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>'
    content = re.sub(pattern, '', content)
    
    # Remove <center> tags for better Pandoc compatibility
    content = content.replace('<center>', '').replace('</center>', '')
    
    # Convert HTML tables to Markdown tables
    table_pattern = r'<table>.*?</table>'
    tables = re.findall(table_pattern, content, re.DOTALL | re.IGNORECASE)
    for html_table in tables:
        md_table = html_table_to_markdown(html_table)
        content = content.replace(html_table, md_table)
    
    # Normalize LaTeX symbols
    content = content.replace('\\coloneqq', ':=')
    content = content.replace('\\eqqcolon', '=:')
    
    # Normalize whitespace
    content = content.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
    
    return content.strip()


def get_pdf_page_count(pdf_path: str) -> Optional[int]:
    """
    Get the number of pages in a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages, or None if error
    """
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception as e:
        logger.error(f"Error reading PDF page count: {e}")
        return None


def split_pdf_into_chunks(pdf_path: str, chunk_size: int = 3) -> List[Tuple[Path, Tuple[int, int]]]:
    """
    Split a PDF into smaller chunk files.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Number of pages per chunk
        
    Returns:
        List of (temp_path, (start_idx, end_idx)) tuples
    """
    temp_files: List[Tuple[Path, Tuple[int, int]]] = []
    doc = fitz.open(pdf_path)
    total = doc.page_count
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        temp_path = Path(pdf_path).with_name(f"{Path(pdf_path).stem}_chunk_{start+1}_{end}.pdf")
        new_doc.save(str(temp_path))
        new_doc.close()
        temp_files.append((temp_path, (start, end)))
    doc.close()
    return temp_files


def concat_api_results(
    results_list: List[Dict[str, Any]],
    chunk_ranges: List[Tuple[int, int]]
) -> Dict[str, Any]:
    """
    Concatenate chunked OCR results and preserve original page indices.
    
    Args:
        results_list: List of API response dicts
        chunk_ranges: List of (start_idx, end_idx) for each chunk
        
    Returns:
        Combined results dict
    """
    out: Dict[str, Any] = {"results": []}
    for (start_idx, end_idx), r in zip(chunk_ranges, results_list):
        if isinstance(r, dict) and isinstance(r.get("results"), list):
            # Annotate each result with its actual PDF page number
            for local_idx, page_result in enumerate(r["results"]):
                actual_page_idx = start_idx + local_idx
                if isinstance(page_result, dict):
                    page_result["_pdf_page_idx"] = actual_page_idx
                out["results"].append(page_result)
        else:
            # fallback: treat as single page string
            out["results"].append({"result": str(r), "_pdf_page_idx": start_idx})
    return out
