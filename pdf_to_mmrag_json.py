#!/usr/bin/env python3
"""
PDF to MM-RAG JSON Pipeline
Converts scientific PDFs into structured JSONL for multimodal RAG ingestion.

This pipeline:
1. Extracts text, figures, and tables using DeepSeek-OCR
2. Parses and structures content by elements (paragraphs, figures, tables)
3. Enriches figures with VLM descriptions and tables with LLM summaries
4. Outputs clean JSONL ready for RAG embedding
"""
import os
import sys
import re
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

import requests
from PIL import Image

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

# Optional VLM imports (lazy loaded)
try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    torch = None
    AutoProcessor = None
    AutoModelForVision2Seq = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mmrag_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


class PDFToMMRAGProcessor:
    """Convert PDFs to structured JSONL for multimodal RAG systems."""
    
    def __init__(self, base_dir: str = "data", api_base_url: str = "http://localhost:8001", use_vlm: bool = False):
        # Set up directory structure
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "mmrag-output"
        self.images_dir = self.base_dir / "images"
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_base_url = api_base_url
        self.use_vlm = use_vlm
        
        # VLM components (lazy loaded)
        self.vlm_model = None
        self.vlm_processor = None
        self.device = None
        
        # VLM statistics
        self.vlm_stats = {"processed": 0, "enriched": 0, "failed": 0}
        
        if not shared_utils.test_api_connection(api_base_url):
            raise ConnectionError(f"Cannot connect to API at {api_base_url}")
            
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Images directory: {self.images_dir}")
        
        # Initialize VLM if requested
        if self.use_vlm:
            self._initialize_vlm()
    
    # ========================================================================
    # PHASE 1: OCR Extraction (reuse shared_utils)
    # ========================================================================
    
    def _extract_ocr_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract OCR content from PDF using DeepSeek-OCR API.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            OCR API result dictionary
        """
        logger.info(f"Phase 1: Extracting OCR content from {pdf_path}")
        
        # Get page count
        total_pages = shared_utils.get_pdf_page_count(pdf_path)
        logger.info(f"PDF has {total_pages} pages")
        
        # Call OCR API
        api_result = shared_utils.call_ocr_api(pdf_path, self.api_base_url)
        if api_result is None:
            raise RuntimeError("OCR API call failed")
        
        # Check if we got all pages
        returned_pages = None
        if isinstance(api_result, dict) and isinstance(api_result.get("results"), list):
            returned_pages = len(api_result["results"])
        
        # If API returned fewer pages, use chunking
        if total_pages and returned_pages and returned_pages < total_pages:
            logger.warning(f"API returned {returned_pages}/{total_pages} pages. Using chunked processing.")
            api_result = self._process_pdf_in_chunks(pdf_path)
        
        return api_result
    
    def _process_pdf_in_chunks(self, pdf_path: str, chunk_size: int = 3) -> Dict[str, Any]:
        """Process large PDFs in chunks to avoid API limits."""
        chunks = shared_utils.split_pdf_into_chunks(pdf_path, chunk_size)
        combined: List[Dict[str, Any]] = []
        
        for temp_path, (start, end) in chunks:
            logger.info(f"Processing chunk pages {start + 1}-{end}")
            chunk_result = shared_utils.call_ocr_api(str(temp_path), self.api_base_url)
            if chunk_result is not None:
                combined.append(chunk_result)
            
            # Clean up temp file
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
        
        return shared_utils.concat_api_results(combined, [(s, e) for _, (s, e) in chunks])
    
    def _extract_all_images(self, pdf_path: str, api_result: Dict[str, Any]) -> Dict[int, List[str]]:
        """
        Extract all images from PDF pages.
        
        Returns:
            Dictionary mapping page_idx -> list of image paths
        """
        logger.info("Extracting images from all pages...")
        page_images: Dict[int, List[str]] = {}
        
        if not isinstance(api_result, dict) or "results" not in api_result:
            return page_images
        
        for page_result in api_result["results"]:
            if not isinstance(page_result, dict):
                continue
            
            page_idx = page_result.get("_pdf_page_idx", 0)
            page_content = page_result.get("result", "")
            
            # Extract embedded images
            seen_hashes: set = set()
            _, _ = shared_utils.extract_embedded_images(
                pdf_path, page_idx, self.images_dir, seen_hashes, relative_path="images"
            )
            
            # Extract crops from grounding boxes
            processed_content, _ = shared_utils.extract_crops_from_boxes(
                pdf_path, page_content, page_idx, self.images_dir, seen_hashes, relative_path="images"
            )
            
            # Find image references in processed content
            image_refs = re.findall(r'!\[.*?\]\(images/(.*?)\)', processed_content)
            if image_refs:
                page_images[page_idx] = image_refs
        
        total_images = sum(len(imgs) for imgs in page_images.values())
        logger.info(f"Extracted {total_images} images across {len(page_images)} pages")
        return page_images
    
    # ========================================================================
    # PHASE 2: Parse & Structure Content
    # ========================================================================
    
    def _parse_content_to_elements(self, pdf_path: str, api_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse OCR content into structured elements (paragraphs, figures, tables).
        
        Returns:
            List of element dictionaries with type, content, metadata
        """
        logger.info("Phase 2: Parsing content into structured elements...")
        elements: List[Dict[str, Any]] = []
        
        if not isinstance(api_result, dict) or "results" not in api_result:
            return elements
        
        for page_result in api_result["results"]:
            if not isinstance(page_result, dict):
                continue
            
            page_idx = page_result.get("_pdf_page_idx", 0)
            page_content = page_result.get("result", "")
            if not page_content:
                continue
            
            # Extract images first
            seen_hashes: set = set()
            _, _ = shared_utils.extract_embedded_images(
                pdf_path, page_idx, self.images_dir, seen_hashes, relative_path="images"
            )
            page_content, _ = shared_utils.extract_crops_from_boxes(
                pdf_path, page_content, page_idx, self.images_dir, seen_hashes, relative_path="images"
            )
            
            # Clean content
            page_content = shared_utils.clean_content(page_content)
            
            # Parse elements from this page
            page_elements = self._parse_page_elements(page_content, page_idx)
            elements.extend(page_elements)
        
        logger.info(f"Parsed {len(elements)} elements from PDF")
        return elements
    
    def _parse_page_elements(self, content: str, page_idx: int) -> List[Dict[str, Any]]:
        """
        Parse a single page into structured elements (figures, tables, text).
        
        Strategy:
        1. Extract figures with captions
        2. Extract tables with captions
        3. Split remaining text into paragraphs
        4. Associate elements with context
        """
        elements = []
        
        # Extract current section heading
        current_section = self._extract_current_section(content)
        
        # Split content into lines for processing
        lines = content.split('\n')
        
        # Detect figures with captions
        figures = self._extract_figures_with_captions(content, page_idx, current_section)
        elements.extend(figures)
        
        # Detect tables with captions
        tables = self._extract_tables_with_captions(content, page_idx, current_section)
        elements.extend(tables)
        
        # Remove figures and tables from content, then extract text paragraphs
        text_content = self._remove_figures_and_tables(content)
        paragraphs = self._extract_paragraphs(text_content, page_idx, current_section)
        elements.extend(paragraphs)
        
        # Add context to each element
        elements = self._add_context_to_elements(elements, content)
        
        return elements
    
    def _extract_current_section(self, content: str) -> str:
        """Extract the current section heading from content."""
        # Look for markdown headings
        headings = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
        if headings:
            # Return the first (highest-level) heading
            return headings[0].strip()
        return ""
    
    def _extract_figures_with_captions(self, content: str, page_idx: int, section: str) -> List[Dict[str, Any]]:
        """
        Extract figures with their captions.
        
        Strategy:
        1. Find all image references: ![...](images/...)
        2. Search nearby text for caption patterns: "Fig. X", "Figure X"
        3. Use fuzzy matching and proximity heuristics
        """
        figures = []
        
        # Find all image references (handle filenames with parentheses)
        # Match everything up to a file extension, then require closing paren
        image_pattern = r'!\[([^\]]*)\]\(images/(.+?\.(?:png|jpg|jpeg|gif|svg|webp))\)'
        image_matches = list(re.finditer(image_pattern, content, re.IGNORECASE))
        
        for idx, match in enumerate(image_matches):
            img_alt = match.group(1)
            img_filename = match.group(2)
            img_pos = match.start()
            
            # Search for caption in surrounding text (±300 characters)
            search_start = max(0, img_pos - 300)
            search_end = min(len(content), img_pos + 300)
            context_window = content[search_start:search_end]
            
            # Look for figure caption patterns
            caption = self._find_figure_caption(context_window, idx + 1)
            
            # If no caption found, try using alt text or generate default
            if not caption:
                caption = img_alt if img_alt else f"Figure on page {page_idx + 1}"
            
            figure_element = {
                "element_id": f"page_{page_idx + 1}_fig_{idx + 1}",
                "type": "figure",
                "page": page_idx + 1,
                "content": caption,
                "context": "",  # Will be filled by _add_context_to_elements
                "metadata": {
                    "section": section,
                    "image_path": f"images/{img_filename}",
                    "caption": caption,
                    "alt_text": img_alt,
                    "position_in_page": img_pos
                }
            }
            figures.append(figure_element)
        
        return figures
    
    def _find_figure_caption(self, text: str, fig_num: int) -> str:
        """
        Find figure caption using pattern matching.
        
        Patterns:
        - Fig. X: Caption text
        - Figure X: Caption text
        - Fig.X Caption text
        """
        # Try multiple patterns
        patterns = [
            rf'Fig\.?\s*{fig_num}[:\s]+([^\n]+)',
            rf'Figure\s+{fig_num}[:\s]+([^\n]+)',
            rf'Fig\.?\s*\d+[:\s]+([^\n]+)',  # Any figure number
            rf'Figure\s+\d+[:\s]+([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                caption = match.group(1).strip()
                # Clean up caption (remove extra whitespace)
                caption = re.sub(r'\s+', ' ', caption)
                return caption
        
        return ""
    
    def _extract_tables_with_captions(self, content: str, page_idx: int, section: str) -> List[Dict[str, Any]]:
        """
        Extract markdown tables with their captions.
        
        Strategy:
        1. Find markdown tables (| --- | --- |)
        2. Search for "Table X" captions nearby
        3. Extract table content
        """
        tables = []
        
        # Find markdown tables using regex
        # Pattern: Lines starting with | and containing separator line
        table_pattern = r'(\|[^\n]+\|\n\|[-\s|]+\|(?:\n\|[^\n]+\|)+)'
        table_matches = list(re.finditer(table_pattern, content, re.MULTILINE))
        
        for idx, match in enumerate(table_matches):
            table_markdown = match.group(1).strip()
            table_pos = match.start()
            
            # Search for caption in surrounding text (±200 characters before table)
            search_start = max(0, table_pos - 200)
            search_end = table_pos
            context_window = content[search_start:search_end]
            
            # Look for table caption
            caption = self._find_table_caption(context_window, idx + 1)
            
            if not caption:
                caption = f"Table {idx + 1} on page {page_idx + 1}"
            
            table_element = {
                "element_id": f"page_{page_idx + 1}_table_{idx + 1}",
                "type": "table",
                "page": page_idx + 1,
                "content": caption,  # Caption as primary content
                "context": "",  # Will be filled by _add_context_to_elements
                "metadata": {
                    "section": section,
                    "table_markdown": table_markdown,
                    "caption": caption,
                    "position_in_page": table_pos
                }
            }
            tables.append(table_element)
        
        return tables
    
    def _find_table_caption(self, text: str, table_num: int) -> str:
        """
        Find table caption using pattern matching.
        
        Patterns:
        - Table X: Caption text
        - Table X Caption text
        """
        patterns = [
            rf'Table\s+{table_num}[:\s]+([^\n]+)',
            rf'Table\s+\d+[:\s]+([^\n]+)',  # Any table number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                caption = match.group(1).strip()
                caption = re.sub(r'\s+', ' ', caption)
                return caption
        
        return ""
    
    def _remove_figures_and_tables(self, content: str) -> str:
        """
        Remove figure and table markdown from content to extract pure text.
        """
        # Remove image references - use greedy match to handle filenames with parentheses
        content = re.sub(r'!\[([^\]]*)\]\(images/[^)]+\.(?:png|jpg|jpeg|gif|svg|webp)\)', '', content, flags=re.IGNORECASE)
        
        # Remove any leftover image fragments like "](images/...)"
        content = re.sub(r'\]\(images/[^)]+\.(?:png|jpg|jpeg|gif|svg|webp)\)', '', content, flags=re.IGNORECASE)
        
        # Remove orphaned image markdown and leftover fragments
        content = re.sub(r'!\[[^\]]*\]', '', content)
        
        # Remove any remaining image filename fragments (ends with image extensions)
        content = re.sub(r'[-_\w()]+\.(?:png|jpg|jpeg|gif|svg|webp)\)', '', content, flags=re.IGNORECASE)
        
        # Remove standalone Fig./Figure captions (will be handled separately)
        content = re.sub(r'^Fig\.?\s*\d+.*?$', '', content, flags=re.MULTILINE)
        
        # Remove markdown tables
        content = re.sub(r'\|[^\n]+\|\n\|[-\s|]+\|(?:\n\|[^\n]+\|)+', '', content, flags=re.MULTILINE)
        
        # Clean up multiple blank lines left by removals
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content
    
    def _extract_paragraphs(self, content: str, page_idx: int, section: str) -> List[Dict[str, Any]]:
        """
        Extract text paragraphs from content.
        
        Strategy:
        1. Split by double newlines
        2. Merge standalone equations with preceding text
        3. Filter out short fragments
        4. Preserve headings
        """
        paragraphs = []
        
        # Split by double newlines
        blocks = re.split(r'\n\s*\n', content)
        
        # Merge standalone equations with previous block
        merged_blocks = []
        i = 0
        while i < len(blocks):
            block = blocks[i].strip()
            
            # Skip empty blocks
            if not block:
                i += 1
                continue
            
            # Check if this is a standalone equation (starts with \[ or \()
            is_equation = re.match(r'^\\[\[\(]', block) and len(block) < 200
            
            # If it's a standalone equation and we have a previous block, merge
            if is_equation and merged_blocks:
                # Merge with previous block
                merged_blocks[-1] = merged_blocks[-1] + ' ' + block
            else:
                merged_blocks.append(block)
            
            i += 1
        
        # Now process merged blocks
        para_idx = 0
        for block in merged_blocks:
            block = block.strip()
            
            # Skip empty blocks or very short fragments (< 20 chars)
            if not block or len(block) < 20:
                continue
            
            # Check if this is a heading
            is_heading = block.startswith('#')
            
            # Skip standalone figure/table captions
            if re.match(r'^(Fig\.?|Figure|Table)\s+\d+', block, re.IGNORECASE):
                continue
            
            para_idx += 1
            paragraph_element = {
                "element_id": f"page_{page_idx + 1}_para_{para_idx}",
                "type": "heading" if is_heading else "text",
                "page": page_idx + 1,
                "content": block,
                "context": "",
                "metadata": {
                    "section": section,
                    "is_heading": is_heading
                }
            }
            paragraphs.append(paragraph_element)
        
        return paragraphs
    
    def _add_context_to_elements(self, elements: List[Dict[str, Any]], full_content: str) -> List[Dict[str, Any]]:
        """
        Add surrounding context to each element.
        
        Strategy:
        - For figures/tables: Get 2 paragraphs before and after
        - For text: Get 1 paragraph before and after
        """
        # Split content into paragraphs for context extraction
        all_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', full_content) if p.strip()]
        
        for element in elements:
            # For figures and tables, extract more context
            if element['type'] in ['figure', 'table']:
                position = element['metadata'].get('position_in_page', 0)
                
                # Find which paragraph contains this position
                context_paras = []
                cumulative_pos = 0
                target_para_idx = 0
                
                for idx, para in enumerate(all_paragraphs):
                    cumulative_pos += len(para)
                    if cumulative_pos >= position:
                        target_para_idx = idx
                        break
                
                # Get surrounding paragraphs (2 before, 2 after)
                start_idx = max(0, target_para_idx - 2)
                end_idx = min(len(all_paragraphs), target_para_idx + 3)
                context_paras = all_paragraphs[start_idx:end_idx]
                
                # Join and clean
                context = ' '.join(context_paras)
                context = re.sub(r'\s+', ' ', context).strip()
                
                # Limit context length
                if len(context) > 500:
                    context = context[:500] + "..."
                
                element['context'] = context
        
        return elements
    
    # ========================================================================
    # PHASE 3: Enrich with VLM/LLM (Stubs for now)
    # ========================================================================
    
    def _initialize_vlm(self) -> None:
        """Initialize Qwen2-VL-2B-Instruct model for figure descriptions."""
        if not VLM_AVAILABLE:
            logger.error("VLM dependencies not available. Install: pip install torch transformers accelerate")
            self.use_vlm = False
            return
        
        try:
            logger.info("Loading Qwen2-VL-2B-Instruct model...")
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load Qwen2-VL model
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            self.vlm_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set to eval mode
            self.vlm_model.eval()
            
            logger.info("Qwen2-VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model: {e}")
            logger.warning("Disabling VLM enrichment")
            self.use_vlm = False
    
    def _describe_figure(self, image_path: str, caption: str, context: str) -> str:
        """
        Generate description for a figure using Qwen2-VL-2B-Instruct.
        
        Args:
            image_path: Path to the image file
            caption: Extracted caption from OCR
            context: Surrounding text context
        
        Returns:
            Concise scientific description of the figure
        """
        if not self.use_vlm or self.vlm_model is None:
            return ""
        
        try:
            # Resolve full path (image_path is relative like "images/...")
            full_image_path = self.base_dir / image_path
            
            if not full_image_path.exists():
                logger.warning(f"Image not found: {full_image_path}")
                self.vlm_stats["failed"] += 1
                return ""
            
            image = Image.open(full_image_path).convert('RGB')
            
            # Enhanced scientific prompt with watermark filtering
            prompt = f"""You are analyzing a scientific figure from a research paper.
Describe this plot objectively using only what you see in the visualization.
DO NOT read any copyright text, watermarks, publication dates, or journal names.

Caption from paper: {caption}

Focus on:
1. Plot type and structure
2. Axes labels and units (if visible)
3. Data trends, patterns, or key observations
4. Variable relationships shown

Provide 2-3 concise, factual sentences describing the visual content."""
            
            # Qwen2-VL uses conversation format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.vlm_processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.vlm_processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.7
                )
            
            # Decode output - trim input tokens from generated sequence
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            description = self.vlm_processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            description = description.strip()
            
            if description:
                self.vlm_stats["enriched"] += 1
                logger.debug(f"Generated description for {image_path}: {description[:100]}...")
            else:
                self.vlm_stats["failed"] += 1
            
            return description
            
        except Exception as e:
            logger.error(f"Error describing figure {image_path}: {e}")
            self.vlm_stats["failed"] += 1
            return ""
    
    
    def _summarize_table(self, table_markdown: str, caption: str, context: str) -> str:
        """
        Generate summary for a table using LLM (GPT-4 or Claude).
        
        TODO: Implement LLM integration
        """
        logger.debug(f"TODO: Summarize table")
        return f"[Table summary placeholder]"
    
    def _enrich_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich figure and table elements with VLM/LLM descriptions.
        
        Processes figures in batches to manage GPU memory.
        """
        logger.info("Phase 3: Enriching elements with VLM/LLM descriptions...")
        
        if not self.use_vlm:
            logger.warning("VLM enrichment disabled - skipping figure descriptions")
            return elements
        
        # Count figures to process
        figure_count = sum(1 for el in elements if el['type'] == 'figure')
        logger.info(f"Processing {figure_count} figures with Qwen2-VL...")
        
        # Reset stats
        self.vlm_stats = {"processed": 0, "enriched": 0, "failed": 0}
        
        # Process each figure
        for element in elements:
            if element['type'] == 'figure':
                image_path = element['metadata'].get('image_path', '')
                caption = element['metadata'].get('caption', '')
                context = element.get('context', '')
                
                # Generate VLM description
                description = self._describe_figure(image_path, caption, context)
                
                # Add to metadata
                if description:
                    element['metadata']['vlm_description'] = description
                    # Also append to content for RAG retrieval
                    element['content'] = f"{caption}\n\nVLM description: {description}"
                
                self.vlm_stats["processed"] += 1
                if self.vlm_stats["processed"] % 5 == 0:
                    logger.info(f"Processed {self.vlm_stats['processed']}/{figure_count} figures")
                
                # Clear GPU cache periodically
                if self.device == "cuda" and self.vlm_stats["processed"] % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Log summary
        logger.info(
            f"VLM Summary: {self.vlm_stats['processed']} processed, "
            f"{self.vlm_stats['enriched']} enriched, "
            f"{self.vlm_stats['failed']} failed"
        )
        
        return elements
    
    # ========================================================================
    # PHASE 4: Assemble JSONL
    # ========================================================================
    
    def _save_jsonl(self, elements: List[Dict[str, Any]], output_path: Path) -> None:
        """Save elements as JSONL file."""
        logger.info(f"Phase 4: Writing JSONL to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for element in elements:
                f.write(json.dumps(element, ensure_ascii=False) + '\n')
        
        logger.info(f"Wrote {len(elements)} records to {output_path}")
    
    def _save_debug_markdown(self, elements: List[Dict[str, Any]], output_path: Path) -> None:
        """Save elements as human-readable Markdown for debugging."""
        md_path = output_path.with_suffix('.debug.md')
        logger.info(f"Saving debug Markdown to {md_path}")
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# MM-RAG Debug Output\n\n")
            f.write(f"Total elements: {len(elements)}\n\n")
            f.write("---\n\n")
            
            for elem in elements:
                f.write(f"## Element: {elem['element_id']}\n\n")
                f.write(f"**Type:** {elem['type']}  \n")
                f.write(f"**Page:** {elem['page']}  \n\n")
                f.write(f"**Content:**\n\n{elem['content']}\n\n")
                f.write("---\n\n")
        
        logger.info(f"Debug Markdown saved")
    
    # ========================================================================
    # Main Processing Pipeline
    # ========================================================================
    
    def convert_one(self, pdf_path: str) -> Optional[Path]:
        """
        Convert a single PDF to JSONL.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Path to output JSONL file
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Phase 1: Extract OCR content
            api_result = self._extract_ocr_content(pdf_path)
            
            # Phase 2: Parse into structured elements
            elements = self._parse_content_to_elements(pdf_path, api_result)
            
            # Phase 3: Enrich with VLM/LLM (TODO)
            elements = self._enrich_elements(elements)
            
            # Phase 4: Save JSONL
            pdf_stem = Path(pdf_path).stem
            output_path = self.output_dir / f"{pdf_stem}.jsonl"
            self._save_jsonl(elements, output_path)
            
            # Save debug Markdown
            self._save_debug_markdown(elements, output_path)
            
            logger.info(f"Successfully converted {pdf_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}", exc_info=True)
            return None
    
    def process_all_pdfs(self) -> List[Path]:
        """Process all PDFs in the input directory."""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.info(f"No PDF files found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        outputs: List[Path] = []
        
        for pdf in pdf_files:
            result = self.convert_one(str(pdf))
            if result:
                outputs.append(result)
        
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to structured JSONL for multimodal RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in data/input/
  python pdf_to_mmrag_json.py
  
  # Process specific PDF
  python pdf_to_mmrag_json.py --input data/input/paper.pdf
  
  # Use custom output directory
  python pdf_to_mmrag_json.py --output data/custom-output/
        """
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input PDF file (if not specified, processes all PDFs in data/input/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/mmrag-output',
        help='Output directory for JSONL files (default: data/mmrag-output/)'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default=None,
        help='DeepSeek-OCR API URL (default: from DEEPSEEK_OCR_API env var or http://localhost:8001)'
    )
    parser.add_argument(
        '--use-vlm',
        action='store_true',
        help='Enable Qwen2-VL for figure descriptions (requires GPU, ~5GB model download on first run)'
    )
    
    args = parser.parse_args()
    
    # Get API URL
    api_url = args.api_url or os.environ.get('DEEPSEEK_OCR_API', 'http://localhost:8001')
    
    # Print header
    print(f"{Colors.CYAN}╔════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.CYAN}║  PDF to MM-RAG JSONL Converter                            ║{Colors.RESET}")
    print(f"{Colors.CYAN}║  Scientific Document → Structured Data for RAG            ║{Colors.RESET}")
    print(f"{Colors.CYAN}╚════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print()
    print(f"{Colors.YELLOW}Directory Structure:{Colors.RESET}")
    print(f"  - Input:  ./data/input/         # Put your PDFs here")
    print(f"  - Output: ./data/mmrag-output/  # JSONL files saved here")
    print(f"  - Images: ./data/images/        # Extracted figures")
    print()
    print(f"{Colors.YELLOW}Using API URL: {api_url}{Colors.RESET}")
    if args.use_vlm:
        print(f"{Colors.CYAN}VLM Enrichment: ENABLED (Qwen2-VL-2B-Instruct){Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}VLM Enrichment: DISABLED (use --use-vlm to enable){Colors.RESET}")
    print()
    
    try:
        # Initialize processor
        if args.input:
            # Process single file
            base_dir = Path(args.input).parent.parent if Path(args.input).parent.name in ['input', 'data'] else 'data'
            proc = PDFToMMRAGProcessor(base_dir=str(base_dir), api_base_url=api_url, use_vlm=args.use_vlm)
            
            print(f"{Colors.YELLOW}Processing single PDF: {args.input}{Colors.RESET}")
            result = proc.convert_one(args.input)
            
            if result:
                print(f"\n{Colors.GREEN}✓ Successfully processed:{Colors.RESET}")
                print(f"  JSONL: {result}")
                print(f"  Debug: {result.with_suffix('.debug.md')}")
            else:
                print(f"\n{Colors.RED}✗ Processing failed{Colors.RESET}")
                sys.exit(1)
        else:
            # Process all PDFs in input directory
            proc = PDFToMMRAGProcessor(api_base_url=api_url, use_vlm=args.use_vlm)
            
            print(f"{Colors.YELLOW}Scanning for PDF files...{Colors.RESET}")
            results = proc.process_all_pdfs()
            
            if results:
                print(f"\n{Colors.GREEN}✓ Successfully processed {len(results)} PDF files:{Colors.RESET}")
                for jsonl_path in results:
                    print(f"  - {jsonl_path}")
            else:
                print(f"\n{Colors.YELLOW}No PDF files were processed.{Colors.RESET}")
    
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to the API server")
        print(f"\n{Colors.RED}✗ Error: Cannot connect to API at {api_url}{Colors.RESET}")
        print(f"{Colors.YELLOW}Make sure the DeepSeek-OCR server is running.{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n{Colors.RED}✗ Error: {str(e)}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
