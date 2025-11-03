# rag_app/ingestion/docling_loader.py

import os
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

def convert_document(source: str) -> Dict[str, Any]:
    """
    Converts a single document file (or URL) using Docling,
    returns a dict with metadata, text content, and page-level information.
    """
    # Configure pipeline for better OCR and text extraction
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR
    pipeline_options.do_table_structure = True  # Extract tables

    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(source)
    doc = result.document

    # Debug: Print document structure
    print(f"[DoclingLoader] Document type: {type(doc)}")
    print(f"[DoclingLoader] Pages in document: {doc.num_pages if hasattr(doc, 'num_pages') else 'N/A'}")

    # Try multiple export methods to get content
    content = ""
    try:
        # Try markdown export first
        content = doc.export_to_markdown()
    except Exception as e:
        print(f"[DoclingLoader] Markdown export failed: {e}, trying text export")
        try:
            # Fallback to text export
            content = doc.export_to_text()
        except Exception as e2:
            print(f"[DoclingLoader] Text export also failed: {e2}")
            # Last resort: try to get raw content
            if hasattr(doc, 'text'):
                content = doc.text
            elif hasattr(doc, 'content'):
                content = str(doc.content)

    # Check if content is empty or too short
    if not content or len(content.strip()) < 100:
        print(f"[DoclingLoader] WARNING: Extracted content is too short ({len(content)} chars)")
        print(f"[DoclingLoader] Attempting alternative extraction methods...")

        # Try to extract from document structure directly
        try:
            all_content_parts = []

            # Method 1: Extract from text items
            if hasattr(doc, 'texts') and doc.texts:
                print(f"[DoclingLoader] Found {len(doc.texts)} text items")
                for text_item in doc.texts:
                    if hasattr(text_item, 'text') and text_item.text:
                        all_content_parts.append(text_item.text)

            # Method 2: Extract from tables
            if hasattr(doc, 'tables') and doc.tables:
                print(f"[DoclingLoader] Found {len(doc.tables)} table items")
                for table_idx, table in enumerate(doc.tables):
                    # Try to get table markdown representation
                    table_text = None
                    try:
                        if hasattr(table, 'export_to_markdown'):
                            table_text = table.export_to_markdown()
                        elif hasattr(table, 'export_to_text'):
                            table_text = table.export_to_text()
                        elif hasattr(table, 'data') and table.data:
                            # Try to manually extract table data
                            if hasattr(table.data, 'grid') and table.data.grid:
                                table_rows = []
                                for row in table.data.grid:
                                    row_text = " | ".join(str(cell) for cell in row if cell)
                                    if row_text.strip():
                                        table_rows.append(row_text)
                                if table_rows:
                                    table_text = "\n".join(table_rows)
                    except Exception as te:
                        print(f"[DoclingLoader] Error extracting table {table_idx}: {te}")

                    if table_text and table_text.strip():
                        all_content_parts.append(f"\n### Table {table_idx + 1}\n{table_text}")

            # Method 3: Try OCR on pictures if available
            if hasattr(doc, 'pictures') and doc.pictures:
                print(f"[DoclingLoader] Found {len(doc.pictures)} picture items (OCR extraction not implemented)")

            # Combine all extracted content
            if all_content_parts:
                content = "\n\n".join(all_content_parts)
                print(f"[DoclingLoader] Combined extraction: {len(content)} chars from {len(all_content_parts)} parts")

            # Method 4: Last resort - iterate through body
            if not content or len(content.strip()) < 100:
                if hasattr(doc, 'body') and hasattr(doc.body, 'iterate_items'):
                    body_texts = []
                    for item in doc.body.iterate_items():
                        item_text = None
                        if hasattr(item, 'text'):
                            item_text = item.text
                        elif hasattr(item, 'export_to_text'):
                            item_text = item.export_to_text()
                        elif hasattr(item, 'export_to_markdown'):
                            item_text = item.export_to_markdown()

                        if item_text and item_text.strip():
                            body_texts.append(item_text)

                    if body_texts:
                        content = "\n\n".join(body_texts)
                        print(f"[DoclingLoader] Extracted from body iteration: {len(content)} chars")

        except Exception as e:
            print(f"[DoclingLoader] Error extracting from document structure: {e}")
            import traceback
            traceback.print_exc()

    # Remove image placeholder tags
    content = content.replace("<!-- image -->", "").replace("<!-- image-->", "")

    # Extract metadata safely - DoclingDocument structure varies by version
    metadata = {"source": source}

    # Try to extract metadata from various possible locations
    try:
        # Check if result has metadata
        if hasattr(result, 'metadata') and result.metadata:
            metadata["title"] = getattr(result.metadata, "title", None)
            metadata["author"] = getattr(result.metadata, "author", None)
            metadata["page_count"] = getattr(result.metadata, "page_count", None)
        # Check if doc has direct attributes
        elif hasattr(doc, 'name'):
            metadata["title"] = getattr(doc, "name", None)
    except Exception as e:
        # If metadata extraction fails, just use the source
        print(f"[DoclingLoader] Could not extract metadata: {e}")

    # Extract page-level content - Simple approach: extract text per page
    pages_content = []

    try:
        if hasattr(result, 'pages') and result.pages:
            print(f"[DoclingLoader] Extracting {len(result.pages)} pages from result.pages")

            # Process each page
            for page_obj in result.pages:
                page_no = page_obj.page_no if hasattr(page_obj, 'page_no') else None

                # Export page text by filtering document items to this page
                page_text = ""
                try:
                    if hasattr(doc, 'body') and hasattr(doc.body, 'iterate_items'):
                        page_items = []
                        for item in doc.body.iterate_items():
                            # Check if item belongs to this page
                            item_page_no = None
                            if hasattr(item, 'prov') and item.prov:
                                for prov in item.prov:
                                    if hasattr(prov, 'page_no') and prov.page_no == page_no:
                                        item_page_no = page_no
                                        break

                            if item_page_no == page_no:
                                # Get text from this item
                                item_text = None
                                if hasattr(item, 'text') and item.text:
                                    item_text = item.text
                                elif hasattr(item, 'export_to_markdown'):
                                    try:
                                        item_text = item.export_to_markdown()
                                    except:
                                        pass

                                if item_text and item_text.strip():
                                    page_items.append(item_text)

                        if page_items:
                            page_text = "\n\n".join(page_items)

                except Exception as e:
                    print(f"[DoclingLoader] Error extracting page {page_no}: {e}")

                if page_text and page_text.strip():
                    # Store in pages_content
                    pages_content.append({
                        "page_number": page_no,
                        "content": page_text
                    })

            if pages_content:
                metadata["total_pages"] = len(pages_content)
                print(f"[DoclingLoader] Extracted {len(pages_content)} pages")

    except Exception as e:
        print(f"[DoclingLoader] Could not extract page-level content: {e}")
        import traceback
        traceback.print_exc()
        # If page extraction fails, we still have the full content

    return {
        "metadata": metadata,
        "content": content,
        "pages": pages_content  # List of page-level content with page numbers
    }

def convert_folder(input_folder: str, output_folder: str) -> List[Dict[str, Any]]:
    """
    Processes all files in input_folder using Docling,
    writes each converted document to output_folder as markdown,
    returns list of conversion results.
    """
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for fname in os.listdir(input_folder):
        path = os.path.join(input_folder, fname)
        if os.path.isfile(path):
            try:
                result = convert_document(path)
                md_filename = os.path.splitext(fname)[0] + ".md"
                output_path = os.path.join(output_folder, md_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result["content"])
                results.append(result)
                print(f"[DoclingLoader] Converted {fname} → {md_filename}")
            except Exception as e:
                print(f"[DoclingLoader] Failed to convert {fname}: {e}")
    return results

if __name__ == "__main__":
    in_folder = "data/raw_docs"
    out_folder = "data/processed_docs"
    converted = convert_folder(in_folder, out_folder)
    print(f"Converted {len(converted)} documents.")
