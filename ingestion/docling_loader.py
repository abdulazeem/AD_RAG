# rag_app/ingestion/docling_loader.py

import os
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter

def convert_document(source: str) -> Dict[str, Any]:
    """
    Converts a single document file (or URL) using Docling,
    returns a dict with metadata and text content.
    """
    converter = DocumentConverter()
    result = converter.convert(source)
    doc = result.document
    # Use export_to_markdown or export_to_text depending on needs
    content = doc.export_to_markdown()

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

    return {"metadata": metadata, "content": content}

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
