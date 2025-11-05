# generation/api/routers/evaluation.py

import os
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel
from enum import Enum

from evaluation.ground_truth_generator import GroundTruthGenerator
from evaluation.llm_evaluator import LLMEvaluator
from config.settings import settings

router = APIRouter()


class BackendType(str, Enum):
    openai = "openai"
    ollama = "ollama"


class GroundTruthRequest(BaseModel):
    backend: str
    num_samples: int = 10
    selected_documents: Optional[List[str]] = None


class GroundTruthResponse(BaseModel):
    success: bool
    message: str
    file_path: Optional[str] = None
    num_generated: int = 0


class EvaluationRequest(BaseModel):
    ground_truth_file: str
    backend: str
    selected_documents: Optional[List[str]] = None


class EvaluationResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    ground_truth_count: int = 0
    evaluation_scores: dict = {}
    results_summary: dict = {}
    results_file_path: Optional[str] = None


class GroundTruthFile(BaseModel):
    filename: str
    filepath: str
    created_at: str
    size_kb: float


@router.post("/generate-ground-truth", response_model=GroundTruthResponse)
async def generate_ground_truth(request: GroundTruthRequest):
    """
    Generate ground truth question-answer pairs.

    Args:
        request: Ground truth generation request

    Returns:
        Generated ground truth file information
    """
    try:
        print(f"[Evaluation API] Generating ground truth with backend: {request.backend}")

        # Create generator
        generator = GroundTruthGenerator(backend=request.backend)

        # Get model name for the backend
        if request.backend == "openai":
            model_name = settings.openai.model.replace("/", "_")
        else:
            model_name = settings.ollama.model.replace("/", "_").replace(":", "_")

        # Create document identifier for filename
        if request.selected_documents and len(request.selected_documents) > 0:
            # Sanitize document name(s) for filename
            def sanitize_filename(name: str) -> str:
                # Remove extension and sanitize
                name_no_ext = os.path.splitext(name)[0]
                # Replace invalid characters with underscore
                sanitized = name_no_ext.replace(" ", "_").replace("/", "_").replace("\\", "_")
                # Limit length
                return sanitized[:50]

            if len(request.selected_documents) == 1:
                doc_part = sanitize_filename(request.selected_documents[0])
            elif len(request.selected_documents) <= 3:
                # List first few documents
                doc_names = [sanitize_filename(d) for d in request.selected_documents[:3]]
                doc_part = "_".join(doc_names)[:80]  # Limit total length
            else:
                # Many documents: show first one + count
                first_doc = sanitize_filename(request.selected_documents[0])
                doc_part = f"{first_doc}_and_{len(request.selected_documents)-1}_more"
        else:
            doc_part = "all_docs"

        # Create output path with model name and document name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ground_truth_{request.backend}_{model_name}_{doc_part}_{timestamp}.xlsx"
        output_dir = os.path.join(settings.data.base_dir, "evaluation")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Generate ground truth
        df = generator.generate_ground_truth_dataset(
            num_samples=request.num_samples,
            output_path=output_path,
            document_filter=request.selected_documents
        )

        return GroundTruthResponse(
            success=True,
            message=f"Successfully generated {len(df)} ground truth pairs",
            file_path=output_path,
            num_generated=len(df)
        )

    except Exception as e:
        print(f"[Evaluation API] Error generating ground truth: {e}")
        import traceback
        traceback.print_exc()

        return GroundTruthResponse(
            success=False,
            message=f"Error: {str(e)}",
            num_generated=0
        )


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag_system(request: EvaluationRequest):
    """
    Evaluate RAG system using LLM-based metrics.

    Args:
        request: Evaluation request with ground truth file

    Returns:
        Evaluation results
    """
    try:
        print(f"[Evaluation API] Running evaluation with backend: {request.backend}")
        print(f"[Evaluation API] Ground truth file: {request.ground_truth_file}")

        # Check if file exists
        if not os.path.exists(request.ground_truth_file):
            raise HTTPException(
                status_code=404,
                detail=f"Ground truth file not found: {request.ground_truth_file}"
            )

        # Create evaluator
        evaluator = LLMEvaluator(backend=request.backend)

        # Get model name for the backend
        if request.backend == "openai":
            model_name = settings.openai.model.replace("/", "_")
        else:
            model_name = settings.ollama.model.replace("/", "_").replace(":", "_")

        # Create export path for evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract ground truth filename without extension
        gt_basename = os.path.basename(request.ground_truth_file)
        gt_name_no_ext = os.path.splitext(gt_basename)[0]

        results_filename = f"eval_results_{request.backend}_{model_name}_{timestamp}.xlsx"
        results_dir = os.path.join(settings.data.base_dir, "evaluation", "results")
        os.makedirs(results_dir, exist_ok=True)
        export_path = os.path.join(results_dir, results_filename)

        # Run evaluation
        results = evaluator.run_full_evaluation(
            ground_truth_path=request.ground_truth_file,
            document_filter=request.selected_documents,
            export_path=export_path
        )

        # Prepare summary from aggregate metrics
        aggregate = results.get('aggregate_metrics', {})
        summary = {
            'questions_evaluated': results.get('ground_truth_count', 0),
            'faithful_count': aggregate.get('faithful_count', 0),
            'faithful_percentage': aggregate.get('faithful_percentage', 0),
            'average_correctness': aggregate.get('average_correctness', 0),
        }

        return EvaluationResponse(
            success=True,
            message="Evaluation completed successfully",
            ground_truth_count=results.get('ground_truth_count', 0),
            evaluation_scores=results,
            results_summary=summary,
            results_file_path=export_path
        )

    except Exception as e:
        print(f"[Evaluation API] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

        return EvaluationResponse(
            success=False,
            message=f"Error: {str(e)}",
            ground_truth_count=0,
            evaluation_scores={},
            results_summary={},
            results_file_path=None
        )


@router.get("/ground-truth-files", response_model=List[GroundTruthFile])
async def list_ground_truth_files():
    """
    List available ground truth files.

    Returns:
        List of ground truth files
    """
    try:
        eval_dir = os.path.join(settings.data.base_dir, "evaluation")

        if not os.path.exists(eval_dir):
            return []

        files = []
        for filename in os.listdir(eval_dir):
            if filename.endswith('.xlsx') and filename.startswith('ground_truth_'):
                filepath = os.path.join(eval_dir, filename)
                stat = os.stat(filepath)

                files.append(GroundTruthFile(
                    filename=filename,
                    filepath=filepath,
                    created_at=datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                    size_kb=stat.st_size / 1024
                ))

        # Sort by creation time (newest first)
        files.sort(key=lambda x: x.created_at, reverse=True)

        return files

    except Exception as e:
        print(f"[Evaluation API] Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


class DeleteGroundTruthResponse(BaseModel):
    success: bool
    message: str
    filename: str


@router.delete("/ground-truth-files/{filename}", response_model=DeleteGroundTruthResponse)
async def delete_ground_truth_file(filename: str):
    """
    Delete a ground truth file.

    Args:
        filename: Name of the ground truth file to delete

    Returns:
        Deletion status
    """
    try:
        eval_dir = os.path.join(settings.data.base_dir, "evaluation")
        filepath = os.path.join(eval_dir, filename)

        # Security check: ensure filename doesn't contain path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Check if file exists
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Only allow deletion of ground truth files
        if not filename.startswith('ground_truth_') or not filename.endswith('.xlsx'):
            raise HTTPException(status_code=400, detail="Can only delete ground truth Excel files")

        # Delete the file
        os.remove(filepath)

        return DeleteGroundTruthResponse(
            success=True,
            message=f"Successfully deleted {filename}",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Evaluation API] Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


class GroundTruthPreview(BaseModel):
    filename: str
    total_rows: int
    columns: List[str]
    preview_data: List[dict]


@router.get("/ground-truth-files/{filename}/preview", response_model=GroundTruthPreview)
async def preview_ground_truth_file(filename: str, rows: int = 5):
    """
    Preview a ground truth file.

    Args:
        filename: Name of the ground truth file to preview
        rows: Number of rows to preview (default: 5)

    Returns:
        Preview of the ground truth file
    """
    try:
        import pandas as pd

        eval_dir = os.path.join(settings.data.base_dir, "evaluation")
        filepath = os.path.join(eval_dir, filename)

        # Security check: ensure filename doesn't contain path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Check if file exists
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Read Excel file
        df = pd.read_excel(filepath)

        # Get preview data
        preview_rows = df.head(rows).to_dict('records')

        # Convert any non-serializable types to strings
        for row in preview_rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    row[key] = str(value)

        return GroundTruthPreview(
            filename=filename,
            total_rows=len(df),
            columns=df.columns.tolist(),
            preview_data=preview_rows
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Evaluation API] Error previewing file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error previewing file: {str(e)}")

class EvaluationResultFile(BaseModel):
    filename: str
    filepath: str
    created_at: str
    size_kb: float


@router.get("/evaluation-results", response_model=List[EvaluationResultFile])
async def list_evaluation_results():
    """
    List available evaluation result files.

    Returns:
        List of evaluation result files
    """
    try:
        results_dir = os.path.join(settings.data.base_dir, "evaluation", "results")

        if not os.path.exists(results_dir):
            return []

        files = []
        for filename in os.listdir(results_dir):
            if filename.endswith('.xlsx') and filename.startswith('eval_results_'):
                filepath = os.path.join(results_dir, filename)
                stat = os.stat(filepath)

                files.append(EvaluationResultFile(
                    filename=filename,
                    filepath=filepath,
                    created_at=datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                    size_kb=stat.st_size / 1024
                ))

        # Sort by creation time (newest first)
        files.sort(key=lambda x: x.created_at, reverse=True)

        return files

    except Exception as e:
        print(f"[Evaluation API] Error listing result files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing result files: {str(e)}")


@router.get("/evaluation-results/{filename}/preview", response_model=GroundTruthPreview)
async def preview_evaluation_results(filename: str, rows: int = 10):
    """
    Preview an evaluation results file.

    Args:
        filename: Name of the evaluation results file to preview
        rows: Number of rows to preview (default: 10)

    Returns:
        Preview of the evaluation results file
    """
    try:
        import pandas as pd

        results_dir = os.path.join(settings.data.base_dir, "evaluation", "results")
        filepath = os.path.join(results_dir, filename)

        # Security check: ensure filename doesn't contain path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Check if file exists
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        # Read Excel file
        df = pd.read_excel(filepath)

        # Get preview data
        preview_rows = df.head(rows).to_dict('records')

        # Convert any non-serializable types to strings
        for row in preview_rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    row[key] = str(value)

        return GroundTruthPreview(
            filename=filename,
            total_rows=len(df),
            columns=df.columns.tolist(),
            preview_data=preview_rows
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Evaluation API] Error previewing results file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error previewing results file: {str(e)}")
