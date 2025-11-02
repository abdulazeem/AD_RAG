# evaluation/schemas.py

from pydantic import BaseModel, Field
from typing import List


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for comparing AI response with ground truth."""

    faithfulness: bool = Field(
        ...,
        description="Whether the AI response is faithful to the ground truth (True/False)"
    )
    correctness: int = Field(
        ...,
        ge=0,
        le=10,
        description="Correctness score from 0-10, comparing AI response with ground truth"
    )
    justification: str = Field(
        ...,
        description="Textual explanation justifying the faithfulness and correctness scores"
    )


class QuestionGroundTruth(BaseModel):
    """Question and ground truth answer pair."""

    question: str = Field(..., description="The question")
    ground_truth_answer: str = Field(..., description="The ground truth answer")
    context: str = Field(..., description="The source context used to generate the Q&A")
    source_file: str = Field(default="Unknown", description="Source document file")
    page_numbers: str = Field(default="[]", description="Page numbers in source document")


class EvaluationResult(BaseModel):
    """Result of evaluating a single question."""

    question: str
    ground_truth_answer: str
    ai_response: str
    metrics: EvaluationMetrics
    contexts_used: int = Field(default=0, description="Number of context chunks used")
