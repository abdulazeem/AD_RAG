# evaluation/llm_evaluator.py

import pandas as pd
import re
from typing import Dict, Any, List
from retrieval.retrieval_pipeline import RetrievalPipeline
from generation.generator import Generator
from config.settings import settings
from evaluation.schemas import EvaluationMetrics, EvaluationResult


class LLMEvaluator:
    """Evaluate RAG system using LLM-based metrics."""

    def __init__(self, backend: str = None):
        """
        Initialize LLM evaluator.

        Args:
            backend: "openai" or "ollama"
        """
        self.backend = backend or settings.llm_backend
        self.pipeline = RetrievalPipeline(backend=self.backend)
        self.generator = Generator(backend=self.backend)

        # Load evaluation prompt template
        self.evaluation_prompt_template = self._load_evaluation_prompt()

    def generate_rag_answers(
        self,
        questions: List[str],
        document_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate RAG system answers for questions.

        Args:
            questions: List of questions
            document_filter: Optional document filter

        Returns:
            List of results with answers and contexts
        """
        results = []

        for idx, question in enumerate(questions, 1):
            print(f"[LLMEvaluator] Generating answer {idx}/{len(questions)}...")

            try:
                # Retrieve contexts
                chunks = self.pipeline.run(question, document_filter=document_filter)

                # Generate answer
                answer, _ = self.generator.generate_answer(question, chunks)

                # Extract contexts
                contexts = [chunk['text'] for chunk in chunks]

                results.append({
                    'question': question,
                    'answer': answer,
                    'contexts': contexts,
                    'contexts_count': len(contexts)
                })

            except Exception as e:
                print(f"[LLMEvaluator] Error generating answer {idx}: {e}")
                results.append({
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'contexts': [],
                    'contexts_count': 0
                })

        return results

    def _load_evaluation_prompt(self) -> str:
        """Load evaluation prompt template."""
        return """You are an expert evaluator tasked with comparing an AI-generated response against a ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth_answer}

AI Response: {ai_response}

Please evaluate the AI response using the following criteria:

1. **FAITHFULNESS** (True/False):
   - Set to TRUE if the AI response contains the same core facts and meaning as the ground truth
   - The AI response can be MORE DETAILED or have additional context - this is still faithful as long as it doesn't contradict the ground truth
   - Only set to FALSE if the AI response contains incorrect facts or contradicts the ground truth

2. **CORRECTNESS** (0-10):
   - 0-2: Completely incorrect or irrelevant
   - 3-4: Mostly incorrect with some relevant information
   - 5-6: Partially correct but missing key information
   - 7-8: Mostly correct with minor issues or missing details
   - 9-10: Fully correct and comprehensive (can be more detailed than ground truth)

3. **JUSTIFICATION**: Brief explanation of your scores

IMPORTANT: Respond with EXACTLY this format using pipe delimiters:
FAITHFULNESS: true/false
CORRECTNESS: 0-10
JUSTIFICATION: your explanation here

Do not use any other format. Start your response with "FAITHFULNESS:"."""

    def evaluate_response(
        self,
        question: str,
        ground_truth_answer: str,
        ai_response: str
    ) -> EvaluationMetrics:
        """
        Evaluate AI response against ground truth using LLM.

        Args:
            question: The question
            ground_truth_answer: Ground truth answer
            ai_response: AI generated response

        Returns:
            EvaluationMetrics with faithfulness, correctness, and justification
        """
        evaluation_prompt = self.evaluation_prompt_template.format(
            question=question,
            ground_truth_answer=ground_truth_answer,
            ai_response=ai_response
        )

        try:
            # Generate evaluation
            response, _ = self.generator.llm.generate(evaluation_prompt)
            response_clean = response.strip()

            print(f"[LLMEvaluator] Raw evaluation response:\n{response_clean}\n")

            # Parse using delimiter-based approach
            faithfulness = False
            correctness = 0
            justification = "Unable to parse evaluation"

            # Extract faithfulness
            if "FAITHFULNESS:" in response_clean:
                faith_line = response_clean.split("FAITHFULNESS:")[1].split("\n")[0].strip().lower()
                faithfulness = "true" in faith_line

            # Extract correctness
            if "CORRECTNESS:" in response_clean:
                corr_line = response_clean.split("CORRECTNESS:")[1].split("\n")[0].strip()
                # Extract just the number
                corr_match = re.search(r'\d+', corr_line)
                if corr_match:
                    correctness = min(10, max(0, int(corr_match.group())))

            # Extract justification
            if "JUSTIFICATION:" in response_clean:
                just_part = response_clean.split("JUSTIFICATION:")[1].strip()
                # Take everything after JUSTIFICATION: (may span multiple lines)
                justification = just_part

            # Create Pydantic model
            metrics = EvaluationMetrics(
                faithfulness=faithfulness,
                correctness=correctness,
                justification=justification,
                raw_response=response_clean
            )

            return metrics

        except Exception as e:
            print(f"[LLMEvaluator] Error parsing evaluation: {e}")
            if 'response' in locals():
                print(f"[LLMEvaluator] Raw response: {response}")

            # Fallback: Try to make a best-guess evaluation
            try:
                response_to_parse = response if 'response' in locals() else ""
                # Look for keywords in the response
                response_lower = response_to_parse.lower()

                # Try to determine faithfulness
                if any(word in response_lower for word in ["faithful", "accurate", "correct", "matches", "consistent"]):
                    faithfulness = True
                else:
                    faithfulness = False

                # Try to find a score
                scores = re.findall(r'\b([0-9]|10)\b', response_to_parse)
                if scores:
                    correctness = int(scores[0])
                else:
                    correctness = 5  # Default middle score

                justification = f"Parsed from LLM response (parsing error occurred): {response_to_parse[:200]}"

                return EvaluationMetrics(
                    faithfulness=faithfulness,
                    correctness=correctness,
                    justification=justification,
                    raw_response=response_to_parse
                )
            except Exception as fallback_error:
                # Ultimate fallback
                return EvaluationMetrics(
                    faithfulness=False,
                    correctness=0,
                    justification=f"Error during evaluation: {str(e)}, Fallback error: {str(fallback_error)}",
                    raw_response=""
                )

    def export_results_to_excel(
        self,
        evaluation_results: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        Export evaluation results to Excel file.

        Args:
            evaluation_results: List of evaluation results
            output_path: Path to save Excel file
        """
        import os

        # Prepare data for Excel
        excel_data = []
        for idx, result in enumerate(evaluation_results, 1):
            metrics = result['metrics']
            excel_data.append({
                'Question #': idx,
                'Question': result['question'],
                'Ground Truth Answer': result['ground_truth_answer'],
                'AI Response': result['ai_response'],
                'Contexts Used': result['contexts_used'],
                'Faithfulness': '✅ True' if metrics['faithfulness'] else '❌ False',
                'Correctness Score (0-10)': metrics['correctness'],
                'Justification': metrics['justification'],
                'Raw LLM Response': metrics.get('raw_response', '')
            })

        # Create DataFrame and save
        df = pd.DataFrame(excel_data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to Excel with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Evaluation Results', index=False)

            # Get worksheet for formatting
            worksheet = writer.sheets['Evaluation Results']

            # Adjust column widths
            worksheet.column_dimensions['A'].width = 12   # Question #
            worksheet.column_dimensions['B'].width = 50   # Question
            worksheet.column_dimensions['C'].width = 50   # Ground Truth
            worksheet.column_dimensions['D'].width = 50   # AI Response
            worksheet.column_dimensions['E'].width = 15   # Contexts Used
            worksheet.column_dimensions['F'].width = 15   # Faithfulness
            worksheet.column_dimensions['G'].width = 20   # Correctness Score
            worksheet.column_dimensions['H'].width = 60   # Justification
            worksheet.column_dimensions['I'].width = 70   # Raw LLM Response

        print(f"[LLMEvaluator] Exported results to: {output_path}")

    def run_full_evaluation(
        self,
        ground_truth_path: str,
        document_filter: List[str] = None,
        export_path: str = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            ground_truth_path: Path to ground truth Excel file
            document_filter: Optional document filter
            export_path: Optional path to export results to Excel
            ground_truth_path: Path to ground truth Excel file
            document_filter: Optional document filter

        Returns:
            Complete evaluation results
        """
        print(f"[LLMEvaluator] Loading ground truth from: {ground_truth_path}")

        # Load ground truth
        ground_truth_df = pd.read_excel(ground_truth_path)

        # Extract questions
        questions = ground_truth_df['question'].tolist()

        # Generate RAG answers
        print(f"[LLMEvaluator] Generating RAG answers...")
        rag_results = self.generate_rag_answers(questions, document_filter)

        # Evaluate each answer
        print(f"[LLMEvaluator] Evaluating answers...")
        evaluation_results = []

        for idx, (row_idx, row) in enumerate(ground_truth_df.iterrows()):
            if idx < len(rag_results):
                result = rag_results[idx]

                print(f"[LLMEvaluator] Evaluating {idx + 1}/{len(questions)}...")

                # Evaluate using LLM
                metrics = self.evaluate_response(
                    question=row['question'],
                    ground_truth_answer=row['ground_truth_answer'],
                    ai_response=result['answer']
                )

                eval_result = EvaluationResult(
                    question=row['question'],
                    ground_truth_answer=row['ground_truth_answer'],
                    ai_response=result['answer'],
                    metrics=metrics,
                    contexts_used=result['contexts_count']
                )

                evaluation_results.append(eval_result.model_dump())

        # Calculate aggregate metrics
        total = len(evaluation_results)
        faithful_count = sum(1 for r in evaluation_results if r['metrics']['faithfulness'])
        avg_correctness = sum(r['metrics']['correctness'] for r in evaluation_results) / total if total > 0 else 0

        aggregate_metrics = {
            'total_questions': total,
            'faithful_count': faithful_count,
            'faithful_percentage': (faithful_count / total * 100) if total > 0 else 0,
            'average_correctness': round(avg_correctness, 2),
            'correctness_distribution': self._calculate_distribution(
                [r['metrics']['correctness'] for r in evaluation_results]
            )
        }

        # Export to Excel if path provided
        if export_path:
            self.export_results_to_excel(evaluation_results, export_path)

        return {
            'ground_truth_count': len(ground_truth_df),
            'evaluation_results': evaluation_results,
            'aggregate_metrics': aggregate_metrics,
            'export_path': export_path
        }

    def _calculate_distribution(self, scores: List[int]) -> Dict[str, int]:
        """Calculate distribution of correctness scores."""
        distribution = {
            '0-2': 0,
            '3-4': 0,
            '5-6': 0,
            '7-8': 0,
            '9-10': 0
        }

        for score in scores:
            if score <= 2:
                distribution['0-2'] += 1
            elif score <= 4:
                distribution['3-4'] += 1
            elif score <= 6:
                distribution['5-6'] += 1
            elif score <= 8:
                distribution['7-8'] += 1
            else:
                distribution['9-10'] += 1

        return distribution
