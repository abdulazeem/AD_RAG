# evaluation/ground_truth_generator.py

import pandas as pd
import os
from typing import List, Dict, Any
from datetime import datetime
from embeddings.vector_store import VectorStore
from generation.generator import Generator
from config.settings import settings
import numpy as np


class GroundTruthGenerator:
    """Generate ground truth question-answer pairs for RAG evaluation using LLM."""

    def __init__(self, backend: str = None):
        """
        Initialize ground truth generator.

        Args:
            backend: "openai" or "ollama"
        """
        self.backend = backend or settings.llm_backend
        self.vector_store = VectorStore(backend=self.backend)
        self.generator = Generator(backend=self.backend)

    def sample_chunks(self, num_samples: int = 10, document_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        Sample random chunks from the vector store.

        Args:
            num_samples: Number of chunks to sample
            document_filter: Optional list of document filenames to filter by

        Returns:
            List of sampled chunks
        """
        random_samples = []

        # Increase over-sampling multiplier to get more diverse chunks
        # We do more rounds with higher top_k to ensure we get enough unique samples
        attempts = 0
        max_attempts = num_samples * 10  # Increased from 3 to 10

        print(f"[GroundTruthGenerator] Attempting to sample {num_samples} unique chunks...")

        while len(random_samples) < num_samples * 20 and attempts < max_attempts:
            random_vector = np.random.randn(self.vector_store.dimensions).tolist()
            results = self.vector_store.query(
                vector=random_vector,
                top_k=10,  # Increased from 5 to 10
                document_filter=document_filter
            )
            random_samples.extend(results)
            attempts += 1

            if attempts % 10 == 0:
                print(f"[GroundTruthGenerator] Collected {len(random_samples)} chunks after {attempts} attempts...")

        # Remove duplicates based on text
        seen_texts = set()
        unique_samples = []
        for sample in random_samples:
            text = sample['text']
            if text not in seen_texts and len(text) > 100:  # Filter short chunks
                seen_texts.add(text)
                unique_samples.append(sample)
                if len(unique_samples) >= num_samples:
                    break

        print(f"[GroundTruthGenerator] Found {len(unique_samples)} unique chunks (requested: {num_samples})")

        if len(unique_samples) < num_samples:
            print(f"[GroundTruthGenerator] WARNING: Could only find {len(unique_samples)} unique chunks. Try:")
            print(f"  1. Reducing num_samples")
            print(f"  2. Ingesting more documents")
            print(f"  3. Removing document filters")

        return unique_samples[:num_samples]

    def generate_question_from_context(self, context: str) -> str:
        """
        Generate a question from a context chunk using LLM.

        Args:
            context: Context text

        Returns:
            Generated question
        """
        prompt = f"""Based on the following context, generate ONE specific question that can be answered using ONLY the information in this context.

Context:
{context}

Generate a clear, specific question that:
1. Can be fully answered using the given context
2. Is not too broad or too narrow
3. Focuses on the key information in the context

Question:"""

        question, _ = self.generator.llm.generate(prompt)
        return question.strip()

    def generate_ground_truth_answer(self, question: str, context: str) -> str:
        """
        Generate ground truth answer from question and context.

        Args:
            question: The question
            context: The context containing the answer

        Returns:
            Ground truth answer
        """
        prompt = f"""Answer the following question based STRICTLY on the provided context. Be precise and accurate.

Context:
{context}

Question: {question}

Answer:"""

        answer, _ = self.generator.llm.generate(prompt)
        return answer.strip()

    def generate_ground_truth_dataset(
        self,
        num_samples: int = 10,
        output_path: str = None,
        document_filter: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate complete ground truth dataset.

        Args:
            num_samples: Number of question-answer pairs to generate
            output_path: Path to save Excel file
            document_filter: Optional list of document filenames to filter by

        Returns:
            DataFrame with ground truth data
        """
        print(f"[GroundTruthGenerator] Sampling {num_samples} chunks...")
        if document_filter:
            print(f"[GroundTruthGenerator] Filtering by documents: {document_filter}")
        chunks = self.sample_chunks(num_samples, document_filter=document_filter)

        if len(chunks) < num_samples:
            print(f"[GroundTruthGenerator] Warning: Only found {len(chunks)} chunks, expected {num_samples}")

        ground_truth_data = []
        failed_count = 0

        for idx, chunk in enumerate(chunks, 1):
            print(f"[GroundTruthGenerator] Generating Q&A {idx}/{len(chunks)}...")

            context = chunk['text']
            metadata = chunk.get('metadata', {})

            # Retry logic for failed generations
            max_retries = 2
            for retry in range(max_retries):
                try:
                    # Generate question
                    question = self.generate_question_from_context(context)

                    # Validate question
                    if not question or len(question.strip()) < 10:
                        raise ValueError("Generated question is too short or empty")

                    # Generate ground truth answer
                    ground_truth_answer = self.generate_ground_truth_answer(question, context)

                    # Validate answer
                    if not ground_truth_answer or len(ground_truth_answer.strip()) < 10:
                        raise ValueError("Generated answer is too short or empty")

                    ground_truth_data.append({
                        'question': question,
                        'ground_truth_answer': ground_truth_answer,
                        'context': context,
                        'source_file': metadata.get('source_file', 'Unknown'),
                        'page_numbers': str(metadata.get('page_numbers', [])),
                        'chunk_id': chunk.get('id', ''),
                        'distance': chunk.get('distance', 0.0)
                    })

                    # Success - break retry loop
                    break

                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"[GroundTruthGenerator] Error generating Q&A {idx} (attempt {retry + 1}/{max_retries}): {e}. Retrying...")
                    else:
                        print(f"[GroundTruthGenerator] Failed to generate Q&A {idx} after {max_retries} attempts: {e}")
                        failed_count += 1

        if failed_count > 0:
            print(f"[GroundTruthGenerator] WARNING: Failed to generate {failed_count} Q&A pairs")

        print(f"[GroundTruthGenerator] Successfully generated {len(ground_truth_data)} Q&A pairs")

        # Create DataFrame
        df = pd.DataFrame(ground_truth_data)

        # Save to Excel if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_excel(output_path, index=False)
            print(f"[GroundTruthGenerator] Saved ground truth to: {output_path}")

        return df
