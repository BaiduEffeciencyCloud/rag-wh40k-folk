import json
import os
from typing import List, Dict, Any, Set
from dataclasses import dataclass, asdict

from vector_search import VectorSearch
from config import Settings

@dataclass
class QueryTest:
    query: str
    relevant_chunk_ids: Set[str]
    expected_answer: str
    top_k: int = 10

@dataclass
class EvalResult:
    query: str
    precision: float
    recall: float
    retrieved_ids: List[str]
    expected_ids: List[str]
    answer: str
    expected_answer: str
    answer_match: bool


def load_tests(path: str) -> List[QueryTest]:
    """
    Load test cases from a JSON file. Each entry should have:
    - query: str
    - relevant_chunk_ids: list of str
    - expected_answer: str
    - top_k: optional int
    """
    data = json.load(open(path, 'r', encoding='utf-8'))
    tests: List[QueryTest] = []
    for item in data:
        tests.append(
            QueryTest(
                query=item['query'],
                relevant_chunk_ids=set(item.get('relevant_chunk_ids', [])),
                expected_answer=item.get('expected_answer', '').strip(),
                top_k=item.get('top_k', 10)
            )
        )
    return tests


def evaluate(vs: VectorSearch, tests: List[QueryTest]) -> List[EvalResult]:
    results: List[EvalResult] = []

    for test in tests:
        # Perform retrieval + answer generation
        resp = vs.search(test.query, top_k=test.top_k)
        # Assumes resp is a dict containing 'answer' and 'chunks'
        answer: str = resp.get('answer', '').strip()
        chunks: List[Dict[str, Any]] = resp.get('chunks', [])

        # Extract retrieved ids
        retrieved_ids = [c['metadata'].get('chunk_id') for c in chunks]
        retrieved_set = set(filter(None, retrieved_ids))

        # Compute metrics
        true_ids = test.relevant_chunk_ids
        tp = len(retrieved_set & true_ids)
        precision = tp / len(retrieved_set) if retrieved_set else 0.0
        recall = tp / len(true_ids) if true_ids else 0.0

        # Simple answer match (exact)
        answer_match = answer.lower() == test.expected_answer.lower()

        results.append(
            EvalResult(
                query=test.query,
                precision=precision,
                recall=recall,
                retrieved_ids=retrieved_ids,
                expected_ids=list(true_ids),
                answer=answer,
                expected_answer=test.expected_answer,
                answer_match=answer_match
            )
        )
    return results


def main():
    # Load configuration
    settings = Settings()
    vs = VectorSearch(
        pinecone_api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index,
        openai_api_key=settings.openai_api_key
    )

    # Path to test cases
    tests_path = os.getenv('EVAL_TESTS_PATH', 'tests.json')
    tests = load_tests(tests_path)

    # Run evaluation
    results = evaluate(vs, tests)

    # Save results
    out_path = os.getenv('EVAL_RESULTS_PATH', 'eval_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    # Print summary
    avg_precision = sum(r.precision for r in results) / len(results)
    avg_recall = sum(r.recall for r in results) / len(results)
    exact_match = sum(r.answer_match for r in results) / len(results)
    print(f"Evaluation completed: {len(results)} queries")
    print(f"Avg Precision: {avg_precision:.2f}, Avg Recall: {avg_recall:.2f}, Exact Match: {exact_match:.2f}")


if __name__ == '__main__':
    main()
