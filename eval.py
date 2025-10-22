from argparse import ArgumentParser
from dotenv import load_dotenv
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
from datetime import datetime
import pandas as pd


from parser import SimpleAgentParser, OneShotParser
from main import DocumentModel, validate_citations


load_dotenv()


def validate_citations(document: str, data: DocumentModel) -> Tuple[bool, str]:
    """Validate the citations in the document."""
    clean_document = document.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("  ", " ")
    if data.title.passage not in clean_document:
        return False, f"`Title` passage not found in document, passage={data.title.passage}"
    if data.author.passage not in clean_document:
        return False, f"`Author` passage not found in document, passage={data.author.passage}"
    if data.date.passage not in clean_document:
        return False, f"`Date` passage not found in document, passage={data.date.passage}"
    return True, "Citations validated successfully"


class ParserConfig:
    """Configuration for a parser variant"""
    def __init__(self, name: str, parser_class, model_name: str = None, validation_functions: List[Callable] = None):
        self.name = name
        self.parser_class = parser_class
        self.model_name = model_name
        self.validation_functions = validation_functions or []


async def load_raw_documents(raw_dir: Path) -> List[tuple[str, str]]:
    """Load all documents from the raw data directory"""
    documents = []
    for file_path in raw_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append((file_path.stem, content))
    return documents


def load_ground_truth(parsed_dir: Path, filename: str) -> Dict[str, Any] | None:
    """Load ground truth from parsed directory"""
    # Try different extensions
    for ext in [".json", ".txt"]:
        gt_path = parsed_dir / f"{filename}{ext}"
        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                if ext == ".json":
                    return json.load(f)
                else:
                    # If it's a text file, return as raw text
                    return {"raw_content": f.read()}
    return None


async def parse_document(parser, document: str, parser_name: str) -> Dict[str, Any]:
    """Parse a document using a given parser"""
    try:
        result = await parser.parse(document)
        return {
            "success": True,
            "data": result.model_dump(mode='json'),
            "parser": parser_name,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "parser": parser_name,
            "error": str(e)
        }


def save_result(result: Dict[str, Any], results_dir: Path, parser_name: str, filename: str):
    """Save parsed result to disk"""
    parser_dir = results_dir / parser_name
    parser_dir.mkdir(parents=True, exist_ok=True)

    output_path = parser_dir / f"{filename}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def evaluate_results(parsed_results: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate parsed results against ground truth using Evidently"""

    # Create a DataFrame for evaluation
    eval_data = []
    for parsed, gt in zip(parsed_results, ground_truths):
        if parsed["success"] and gt:
            eval_data.append({
                "parser": parsed["parser"],
                "success": parsed["success"],
                "has_ground_truth": gt is not None,
                "parsed_fields": len(parsed["data"]) if parsed["data"] else 0,
                "gt_fields": len(gt) if isinstance(gt, dict) else 0
            })
        else:
            eval_data.append({
                "parser": parsed["parser"],
                "success": parsed["success"],
                "has_ground_truth": gt is not None,
                "parsed_fields": 0,
                "gt_fields": len(gt) if gt and isinstance(gt, dict) else 0
            })

    df = pd.DataFrame(eval_data)

    # Calculate basic metrics
    metrics = {
        "total_documents": len(parsed_results),
        "successful_parses": sum(1 for r in parsed_results if r["success"]),
        "failed_parses": sum(1 for r in parsed_results if not r["success"]),
        "success_rate": sum(1 for r in parsed_results if r["success"]) / len(parsed_results) if parsed_results else 0,
        "avg_fields_parsed": df["parsed_fields"].mean() if not df.empty else 0,
        "parsers": {}
    }

    # Group by parser
    for parser_name in df["parser"].unique():
        parser_df = df[df["parser"] == parser_name]
        metrics["parsers"][parser_name] = {
            "success_rate": parser_df["success"].mean(),
            "avg_fields_parsed": parser_df["parsed_fields"].mean(),
            "total_attempts": len(parser_df)
        }

    return metrics


def generate_report(all_results: Dict[str, List[Dict[str, Any]]],
                   metrics: Dict[str, Any],
                   results_dir: Path):
    """Generate an evaluation report"""

    report_path = results_dir / "evaluation_report.html"

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Parser Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .success {{ color: green; }}
            .error {{ color: red; }}
            .metric {{ background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Parser Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="metric">
            <h2>Overall Metrics</h2>
            <p>Total Documents: {metrics['total_documents']}</p>
            <p>Successful Parses: <span class="success">{metrics['successful_parses']}</span></p>
            <p>Failed Parses: <span class="error">{metrics['failed_parses']}</span></p>
            <p>Success Rate: {metrics['success_rate']:.2%}</p>
            <p>Average Fields Parsed: {metrics['avg_fields_parsed']:.2f}</p>
        </div>

        <h2>Parser Performance</h2>
        <table>
            <tr>
                <th>Parser</th>
                <th>Success Rate</th>
                <th>Avg Fields Parsed</th>
                <th>Total Attempts</th>
            </tr>
    """

    for parser_name, parser_metrics in metrics["parsers"].items():
        html_content += f"""
            <tr>
                <td>{parser_name}</td>
                <td>{parser_metrics['success_rate']:.2%}</td>
                <td>{parser_metrics['avg_fields_parsed']:.2f}</td>
                <td>{parser_metrics['total_attempts']}</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>Detailed Results by Document</h2>
    """

    for filename, results in all_results.items():
        html_content += f"<h3>Document: {filename}</h3><table><tr><th>Parser</th><th>Status</th><th>Error</th></tr>"
        for result in results:
            status_class = "success" if result["success"] else "error"
            status_text = "Success" if result["success"] else "Failed"
            error_text = result.get("error", "N/A") if not result["success"] else "-"
            html_content += f"""
                <tr>
                    <td>{result['parser']}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{error_text}</td>
                </tr>
            """
        html_content += "</table>"

    html_content += """
    </body>
    </html>
    """

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nEvaluation report generated: {report_path}")


async def main():
    args = ArgumentParser()
    args.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw documents")
    args.add_argument("--parsed-dir", type=str, default="data/parsed", help="Directory containing ground truth")
    args.add_argument("--results-dir", type=str, default="results", help="Directory to store results")
    args = args.parse_args()

    raw_dir = Path(args.raw_dir)
    parsed_dir = Path(args.parsed_dir)
    results_dir = Path(args.results_dir)

    # Ensure results directory exists
    results_dir.mkdir(exist_ok=True)

    # Setup API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    model_name="gemini/gemini-2.5-flash"

    # Define parser configurations to test
    parser_configs = [
        ParserConfig(
            name=f"SimpleAgentParser_{model_name}",
            parser_class=SimpleAgentParser,
            model_name=model_name,
            validation_functions=[validate_citations]
        ),
        ParserConfig(
            name=f"OneShotParser_{model_name}",
            parser_class=OneShotParser,
            model_name=model_name
        ),
    ]

    print(f"Loading documents from {raw_dir}...")
    documents = await load_raw_documents(raw_dir)
    print(f"Found {len(documents)} documents")

    all_results = {}
    all_parsed_results = []
    all_ground_truths = []

    # Process each document with all parsers
    for filename, content in documents:
        print(f"\nProcessing document: {filename}")
        document_results = []

        # Load ground truth
        ground_truth = load_ground_truth(parsed_dir, filename)
        if ground_truth:
            print(f"  Ground truth found for {filename}")
        else:
            print(f"  No ground truth found for {filename}")

        # Test each parser
        for config in parser_configs:
            print(f"  Testing parser: {config.name}")

            # Create parser with validation functions if applicable
            if config.validation_functions:
                parser = config.parser_class(DocumentModel, config.model_name, config.validation_functions)
            else:
                parser = config.parser_class(DocumentModel, config.model_name)

            # Parse document
            result = await parse_document(parser, content, config.name)
            document_results.append(result)
            all_parsed_results.append(result)
            all_ground_truths.append(ground_truth)

            # Save result
            save_result(result, results_dir, config.name, filename)

            if result["success"]:
                print(f"    ✓ Success")
            else:
                print(f"    ✗ Failed: {result['error']}")

        all_results[filename] = document_results

    # Evaluate results
    print("\nEvaluating results...")
    metrics = evaluate_results(all_parsed_results, all_ground_truths)

    # Save metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Generate report
    generate_report(all_results, metrics, results_dir)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Documents: {metrics['total_documents']}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"\nParser Performance:")
    for parser_name, parser_metrics in metrics["parsers"].items():
        print(f"  {parser_name}:")
        print(f"    Success Rate: {parser_metrics['success_rate']:.2%}")
        print(f"    Avg Fields: {parser_metrics['avg_fields_parsed']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
