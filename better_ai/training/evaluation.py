"""
Comprehensive evaluation suite for RLHF and coding benchmarks
Metrics for reasoning quality, multi-attribute performance, SWE-bench, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import subprocess
import tempfile
import os


@dataclass
class EvaluationMetrics:
    """Dataclass for storing evaluation metrics"""
    coding_accuracy: float = 0.0
    reasoning_quality: float = 0.0
    efficiency_score: float = 0.0
    readability_score: float = 0.0
    robustness_score: float = 0.0
    correctness_score: float = 0.0
    multi_attr_scores: Dict[str, float] = None
    swe_bench_pass_rate: float = 0.0
    json_compliance: float = 0.0
    grammar_compliance: float = 0.0
    tool_use_accuracy: float = 0.0
    entropy_metrics: Dict[str, float] = None


class RLHFEvaluator:
    """Evaluator for RLHF metrics"""
    
    def __init__(self, model, reward_model, device: torch.device):
        self.model = model
        self.reward_model = reward_model
        self.device = device
    
    def compute_reward_correlation(
        self,
        predictions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> float:
        """Compute correlation between model predictions and rewards"""
        correlation = torch.corrcoef(
            torch.stack([predictions, rewards])
        )[0, 1]
        return correlation.item()
    
    def compute_preference_accuracy(
        self,
        chosen_scores: torch.Tensor,
        rejected_scores: torch.Tensor,
    ) -> float:
        """Compute accuracy of preference prediction (chosen > rejected)"""
        correct = (chosen_scores > rejected_scores).float().mean()
        return correct.item()
    
    def compute_alignment_score(
        self,
        model_outputs: torch.Tensor,
        human_preferences: torch.Tensor,
    ) -> float:
        """Compute alignment with human preferences"""
        # Compute KL divergence between model and preference distributions
        model_probs = F.softmax(model_outputs, dim=-1)
        preference_probs = F.softmax(human_preferences, dim=-1)
        
        kl_div = F.kl_div(
            torch.log(model_probs + 1e-10),
            preference_probs,
            reduction='batchmean'
        )
        
        # Convert KL divergence to alignment score (lower KL = higher alignment)
        alignment = torch.exp(-kl_div)
        return alignment.item()
    
    def evaluate_reasoning_quality(
        self,
        reasoning_traces: torch.Tensor,
        correctness_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate quality of reasoning traces
        
        Returns:
            Dictionary with reasoning metrics
        """
        batch_size, num_steps, hidden_dim = reasoning_traces.shape
        
        # Trace completeness: measure diversity across steps
        trace_diversity = []
        for i in range(batch_size):
            trace = reasoning_traces[i]
            # Compute pairwise distances between steps
            distances = torch.pdist(trace)
            diversity = distances.mean().item()
            trace_diversity.append(diversity)
        
        # Trace coherence: measure semantic consistency
        coherence_scores = []
        for i in range(batch_size):
            trace = reasoning_traces[i]
            # Compute cosine similarity between consecutive steps
            similarities = []
            for j in range(num_steps - 1):
                sim = F.cosine_similarity(
                    trace[j].unsqueeze(0),
                    trace[j+1].unsqueeze(0)
                )
                similarities.append(sim.item())
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            coherence_scores.append(avg_similarity)
        
        # Correctness correlation with reasoning
        correctness_correlation = 0.0
        if correctness_labels is not None:
            trace_means = reasoning_traces.mean(dim=1)
            # Correlate reasoning quality with correctness
            correctness_correlation = torch.corrcoef(
                torch.stack([trace_means.norm(dim=-1), correctness_labels.float()])
            )[0, 1].item()
        
        return {
            "trace_diversity": float(np.mean(trace_diversity)),
            "trace_coherence": float(np.mean(coherence_scores)),
            "correctness_correlation": correctness_correlation,
        }
    
    def evaluate_multi_attribute(
        self,
        model_outputs: torch.Tensor,
        attribute_labels: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Evaluate multi-attribute regression performance
        
        Args:
            model_outputs: Model's attribute predictions
            attribute_labels: Ground truth labels for each attribute
        
        Returns:
            Scores for each attribute
        """
        scores = {}
        
        for attr_name, labels in attribute_labels.items():
            if attr_name in model_outputs:
                predictions = model_outputs[attr_name]
                
                # Compute MSE
                mse = F.mse_loss(predictions, labels)
                
                # Compute correlation
                correlation = torch.corrcoef(
                    torch.stack([predictions.view(-1), labels.view(-1)])
                )[0, 1]
                
                # Compute ranking accuracy (for preference learning)
                ranking_acc = self._compute_ranking_accuracy(predictions, labels)
                
                scores[attr_name] = {
                    "mse": mse.item(),
                    "correlation": correlation.item(),
                    "ranking_accuracy": ranking_acc,
                }
        
        return scores
    
    def _compute_ranking_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute ranking accuracy (pairwise correctness)"""
        batch_size = predictions.shape[0]
        
        if batch_size < 2:
            return 0.0
        
        correct = 0
        total = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                pred_correct = predictions[i] > predictions[j]
                label_correct = labels[i] > labels[j]
                
                if pred_correct == label_correct:
                    correct += 1
                total += 1
        
        return correct / max(total, 1)


class CodingBenchmarkEvaluator:
    """Evaluator for coding-specific benchmarks"""
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
    
    def execute_and_test(
        self,
        code: str,
        test_cases: List[Tuple[List, Any]],
    ) -> Dict[str, Any]:
        """
        Execute code and test against test cases
        
        Args:
            code: Python code to execute
            test_cases: List of (inputs, expected_output) tuples
        
        Returns:
            Dictionary with execution results
        """
        results = {
            "passed": 0,
            "total": len(test_cases),
            "errors": [],
            "correct_output": True,
        }
        
        # Create temporary file for code execution
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            for inputs, expected_output in test_cases:
                try:
                    # Execute code with timeout
                    result = subprocess.run(
                        ['python', temp_file],
                        capture_output=True,
                        timeout=self.timeout,
                        input=' '.join(map(str, inputs)) if inputs else '',
                    )
                    
                    if result.returncode == 0:
                        output = result.stdout.decode().strip()
                        expected_str = str(expected_output).strip()
                        
                        if output == expected_str:
                            results["passed"] += 1
                        else:
                            results["errors"].append(f"Output mismatch: {output} != {expected_str}")
                            results["correct_output"] = False
                    else:
                        results["errors"].append(f"Execution error: {result.stderr.decode()}")
                        results["correct_output"] = False
                        
                except subprocess.TimeoutExpired:
                    results["errors"].append("Timeout")
                    results["correct_output"] = False
                except Exception as e:
                    results["errors"].append(str(e))
                    results["correct_output"] = False
        finally:
            os.unlink(temp_file)
        
        results["pass_rate"] = results["passed"] / results["total"]
        return results
    
    def evaluate_code_quality(self, code: str) -> Dict[str, float]:
        """
        Evaluate code quality metrics
        
        Returns:
            Dictionary with readability, efficiency estimates
        """
        lines = code.split('\n')
        
        metrics = {
            "lines_of_code": len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
            "comment_ratio": len([l for l in lines if l.strip().startswith('#')]) / max(len(lines), 1),
            "avg_line_length": np.mean([len(l) for l in lines]) if lines else 0,
            "complexity_estimate": self._estimate_cyclomatic_complexity(code),
        }
        
        return metrics
    
    def _estimate_cyclomatic_complexity(self, code: str) -> float:
        """Estimate cyclomatic complexity of code"""
        complexity = 1  # Base complexity
        
        # Count control flow keywords
        complexity += code.count('if ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('except ')
        complexity += code.count('elif ')
        
        return float(complexity)


class SWEBenchEvaluator:
    """Evaluator for SWE-bench software engineering tasks"""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
    
    def evaluate_patch(
        self,
        repo_path: str,
        patch: str,
        test_script: str,
    ) -> Dict[str, Any]:
        """
        Evaluate patch on actual repository tests
        
        Args:
            repo_path: Path to repository
            patch: Unified diff patch to apply
            test_script: Test command to run
        
        Returns:
            Results of patch evaluation
        """
        results = {
            "patch_applied": False,
            "tests_passed": False,
            "num_tests": 0,
            "num_passed": 0,
            "errors": [],
        }
        
        # Apply patch
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch)
                patch_file = f.name
            
            # Apply patch using git/patch utility
            apply_result = subprocess.run(
                ['patch', '-p0'],
                stdin=open(patch_file, 'r'),
                cwd=repo_path,
                capture_output=True,
                timeout=self.timeout,
            )
            
            if apply_result.returncode == 0:
                results["patch_applied"] = True
            else:
                results["errors"].append(f"Patch application failed: {apply_result.stderr.decode()}")
                return results
            
            # Run tests
            test_result = subprocess.run(
                test_script,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                timeout=self.timeout,
            )
            
            if test_result.returncode == 0:
                results["tests_passed"] = True
            
            # Parse test output for more details
            output = test_result.stdout.decode()
            results["test_output"] = output
            
        except Exception as e:
            results["errors"].append(str(e))
        finally:
            try:
                os.unlink(patch_file)
            except:
                pass
        
        return results


class MetricsAggregator:
    """Aggregate multiple evaluation metrics"""
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_overall_score(self, metrics: EvaluationMetrics) -> float:
        """
        Compute overall model score from individual metrics
        
        Weighted combination of multiple metrics
        """
        weights = {
            "coding_accuracy": 0.25,
            "reasoning_quality": 0.20,
            "correctness_score": 0.20,
            "multi_attr": 0.15,
            "efficiency_score": 0.10,
            "json_compliance": 0.10,
        }
        
        score = (
            weights["coding_accuracy"] * metrics.coding_accuracy +
            weights["reasoning_quality"] * metrics.reasoning_quality +
            weights["correctness_score"] * metrics.correctness_score +
            weights["efficiency_score"] * metrics.efficiency_score +
            weights["json_compliance"] * metrics.json_compliance
        )
        
        # Add multi-attribute bonus
        if metrics.multi_attr_scores:
            multi_score = np.mean(list(metrics.multi_attr_scores.values()))
            score += weights["multi_attr"] * multi_score
        
        return min(1.0, max(0.0, score))  # Clip to [0, 1]
    
    def log_metrics(self, metrics: EvaluationMetrics, step: int):
        """Log metrics for tracking over time"""
        self.metrics_history.append({
            "step": step,
            "metrics": metrics,
        })
    
    def generate_report(self, output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        report = {
            "timestamp": str(np.datetime64('now')),
            "num_evaluations": len(self.metrics_history),
            "metrics_history": []
        }
        
        for entry in self.metrics_history:
            metrics_dict = {
                "step": entry["step"],
                "coding_accuracy": entry["metrics"].coding_accuracy,
                "reasoning_quality": entry["metrics"].reasoning_quality,
                "efficiency_score": entry["metrics"].efficiency_score,
                "readability_score": entry["metrics"].readability_score,
                "robustness_score": entry["metrics"].robustness_score,
                "json_compliance": entry["metrics"].json_compliance,
                "grammar_compliance": entry["metrics"].grammar_compliance,
                "swe_bench_pass_rate": entry["metrics"].swe_bench_pass_rate,
            }
            report["metrics_history"].append(metrics_dict)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
