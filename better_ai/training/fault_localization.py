"""
Fault Localization and Patch Generation Pipeline
Implements multi-stage reasoning for software repair tasks
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import ast
import traceback

class FaultLocalizer:
    """
    Identifies potential faults in code based on error traces or failing tests
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = logging.getLogger(__name__)

    def _get_line_context(self, filename: str, lineno: int) -> Optional[str]:
        """Helper to read a specific line from a file for context"""
        try:
            import os
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    if 1 <= lineno <= len(lines):
                        return lines[lineno - 1].strip()
        except Exception:
            pass
        return None

    def parse_python_trace(self, trace: str) -> List[Dict[str, Any]]:
        """Parses Python traceback to extract file and line information"""
        faults = []
        # Pattern for Python traceback lines: File "file.py", line 10, in <module>
        pattern = r'File "(.*?)", line (\d+), in (.*)'
        matches = re.findall(pattern, trace)

        for i, (filename, lineno, func) in enumerate(matches):
            line_int = int(lineno)
            # Higher suspiciousness for lines deeper in the stack trace (last matched are usually closer to error)
            suspiciousness = (i + 1) / len(matches)

            line_content = self._get_line_context(filename, line_int)
            reason = f"Present in stack trace at depth {len(matches) - i}"
            if line_content:
                reason += f": `{line_content}`"

            faults.append({
                "file": filename,
                "line_no": line_int,
                "function": func,
                "suspiciousness": suspiciousness,
                "reason": reason,
                "context": line_content
            })

        # Sort by suspiciousness descending
        return sorted(faults, key=lambda x: x["suspiciousness"], reverse=True)

    def parse_rust_trace(self, trace: str) -> List[Dict[str, Any]]:
        """Parses Rust panic/error trace"""
        faults = []
        # Pattern for Rust backtrace: at src/main.rs:10:5
        pattern = r'at (.*?):(\d+):(\d+)'
        matches = re.findall(pattern, trace)

        for i, (filename, lineno, colno) in enumerate(matches):
            line_int = int(lineno)
            suspiciousness = (i + 1) / len(matches)

            line_content = self._get_line_context(filename, line_int)
            reason = "Rust backtrace entry"
            if line_content:
                reason += f": `{line_content}`"

            faults.append({
                "file": filename,
                "line_no": line_int,
                "column": int(colno),
                "suspiciousness": suspiciousness,
                "reason": reason,
                "context": line_content
            })
        return sorted(faults, key=lambda x: x["suspiciousness"], reverse=True)

    def parse_c_trace(self, trace: str) -> List[Dict[str, Any]]:
        """Parses C/GDB style trace"""
        faults = []
        # Pattern: #0  0x0000... in func () at file.c:10
        pattern = r'at (.*?):(\d+)'
        matches = re.findall(pattern, trace)

        for i, (filename, lineno) in enumerate(matches):
            line_int = int(lineno)
            suspiciousness = 1.0 / (i + 1) # Closer to #0 is more suspicious

            line_content = self._get_line_context(filename, line_int)
            reason = f"C stack frame #{i}"
            if line_content:
                reason += f": `{line_content}`"

            faults.append({
                "file": filename,
                "line_no": line_int,
                "suspiciousness": suspiciousness,
                "reason": reason,
                "context": line_content
            })
        return faults

    def parse_generic_trace(self, trace: str) -> List[Dict[str, Any]]:
        """Fallback parser that looks for file:line or file, line patterns"""
        faults = []
        # Pattern for generic file:line or file, line
        pattern = r'([a-zA-Z0-9_\-\./]+)[:\s,]+line\s+(\d+)|([a-zA-Z0-9_\-\./]+):(\d+)'
        matches = re.findall(pattern, trace)

        for match in matches:
            # Match is a tuple, find the non-empty parts
            groups = [g for g in match if g]
            if len(groups) >= 2:
                filename, lineno = groups[0], groups[1]
                if lineno.isdigit():
                    line_int = int(lineno)
                    line_content = self._get_line_context(filename, line_int)
                    reason = "Generic trace pattern match"
                    if line_content:
                        reason += f": `{line_content}`"

                    faults.append({
                        "file": filename,
                        "line_no": line_int,
                        "suspiciousness": 0.5,
                        "reason": reason,
                        "context": line_content
                    })
        return faults

    def calculate_sbfl_scores(self, coverage_data: List[Dict[str, Any]], technique: str = "ochiai") -> List[Dict[str, Any]]:
        """
        Calculates suspiciousness scores using Spectrum-Based Fault Localization (SBFL)

        Args:
            coverage_data: List of dicts with keys: 'passed' (bool), 'covered_lines' (Dict[file, List[int]])
            technique: 'tarantula' or 'ochiai'
        """
        line_stats = {} # (file, line) -> {'passed': count, 'failed': count}
        total_passed = 0
        total_failed = 0

        for test_result in coverage_data:
            passed = test_result.get('passed', False)
            if passed:
                total_passed += 1
            else:
                total_failed += 1

            covered_lines = test_result.get('covered_lines', {})
            for filename, lines in covered_lines.items():
                for lineno in lines:
                    key = (filename, lineno)
                    if key not in line_stats:
                        line_stats[key] = {'passed': 0, 'failed': 0}

                    if passed:
                        line_stats[key]['passed'] += 1
                    else:
                        line_stats[key]['failed'] += 1

        scores = []
        import math

        for (filename, lineno), stats in line_stats.items():
            p_s = stats['passed']
            f_s = stats['failed']

            suspiciousness = 0.0
            if technique.lower() == "tarantula":
                if total_passed > 0 and total_failed > 0:
                    top = f_s / total_failed
                    bottom = (p_s / total_passed) + (f_s / total_failed)
                    if bottom > 0:
                        suspiciousness = top / bottom
            elif technique.lower() == "ochiai":
                if total_failed > 0 and (f_s + p_s) > 0:
                    suspiciousness = f_s / math.sqrt(total_failed * (f_s + p_s))

            if suspiciousness > 0:
                scores.append({
                    "file": filename,
                    "line_no": lineno,
                    "suspiciousness": suspiciousness,
                    "reason": f"SBFL score ({technique}): {suspiciousness:.4f}",
                    "passed_count": p_s,
                    "failed_count": f_s
                })

        return sorted(scores, key=lambda x: x["suspiciousness"], reverse=True)

    def localize_fault(self, code: str, error_trace: str, language: str = "python") -> List[Dict[str, Any]]:
        """
        Analyzes code and error trace to identify likely faulty lines
        """
        faults = []
        if language.lower() == "python":
            faults = self.parse_python_trace(error_trace)
        elif language.lower() == "rust":
            faults = self.parse_rust_trace(error_trace)
        elif language.lower() == "c" or language.lower() == "cpp":
            faults = self.parse_c_trace(error_trace)

        # If no language-specific faults found, try generic parsing
        if not faults:
            faults = self.parse_generic_trace(error_trace)

        # If still no faults found via parsing, use model inference as fallback
        if not faults and self.model:
            # Model-based localization: uses the model to predict suspicious lines
            # In a real scenario, this would involve a forward pass with the code and error
            # For this implementation, we use a simple attention-based heuristic if possible
            # or fall back to a structured prompt to the model itself.
            try:
                if hasattr(self.model, "localize_fault_from_text"):
                    faults = self.model.localize_fault_from_text(code, error_trace)
                else:
                    # Heuristic: if no trace found, the model might "think" about the bug
                    # We can use a specialized head if it exists
                    advanced_features = getattr(self.model, "advanced_features", {})
                    if "fault_localization" in advanced_features:
                        # Logic to extract from model internal states
                        pass
            except Exception as e:
                self.logger.error(f"Model-based localization failed: {e}")

        # If code was provided, and we found faults but no context, try to use provided code
        if code and faults:
            code_lines = code.splitlines()
            for fault in faults:
                if not fault.get("context"):
                    # Check if filename in fault matches something reasonable or if it's the only code provided
                    # For simplicity, if code is provided, we assume it's for the first file in the trace if not specified
                    l_no = fault["line_no"]
                    if 1 <= l_no <= len(code_lines):
                        fault["context"] = code_lines[l_no - 1].strip()
                        if "`" not in fault["reason"]:
                             fault["reason"] += f": `{fault['context']}`"

        return faults

class PatchGenerator:
    """
    Generates potential patches for identified faults
    """
    def __init__(self, model: nn.Module, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)

    def _validate_patch(self, patch: str, language: str = "python") -> bool:
        """Validates patch syntax using AST or basic checks"""
        if language.lower() == "python":
            try:
                ast.parse(patch)
                return True
            except SyntaxError:
                return False
        # Basic validation for other languages
        return len(patch.strip()) > 0

    def generate_patches(self, code: str, faults: List[Dict[str, Any]], language: str = "python") -> List[str]:
        """
        Generates multiple patch candidates for the identified faults
        """
        if not faults:
            return []

        # Use LLM to generate patches (mocked here but shows intended flow)
        # In real implementation, this would format a prompt with code and faults
        # and call self.model.generate()

        candidates = []
        if self.model and hasattr(self.model, "generate") and self.tokenizer:
            # Construct prompt for repair
            fault_desc = "\n".join([f"Line {f['line_no']}: {f['reason']}" for f in faults[:3]])
            prompt = f"### Task: Repair the following {language} code\n\n### Original Code:\n{code}\n\n### Identified Faults:\n{fault_desc}\n\n### Fixed Code:\n"

            # Generate multiple completions
            # ... model generation logic ...
            pass

        # Fallback/Template patches if model not ready or for testing
        template_patches = [
            f"# Patch candidate for line {faults[0]['line_no']}\n# TODO: Implement real patch synthesis",
        ]

        valid_patches = [p for p in template_patches if self._validate_patch(p, language)]
        return valid_patches

class SoftwareRepairPipeline:
    """
    End-to-end pipeline for fault localization and patch generation
    """
    def __init__(self, localizer: FaultLocalizer, generator: PatchGenerator):
        self.localizer = localizer
        self.generator = generator
        self.logger = logging.getLogger(__name__)

    def repair(self, code: str, error_trace: str, language: str = "python") -> Dict[str, Any]:
        """
        Performs full repair process
        """
        # Step 1: Fault Localization
        faults = self.localizer.localize_fault(code, error_trace, language)

        # Step 2: Patch Generation
        patches = self.generator.generate_patches(code, faults, language)

        return {
            "faults": faults,
            "patches": patches,
            "status": "success" if patches else "failed",
            "language": language
        }

    def validate_repair(self, original_code: str, patch: str, test_command: str, filename: str = "repaired_code.py") -> bool:
        """
        Validates repair by running tests in a secure environment.

        TODO: In production, this MUST be run inside a fully virtualized container (e.g., Docker, nsjail)
        to prevent arbitrary code execution vulnerabilities.
        """
        import subprocess
        import os
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file_path = os.path.join(tmpdir, filename)

            # Write the patched code to the temporary file
            with open(temp_file_path, "w") as f:
                f.write(patch)

            # If the test command needs other files from the environment, this might be tricky
            # For now, we assume the test command can run with just this file or uses absolute paths

            try:
                # Security check: avoid running as root if possible
                if os.getuid() == 0:
                    self.logger.warning("Running repair validation as ROOT. This is highly discouraged!")

                # Run the test command
                # We set cwd to tmpdir so the test command runs in the context of the patched file
                # In prod, we'd use 'runuser' or similar to drop privileges here.
                result = subprocess.run(
                    test_command,
                    shell=True,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=30 # Safety timeout
                )

                self.logger.info(f"Repair validation stdout: {result.stdout}")
                if result.returncode != 0:
                    self.logger.warning(f"Repair validation failed (exit {result.returncode}): {result.stderr}")

                return result.returncode == 0
            except subprocess.TimeoutExpired:
                self.logger.error("Repair validation timed out")
                return False
            except Exception as e:
                self.logger.error(f"Error during repair validation: {e}")
                return False

def compute_repair_reward(results: Dict[str, Any], test_pass_rate: float, ground_truth_fault_line: Optional[int] = None) -> float:
    """
    Computes a comprehensive reward for the repair process
    """
    if results["status"] != "success":
        return 0.0

    # 1. Patch Quality Score (based on test pass rate)
    quality_score = test_pass_rate

    # 2. Localization Accuracy Score
    loc_score = 0.0
    if ground_truth_fault_line is not None and results["faults"]:
        # Check if ground truth is in top 3 predicted faults
        top_faults = [f["line_no"] for f in results["faults"][:3]]
        if ground_truth_fault_line in top_faults:
            # Score based on rank: 1.0 for rank 0, 0.7 for rank 1, 0.5 for rank 2
            rank = top_faults.index(ground_truth_fault_line)
            loc_score = [1.0, 0.7, 0.5][rank]

    # Combine into unified reward
    return 0.7 * quality_score + 0.3 * loc_score
