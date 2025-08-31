#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze the test results of the RAG system.
"""

import os
import pandas as pd

# Path to the most recent test results
results_dir = os.path.join(os.path.dirname(__file__), "test_results")
result_files = [f for f in os.listdir(results_dir) if f.endswith('.xlsx')]
result_files.sort(reverse=True)  # Most recent first
latest_result = os.path.join(results_dir, result_files[0])

print(f"Analyzing results from: {latest_result}")

# Load the results
df = pd.read_excel(latest_result)
total_cases = len(df)
print(f"Total test cases: {total_cases}")

# Analyze diagnoses
correct = 0
partial = 0
incorrect = 0

print("\nResults analysis:")
print("=" * 80)
print(f"{'Case':^5}|{'Actual Diagnosis':^25}|{'System Diagnosis':^50}")
print("-" * 80)

for idx, row in df.iterrows():
    case_id = row['CaseID']
    actual = row['ActualDiagnosis']
    system = str(row['SystemDiagnosis'])
    
    # Check if the actual diagnosis is contained in the system diagnosis
    if actual in system:
        status = "Correct"
        correct += 1
    elif any(term in system for term in actual.split()):
        status = "Partial"
        partial += 1
    else:
        status = "Incorrect"
        incorrect += 1
    
    # Truncate system diagnosis if too long
    system_short = system[:47] + "..." if len(system) > 50 else system
    print(f"{case_id:^5}|{actual:^25}|{system_short:<50} [{status}]")

# Print summary
print("\nSummary:")
print(f"Correct diagnoses: {correct}/{total_cases} ({correct/total_cases*100:.2f}%)")
print(f"Partially correct: {partial}/{total_cases} ({partial/total_cases*100:.2f}%)")
print(f"Incorrect diagnoses: {incorrect}/{total_cases} ({incorrect/total_cases*100:.2f}%)")
print(f"Overall accuracy (including partial): {(correct+partial)/total_cases*100:.2f}%")

if __name__ == "__main__":
    pass 