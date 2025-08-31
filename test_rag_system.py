#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script tests the RAG system implemented in r1_smolagent_rag.py.
It uses patient data from the test_data.xlsx file to evaluate the system's diagnostic capabilities.
"""

import os
import sys
import time
import pandas as pd
import re
from tqdm import tqdm
import json
from datetime import datetime

# Import modules for calling the RAG system
from r1_smolagent_rag import (
    retriever, 
    llm,  
    prompt, 
    format_docs,
    format_chat_history,
    create_chain
)

# Function to get RAG response using patient symptoms and medical history
def get_rag_response(chief_complaint, medical_history, chat_history=None):
    """
    Queries the RAG system with combined chief complaint and medical history.
    
    Args:
        chief_complaint (str): The patient's chief complaint
        medical_history (str): The patient's medical history
        chat_history (list, optional): Previous conversation history
        
    Returns:
        str: The RAG system's response
    """
    if chat_history is None:
        chat_history = []
        
    # Combine chief complaint and medical history for the query
    query = f"{chief_complaint}. {medical_history}"
    print(f"\nQuery: {query[:150]}...")
    
    # Create and call the RAG chain
    chain = create_chain(chat_history)
    try:
        response = chain.invoke(query)
        return response
    except Exception as e:
        print(f"Error during RAG system call: {e}")
        return f"Error: {str(e)}"

def extract_diagnosis(response):
    """
    Extracts the diagnosed condition from the RAG system response.
    
    Args:
        response (str): The RAG system response
        
    Returns:
        str: The extracted diagnosis
    """
    # Use regex to extract diagnosis after '最终诊断' or '最可能诊断'
    pattern = r'最终诊断[：:]\s*(.+?)[\n\r]|最可能诊断[：:]\s*(.+?)[\n\r]'
    matches = re.search(pattern, response)
    
    if matches:
        # Return either the first or second group that has a value
        return matches.group(1) or matches.group(2)
    
    # If pattern not found, look for common disease names throughout the response
    disease_patterns = [
        r'新型冠状病毒感染', r'疟疾', r'流行性感冒', r'肺炎', 
        r'腹泻', r'肠胃炎', r'病毒性感染', r'细菌性感染'
    ]
    
    for pattern in disease_patterns:
        if re.search(pattern, response):
            return pattern
    
    return "No diagnosis found"

def main():
    # Load test data
    excel_path = os.path.join(os.path.dirname(__file__), "test_data", "test_data.xlsx")
    print(f"Loading test data file: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
        print(f"Data loading complete: {df.shape[0]} cases found")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # List to store results
    test_results = []
    
    # Show progress with tqdm
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Testing in progress"):
        case_id = idx + 1
        chief_complaint = row['主诉']
        medical_history = row['现病史']
        lab_results = row['检验检查结果']
        actual_diagnosis = row['疾病名称']
        
        print(f"\n[Test Case {case_id}/{df.shape[0]}]")
        print(f"Chief Complaint: {chief_complaint}")
        print(f"Medical History: {medical_history[:150]}...")
        print(f"Actual Diagnosis: {actual_diagnosis}")
        
        # Call the RAG system
        start_time = time.time()
        response = get_rag_response(chief_complaint, medical_history)
        end_time = time.time()
        
        # Extract diagnosis from response
        extracted_diagnosis = extract_diagnosis(response)
        
        # Store result
        result = {
            'CaseID': case_id,
            'ChiefComplaint': chief_complaint,
            'MedicalHistory': medical_history,
            'LabResults': lab_results,
            'ActualDiagnosis': actual_diagnosis,
            'SystemDiagnosis': extracted_diagnosis,
            'ResponseTime(s)': round(end_time - start_time, 2),
            'FullResponse': response
        }
        
        test_results.append(result)
        
        print(f"System Diagnosis: {extracted_diagnosis}")
        print(f"Response Time: {round(end_time - start_time, 2)} seconds")
        
        # Wait 2 seconds (considering API rate limits)
        time.sleep(2)
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"rag_test_results_{timestamp}.json")
    
    # Save results to JSON
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    # Save to Excel as well
    excel_results_path = os.path.join(results_dir, f"rag_test_results_{timestamp}.xlsx")
    pd.DataFrame(test_results).to_excel(excel_results_path, index=False)
    
    print(f"\nDetailed results saved to:")
    print(f"- JSON: {results_path}")
    print(f"- Excel: {excel_results_path}")

if __name__ == "__main__":
    main() 