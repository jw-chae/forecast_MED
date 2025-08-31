
import fitz  # PyMuPDF
import os
import json

# Define paths
BASE_DIR = '/home/joongwon00/Project_Tsinghua_Paper/med_deepseek'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
PDF_FILE = os.path.join(DATA_DIR, 'Epidemic_guide.pdf')
OUTPUT_FILE = os.path.join(PROCESSED_DIR, 'Epidemic_guide.json')

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages_content = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages_content.append({
            "page_number": page_num + 1,
            "content": text
        })
    doc.close()
    return pages_content

def main():
    """
    Main function to extract text from the PDF and save it as JSON.
    """
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    try:
        if os.path.exists(PDF_FILE):
            print(f"Processing PDF: {PDF_FILE}")
            pdf_data = extract_text_from_pdf(PDF_FILE)
            
            # Save the extracted text to a JSON file
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(pdf_data, f, ensure_ascii=False, indent=4)
            
            print(f"Successfully extracted text from PDF and saved to {OUTPUT_FILE}")
        else:
            print(f"PDF file not found: {PDF_FILE}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
