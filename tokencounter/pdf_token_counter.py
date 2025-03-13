#!/usr/bin/env python3
"""
PDF Token Counter

This script extracts text from a PDF file and counts the number of tokens.
It uses PyPDF2 for PDF text extraction and nltk for tokenization.
"""

import os
import sys
import argparse
from pathlib import Path
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('all')

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            print(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
                
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def count_tokens(text):
    """
    Count tokens in the given text.
    
    Args:
        text (str): Text to tokenize and count
        
    Returns:
        tuple: (total_token_count, unique_token_count, token_frequency)
    """
    if not text:
        return 0, 0, {}
    
    # Use a simpler tokenization approach that doesn't rely on punkt_tab
    # Split by whitespace and then clean up the tokens
    tokens = []
    for word in text.split():
        # Remove punctuation from the beginning and end of words
        word = word.strip('.,;:!?()[]{}"\'-')
        if word:  # Only add non-empty tokens
            tokens.append(word)
    
    # Count token frequencies
    token_freq = Counter(tokens)
    
    return len(tokens), len(token_freq), token_freq

def main():
    parser = argparse.ArgumentParser(description='Count tokens in a PDF file')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--top', type=int, default=10, help='Show top N most frequent tokens')
    
    args = parser.parse_args()
    
    # Check if file exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File '{pdf_path}' does not exist")
        sys.exit(1)
    
    print(f"Processing PDF: {pdf_path}")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("No text was extracted from the PDF")
        sys.exit(1)
    
    # Count tokens
    total_count, unique_count, token_freq = count_tokens(text)
    
    # Print results
    print(f"\nResults:")
    print(f"Total tokens: {total_count}")
    print(f"Unique tokens: {unique_count}")
    
    # Print most common tokens
    print(f"\nTop {args.top} most frequent tokens:")
    for token, count in token_freq.most_common(args.top):
        print(f"  {token}: {count}")

if __name__ == "__main__":
    main() 