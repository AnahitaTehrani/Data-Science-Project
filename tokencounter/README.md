# PDF Token Counter

A simple Python script to count the number of tokens (words) in a PDF document.

## Features

- Extracts text from PDF documents
- Counts total number of tokens
- Counts unique tokens
- Shows the most frequent tokens

## Requirements

- Python 3.6+
- PyPDF2
- NLTK

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with a PDF file path as an argument:

```bash
python pdf_token_counter.py path/to/your/document.pdf
```

### Options

- `--top N`: Show the top N most frequent tokens (default: 10)

Example:

```bash
python pdf_token_counter.py path/to/your/document.pdf --top 20
```

## Example Output

```
Processing PDF: document.pdf
PDF has 10 pages

Results:
Total tokens: 5000
Unique tokens: 1200

Top 10 most frequent tokens:
  the: 300
  and: 250
  to: 200
  of: 180
  a: 150
  in: 120
  is: 100
  for: 90
  that: 80
  with: 70
``` 