# FSRDC Research Outputs Collection

This project aims to collect and process research outputs related to Federal Statistical Research Data Centers (FSRDC) through web scraping and API integration.

## Project Structure

```
Group1_Project/
├── src/
│   ├── web_scraping.py      # Web scraping for research outputs
│   ├── api_integration.py   # API data collection
│   └── data_processing.py   # Data processing and deduplication
├── data/
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data files
└── README.md
```

## Features

1. Web Scraping
   - Extracts metadata from various sources
   - Implements FSRDC criteria checking
   - Handles rate limiting and errors

2. API Integration
   - Connects to multiple APIs (OpenAlex, CrossRef, etc.)
   - Processes API responses
   - Implements error handling

3. Data Processing
   - Deduplication using exact and fuzzy matching
   - FSRDC criteria validation
   - Data cleaning and standardization

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure data directories:
```bash
mkdir -p data/raw data/processed
```

3. Place input files:
- Put `ProjectsAllMetadata.xlsx` in `data/raw/`
- Put `ResearchOutputs.xlsx` in `data/raw/`

## Usage

1. Run web scraping:
```bash
python src/web_scraping.py
```

2. Run API integration:
```bash
python src/api_integration.py
```

3. Process and deduplicate data:
```bash
python src/data_processing.py
```

## Output Files

- `data/processed/scraped_data.csv`: Web scraping results
- `data/processed/api_data.csv`: API integration results
- `data/processed/deduplicated_scraped_data.csv`: Deduplicated web scraping results
- `data/processed/final_deduped_data.csv`: Final deduplicated results
- `data/processed/final_deduped_data_withkeyword.csv`: Results with OpenAlex keywords 