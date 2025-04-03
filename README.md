# FSRDC Research Output Analysis

This project analyzes research outputs related to Federal Statistical Research Data Centers (FSRDC) through multiple data sources and processing methods.

## Project Structure

```
.
├── data/
│   ├── raw/                  # Raw data files
│   │   ├── ResearchOutputs.xlsx
│   │   ├── ProjectsAllMetadata.xlsx
│   │   ├── cleaned_biblio.csv
│   │   └── cleaned_data.csv
│   └── processed/            # Processed data files
│       ├── final_deduped_data.csv
│       ├── final_deduped_data_withkeyword.csv
│       ├── merged_3_enriched_data.csv
│       ├── enriched_scraped_data_openalex.csv
│       ├── enriched_cleaned_data_openalex.csv
│       ├── deduplicate_self.csv
│       ├── duplicate_data.csv
│       ├── duplicate_entries.csv
│       ├── deduplicated_scraped_data.csv
│       ├── scraped_data.csv
│       ├── scraped_data_intermediate.csv
│       ├── fsrdc5_related_papers_api_all.csv
│       ├── failed_projects.csv
│       └── processing.log
├── src/
│   ├── main.py               # Main entry point
│   ├── data_processing.py    # Data processing and deduplication
│   ├── web_scraping.py       # Web scraping functionality
│   └── api_integration.py    # API integration for citation analysis
└── README.md
```

## Code Execution Flow

The project follows a sequential execution flow:

1. **Main Entry Point** (`main.py`)
   - Checks for existence of processed data files
   - If files don't exist, triggers data processing
   - Coordinates the overall workflow

2. **Data Processing** (`data_processing.py`)
   - Reads raw data from `ResearchOutputs.xlsx`
   - Processes web scraping data
   - Processes API data
   - Enriches data with OpenAlex API
   - Deduplicates and merges datasets
   - Outputs final processed dataset

3. **Web Scraping** (`web_scraping.py`)
   - Scrapes research data from web sources
   - Extracts relevant information
   - Saves to intermediate CSV files

4. **API Integration** (`api_integration.py`)
   - Connects to OpenAlex API
   - Retrieves paper metadata
   - Analyzes citation networks
   - Identifies FSRDC-related papers
   - Saves results to CSV

## Data Processing Pipeline

1. **Initial Data Loading**
   - Read `ResearchOutputs.xlsx` (1735 records)
   - Process web scraping data (943 → 652 records after deduplication)
   - Process API data (3642 → 2840 records after deduplication)

2. **Data Enrichment**
   - Supplement metadata using OpenAlex API
   - Add keywords and abstracts
   - Extract author and institution information

3. **Deduplication**
   - Remove duplicates within each data source
   - Cross-reference between sources
   - Apply fuzzy matching for similar titles

4. **Final Dataset Creation**
   - Merge enriched datasets
   - Apply FSRDC relevance filtering
   - Generate final output files

## Output Files

- `data/processed/final_deduped_data.csv`: Main deduplicated dataset
- `data/processed/final_deduped_data_withkeyword.csv`: Dataset with added keywords
- `data/processed/merged_3_enriched_data.csv`: Merged enriched datasets
- `data/processed/enriched_scraped_data_openalex.csv`: Enriched web scraping data
- `data/processed/enriched_cleaned_data_openalex.csv`: Enriched cleaned data
- `data/processed/duplicate_data.csv`: Records of duplicate entries
- `data/processed/processing.log`: Processing log file

## Running the Project

1. Ensure all required Python packages are installed:
   ```
   pip install pandas numpy requests beautifulsoup4 tqdm
   ```

2. Place raw data files in `data/raw/`:
   - `ResearchOutputs.xlsx`
   - `ProjectsAllMetadata.xlsx`
   - `cleaned_biblio.csv`
   - `cleaned_data.csv`

3. Run the main script:
   ```
   python src/main.py
   ```

4. The script will automatically:
   - Check for existing processed files
   - Process data if needed
   - Generate output files in `data/processed/`

## Notes

- The API integration uses rate limiting to respect API constraints
- Processing may take significant time due to API calls
- Progress is logged to `data/processed/processing.log` 