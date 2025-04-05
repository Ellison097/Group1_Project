# FSRDC Research Output Analysis

This project provides comprehensive analysis and visualization of research outputs related to Federal Statistical Research Data Centers (FSRDC). It combines web scraping, API integration, data processing, graph analysis, and interactive visualization to provide insights into research trends, collaboration networks, and impact metrics.

## Project Overview

The system performs the following key functions:
- Collects research data from web sources and APIs
- Processes and deduplicates data from multiple sources
- Enriches metadata using OpenAlex API
- Constructs various network graphs (author collaboration, institution collaboration, keyword co-occurrence, citation)
- Calculates network metrics and performs community detection
- Generates interactive visualizations for research analysis

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
│       ├── New_And_Original_ResearchOutputs.csv   # Final merged dataset
│       ├── merged_3_enriched_data.csv             # Merged enriched data
│       ├── enriched_scraped_data_openalex.csv     # Enriched API data
│       ├── enriched_cleaned_data_openalex.csv     # Enriched cleaned data
│       ├── final_deduped_data_withkeyword.csv     # API data with keywords
│       ├── fsrdc5_related_papers_api_all.csv      # API data(raw)
│       ├── scraped_data.csv                       # Web scraped data
│       └── processing.log                         # Processing log
├── output/
│   ├── analysis_results.pkl                       # Saved analysis results
│   └── visualizations/                            # Generated visualizations
│       ├── author_graph.html                      # Author collaboration network
│       ├── citation_graph.html                    # Citation network
│       ├── keyword_graph.html                     # Keyword co-occurrence
│       ├── institution_collaboration.html         # Institution collaboration
│       ├── publication_by_year.html               # Temporal analysis
│       └── ... (other visualizations)
├── src/
│   ├── main.py                                    # Main entry point
│   ├── web_scraping.py                            # Web scraping functionality
│   ├── api_integration.py                         # API integration
│   ├── data_processing.py                         # Data processing & deduplication
│   ├── graph_analysis.py                          # Network analysis
│   └── visualization.py                           # Visualization generation
├── EDA.ipynb                                      # Exploratory Data Analysis
└── README.md                                      # This file
```

## Execution Strategies

You have two options for running this project:

### Option 1: Sequential Module Execution

Run each module independently in the following order:

1. **Web Scraping**: Collect data from web sources
   ```bash
   python src/web_scraping.py
   ```

2. **API Integration**: Fetch citation data and metadata
   ```bash
   python src/api_integration.py
   ```

3. **Data Processing**: Clean, deduplicate and merge data
   ```bash
   python src/data_processing.py
   ```

4. **Graph Analysis**: Construct networks and calculate metrics
   ```bash
   python src/graph_analysis.py
   ```

5. **Visualization**: Generate interactive visualizations
   ```bash
   python src/visualization.py
   ```

6. **Exploratory Data Analysis**: Run the Jupyter notebook
   ```bash
   jupyter notebook EDA.ipynb
   ```

This approach gives you more control over each step and lets you inspect intermediate results.

### Option 2: One-Click Execution

Run the main script to execute the entire pipeline:

```bash
python src/main.py
```

This will:
1. Check for existing data files
2. Process only what's needed (skip steps with existing outputs)
3. Generate all analysis results and visualizations

This approach is more efficient for repeated runs as it avoids reprocessing existing data.

## Data Processing Pipeline

1. **Data Collection**
   - Web scraping from FSRDC website
   - API integration with OpenAlex
   - Loading existing research output data

2. **Data Processing & Deduplication**
   - Standardizing text fields
   - Removing duplicate records
   - Cross-referencing between data sources

3. **Metadata Enrichment**
   - Adding citation information
   - Extracting keywords
   - Standardizing author and institution information

4. **Graph Analysis**
   - Building collaboration networks
   - Creating keyword co-occurrence graphs
   - Analyzing citation patterns
   - Computing network metrics (centrality, communities)

5. **Visualization**
   - Interactive network graphs
   - Statistical charts and trends
   - Community visualizations
   - Temporal analysis

## Requirements

- Python 3.8+
- Required packages:
  ```
  pandas
  numpy
  networkx
  plotly
  requests
  beautifulsoup4
  thefuzz
  tqdm
  scikit-learn
  community
  simpy
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ellison097/Group1_Project.git
   cd fsrdc-research-analysis
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure raw data files are in place:
   - Place `ResearchOutputs.xlsx` and other raw files in `data/raw/`

## Outputs

The system generates:

1. **Processed Data Files**: Cleaned, deduplicated, and enriched datasets
2. **Analysis Results**: Network metrics, community detection, influence scores
3. **Interactive Visualizations**: Network graphs, statistical charts, trend analyses

All visualizations are saved as interactive HTML files in the `output/visualizations/` directory.

## Notes

- First-time execution will take longer due to API calls and data processing
- Subsequent runs will be faster by loading cached results
- The system handles rate limiting for API calls to avoid overloading external services
- All progress and errors are logged in `data/processed/processing.log`

## Acknowledgments

This project utilizes data from:
- Federal Statistical Research Data Centers (FSRDC)
- OpenAlex API for scholarly data 