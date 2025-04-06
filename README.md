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
- Performs statistical analysis to discover patterns and relationships

## Complete Project Structure

```
.
├── data/
│   ├── raw/                          # Original input data files
│   │   ├── ResearchOutputs.xlsx      # Original 2024 FSRDC dataset provided
│   │   ├── ProjectsAllMetadata.xlsx  # Metadata about FSRDC projects and PIs
│   │   ├── cleaned_biblio.csv        # Processed version of ResearchOutputs
│   │   └── cleaned_data.csv          # Additional cleaned baseline data
│   ├── processed/                    # Intermediate and final processed data
│   │   ├── New_And_Original_ResearchOutputs.csv   # Final merged dataset with all research outputs
│   │   ├── deduplicated_scraped_data.csv          # Web scraped data after deduplication
│   │   ├── duplicate_data.csv                     # Records identified as duplicates
│   │   ├── enriched_cleaned_data_openalex.csv     # Baseline data enriched with OpenAlex metadata
│   │   ├── enriched_scraped_data_openalex.csv     # Scraped data enriched with OpenAlex metadata
│   │   ├── failed_projects.csv                    # Log of scraping failures
│   │   ├── final_deduped_data.csv                 # API data after deduplication
│   │   ├── final_deduped_data_withkeyword.csv     # Deduplicated API data with keywords
│   │   ├── fsrdc5_related_papers_api_all.csv      # Raw API data before processing
│   │   ├── merged_3_enriched_data.csv             # Intermediate merged dataset from all sources
│   │   └── scraped_data.csv                       # Raw data from web scraping
│   └── logs/                         # Log files from all processing stages
│       ├── data_processing.log       # Logs from data processing stage
│       ├── graph_analysis.log        # Logs from graph analysis operations
│       └── web_scraping.log          # Logs from web scraping operations
├── output/
│   ├── analysis_results.pkl          # Serialized analysis results from graph construction
│   └── visualizations/               # Generated interactive visualizations
│       ├── author_graph.html                      # Author collaboration network
│       ├── centrality_distribution.html           # Distribution of node centralities
│       ├── clustering_distribution.html           # Distribution of clustering coefficients
│       ├── community_distribution.html            # Size distribution of communities
│       ├── community_heatmap.html                 # Heatmap of inter-community connections
│       ├── institution_collaboration.html         # Institution collaboration network
│       ├── keyword_graph.html                     # Keyword co-occurrence network
│       ├── main_graph.html                        # Main research papers graph
│       ├── publication_trend.html                 # Temporal trend of publications
│       ├── research_output_graph.html             # Paper-to-paper relationships
│       ├── simulation_results.html                # Results from DES simulation
│       ├── top_institution_collaborations.html    # Top collaborating institution pairs
│       ├── top_keywords.html                      # Most frequent keywords
│       ├── top_keyword_cooccurrences.html         # Most frequent keyword pairs
│       ├── topic_evolution.html                   # Evolution of research topics over time
│       └── year_subplots.html                     # Combined annual research metrics
├── results/
│   ├── graph_metrics.json            # Detailed network metrics in JSON format
│   └── statistics.json               # Summary statistics and DES simulation results
├── src/
│   ├── main.py                       # Main entry point script that orchestrates all processing
│   ├── web_scraping.py               # Web scraping functionality to collect research metadata
│   ├── api_integration.py            # API integration for retrieving citation data and metadata
│   ├── data_processing.py            # Data processing, cleaning, and entity resolution
│   ├── graph_analysis.py             # Network construction and analysis algorithms
│   └── visualization.py              # Generation of interactive visualizations
├── tests/
│   └── test_utils.py                 # Unit tests for utility functions
├── EDA.ipynb                         # Exploratory Data Analysis notebook with interactive visualizations
├── report.md                         # Comprehensive project report
└── requirements.txt                  # Python dependencies for the project
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

## Key File Descriptions

### Source Code Files

- **src/main.py**: The primary script that orchestrates the entire pipeline, checking for existing outputs and executing only necessary steps.
- **src/web_scraping.py**: Implements the `WebScraper` class that extracts metadata from the FSRDC website using BeautifulSoup, with robust error handling and rate limiting.
- **src/api_integration.py**: Contains functions to query the OpenAlex API to discover papers citing known FSRDC research and retrieve comprehensive metadata.
- **src/data_processing.py**: Implements data cleaning, deduplication using fuzzy matching, entity resolution, and metadata enrichment across multiple data sources.
- **src/graph_analysis.py**: Defines the `ResearchGraphBuilder` class for constructing various network graphs and computing metrics, along with the `ResearchDES` class for discrete event simulation.
- **src/visualization.py**: Generates interactive visualizations using Plotly, including network graphs, statistical distributions, and temporal trends.

### Key Data Files

- **data/raw/ResearchOutputs.xlsx**: The original 2024 FSRDC baseline dataset containing metadata about known research outputs.
- **data/processed/New_And_Original_ResearchOutputs.csv**: The final merged dataset containing all research outputs (original, API-derived, and web-scraped) with enriched metadata.
- **output/analysis_results.pkl**: Serialized Python object containing all computed graph structures and analysis results.
- **results/graph_metrics.json**: Comprehensive network metrics including centrality measures, clustering coefficients, and community detection results in JSON format.
- **results/statistics.json**: Summary statistics and results from the Discrete Event Simulation of the research publication lifecycle.

### Visualization Outputs

All visualizations are interactive HTML files generated using Plotly and saved in the `output/visualizations/` directory:

- Network visualizations: `author_graph.html`, `institution_collaboration.html`, `keyword_graph.html`, etc.
- Statistical distributions: `centrality_distribution.html`, `clustering_distribution.html`, etc.
- Temporal analyses: `publication_trend.html`, `topic_evolution.html`, etc.
- Summary visualizations: `top_institution_collaborations.html`, `top_keywords.html`, etc.

## Data Processing Pipeline

1. **Data Collection**
   - Web scraping from FSRDC website (`web_scraping.py`)
   - API integration with OpenAlex (`api_integration.py`)
   - Loading existing research output data

2. **Data Processing & Deduplication** (`data_processing.py`)
   - Standardizing text fields
   - Removing duplicate records using fuzzy matching
   - Cross-referencing between data sources
   - FSRDC relevance validation

3. **Metadata Enrichment** (`data_processing.py`)
   - Adding citation information
   - Extracting keywords
   - Standardizing author and institution information

4. **Graph Analysis** (`graph_analysis.py`)
   - Building collaboration networks (institution, author)
   - Creating keyword co-occurrence graphs
   - Analyzing temporal patterns
   - Computing network metrics (centrality, communities)
   - Running Discrete Event Simulation (DES)

5. **Visualization** (`visualization.py`)
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
  ast
  logging
  json
  matplotlib
  scipy
  statsmodels
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ellison097/Group1_Project.git
   cd Group1_Project
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
4. **JSON Results**: Structured metrics and statistics in machine-readable format

## Key Findings

The analysis reveals several important insights about the FSRDC research ecosystem:

- The volume of FSRDC-related research has grown exponentially since 2000, with particularly rapid expansion after 2010
- The institutional network exhibits a scale-free structure with the University of California system, National Bureau of Economic Research, and major research universities forming central hubs
- Research communities organize primarily around research domains with secondary clustering by geographic regions
- Key institutions like University of California (total score 0.226) and National Bureau of Economic Research (total score 0.183) demonstrate the highest centrality and influence in the collaboration network
- FSRDC research evolved from early focus on economic indicators to more diverse applications including data science and computational methods in recent years

## Notes

- First-time execution will take longer due to API calls and data processing
- Subsequent runs will be faster by loading cached results
- The system handles rate limiting for API calls to avoid overloading external services
- All progress and errors are logged in the `data/logs/` directory
- Visualizations are interactive HTML files that can be opened in any modern web browser

## Acknowledgments

This project utilizes data from:
- Federal Statistical Research Data Centers (FSRDC)
- OpenAlex API for scholarly data 