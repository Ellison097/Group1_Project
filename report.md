# Group1: FSRDC Research Output Analysis
###### Yukun Gao (gaoyukun@seas.upenn.edu)
###### Licheng Guo (guolc@seas.upenn.edu)
###### Yixuan Xu (xuyixuan@seas.upenn.edu)
###### Zining Hua (znhua@seas.upenn.edu)
###### Qiming Zhang (zqiming@seas.upenn.edu) 
###### Yongchen Lu (ylu178@seas.upenn.edu)

## 0. Introduction

This report presents our approach to discovering and analyzing research outputs related to Federal Statistical Research Data Centers (FSRDC). The Federal Statistical Research Data Centers provide researchers with secure access to restricted microdata from various federal statistical agencies, including the Census Bureau, Bureau of Economic Analysis (BEA), and Internal Revenue Service (IRS). These data centers play a crucial role in enabling research that informs public policy and economic understanding.

Our project aims to identify and analyze research outputs that utilize FSRDC data through a comprehensive five-stage approach:

1. **Web Scraping**: Extracting metadata from publicly available sources to identify potential FSRDC-related research outputs.
2. **API Integration**: Utilizing academic APIs to retrieve comprehensive metadata about research papers and trace citation networks.
3. **Data Processing and Entity Resolution**: Cleaning, deduplicating, and validating the collected data to ensure quality and relevance.
4. **Graph Construction and Analysis**: Building and analyzing complex networks of institutions, authors, and research topics to uncover collaboration patterns and influential nodes.
5. **Data Visualization and Statistical Analysis**: Creating interactive visualizations and applying statistical techniques to derive actionable insights from the FSRDC research landscape.

The methodology employed in this project addresses several key challenges in research data curation and analysis:
- Identifying research outputs that use restricted data without direct access to the data itself
- Handling variations in how authors and institutions reference FSRDC
- Ensuring comprehensive coverage while avoiding duplicates with existing datasets
- Validating the relevance of research outputs to FSRDC
- Understanding complex collaboration networks and research evolution patterns
- Extracting meaningful insights about research impact and institutional influence

Our analysis reveals a rich and evolving landscape of FSRDC research with several notable findings:
- The volume of FSRDC-related research has grown exponentially since 2000, with particularly rapid expansion after 2010
- The institutional network exhibits a scale-free structure with the Census Bureau and major research universities forming central hubs
- Research communities organize primarily around research domains with secondary clustering by geographic regions
- Publication impact correlates significantly with institutional network position and collaboration patterns
- Clear temporal trends in research topics illustrate the evolution of FSRDC data usage over time

This report details our implementation of each stage of the analysis pipeline, the challenges encountered, and the results achieved. By combining web scraping, API integration, network analysis, and advanced visualization techniques, we have developed a comprehensive framework for understanding the FSRDC research ecosystem and its evolution over time. Our findings provide valuable insights for researchers, institutions, and policymakers seeking to maximize the scientific and societal impact of restricted federal data resources.

## 1. Web Scraping

Our web scraping approach focuses on extracting research output metadata from the Federal Statistical Research Data Centers (FSRDC) website. The implementation uses a systematic process with robust error handling and rate limiting to ensure reliable data collection.

### 1.1 Implementation Architecture 

The web scraping component is implemented in `web_scraping.py` with a `WebScraper` class that handles:

1. **Base Configuration**
   - Targets the FSRDC research outputs page (fsrdc.org/research-outputs/)
   - Uses configurable request headers to identify as a legitimate client
   - Implements output paths for both successful and failed scraping attempts

2. **Robust Request Handling**
   - Employs exponential backoff retry mechanism for failed requests
   - Implements rate limiting to avoid server overload
   - Maintains detailed logging of the scraping process

3. **Content Extraction**
   - Uses BeautifulSoup for HTML parsing
   - Extracts structured metadata including:
     - Project titles
     - Abstracts
     - Author information
     - Publication years
     - Keywords

### 1.2 Data Collection Process

The scraper follows a two-stage process:

```python
def scrape_all(self):
    """Main scraping process"""
    # 1. Fetch main page and extract project links
    content = self.get_page_content(self.base_url)
    soup = BeautifulSoup(content, "html.parser")
    project_links = soup.find_all("a", class_="project-link")

    # 2. Process individual project pages
    for link in project_links:
        project_url = link.get("href")
        if project_url:
            project_data = self.parse_project_page(project_url)
            # Store and process results...
```

### 1.3 Error Handling and Quality Control

The implementation includes several quality control measures:

1. **Request Retry Logic**
   - Maximum of 3 retry attempts per request
   - Exponential backoff between attempts (1, 2, 4 seconds)
   - Detailed error logging for failed requests

2. **Data Validation**
   - Checks for missing or malformed content
   - Maintains separate tracking of failed scraping attempts
   - Saves failed project information for later analysis

3. **Output Management**
   - Structured CSV output for successful scrapes
   - Separate logging of failed attempts
   - Maintains processing logs for debugging

### 1.4 Integration with Processing Pipeline

The scraped data undergoes several processing steps after collection:

1. **Initial Processing** (`data_processing.py`)
   - Deduplication against existing research outputs
   - Standardization of author names and institutions
   - Enrichment with additional metadata

2. **API Integration** (`api_integration.py`)
   - Enhancement with OpenAlex metadata
   - Addition of citation information
   - Keyword and institution standardization

3. **Final Validation** (`data_processing.py`)
   - FSRDC relevance verification
   - Data completeness checks
   - Format standardization for graph analysis

### 1.5 Results and Impact

The web scraping component successfully:

1. **Data Collection**
   - Extracts metadata from the FSRDC research outputs page
   - Captures detailed project information
   - Maintains data quality through robust error handling

2. **Integration**
   - Provides structured data for the processing pipeline
   - Enables enrichment through API integration
   - Supports comprehensive research output analysis

3. **Quality Assurance**
   - Tracks failed scraping attempts for manual review
   - Maintains detailed processing logs
   - Enables continuous improvement of the scraping process

This implementation provides a reliable foundation for discovering and analyzing FSRDC research outputs, feeding into subsequent stages of data processing and analysis.

## 2. API Integration

Our approach to discovering new FSRDC research outputs builds on Project 1's dataset by analyzing citation networks. Starting with **Donald Moratz's cleaned bibliography (cleaned_biblio.csv)** which is the cleaned verion of ResearchOutputs.xlsx provided , we trace forward citations to identify papers that cite known FSRDC research. This approach is based on the reasoning that:

1. Papers citing FSRDC research are likely to:
   - Use similar restricted datasets
   - Build upon FSRDC-based findings
   - Include authors familiar with FSRDC processes

2. By analyzing the entire citation network, we can:
   - Discover research outputs not captured in the 2024 dataset
   - Identify emerging research trends
   - Map the influence of FSRDC research

### 2.1 API Integration Strategy

We chose OpenAlex API for this process because it provides:
- Comprehensive paper metadata including abstracts and affiliations
- Standardized institution and author information
- Free access with reasonable rate limits
- Structured data format

The `api_integration.py` script implements this strategy through several key components:

1. **Data Processing Pipeline**
   - Processes papers from the cleaned bibliography
   - Implements rate limiting (0.15s between requests)
   - Handles API errors with robust error checking

2. **Metadata Enhancement**
   - Enriches papers with complete metadata
   - Reconstructs abstracts from OpenAlex's inverted index format
   - Standardizes author and institution information

3. **FSRDC Relevance Verification**
   - Validates papers against FSRDC-related criteria
   - Uses carefully selected keywords for relevance checking
   - Maintains detailed matching records

### 2.2 Keyword Selection Strategy

The implementation uses a comprehensive set of keywords to identify FSRDC-related research:

```python
fsrdc_keywords = [
    # Core FSRDC terms
    "census bureau", "cecus", "bureau", "fsrdc", "fsrdc data",
    "research data center", "rdc",
    
    # Federal statistical agencies
    "bea", "irs", "internal revenue service", "federal reserve",
    "nber", "cepr", "national bureau of economic research",
    
    # Data access terminology
    "restricted microdata", "confidential data", "restricted data",
    "microdata", "confidential data", "confidential microdata", "restricted",
    
    # Major Census datasets
    "annual survey of manufactures",
    "census of construction industries", 
    "census of agriculture",
    "census of retail trade",
    "census of manufacturing",
    "census of transportation",
    "census of population"
]
```


The keywords were selected to capture mentions of key federal statistical agencies (Census Bureau, BEA, IRS), different ways of referring to research data centers (FSRDC, RDC), various types of restricted and confidential data access, and specific census datasets like the Annual Survey of Manufactures and Census of Population.

### 2.3 Data Collection Process

The metadata collection process involves several key steps:

1. **Abstract Processing**
   ```python
   def reconstruct_abstract(inverted_index):
       """Reconstruct abstract from OpenAlex's inverted index format"""
       if not isinstance(inverted_index, dict) or not inverted_index:
           return "No abstract available."
       
       max_index = max(pos for positions in inverted_index.values() for pos in positions)
       words = [None] * (max_index + 1)
       
       for word, positions in inverted_index.items():
           for pos in positions:
               words[pos] = word
               
       return " ".join(word for word in words if word is not None)
   ```

2. **Paper Metadata Retrieval** (`get_paper_metadata`)
   - Makes API calls to OpenAlex with rate limiting
   - Extracts comprehensive metadata including:
     - Basic paper information (title, year, DOI)
     - Author information and affiliations
     - Abstract and keywords
   ```python
   def get_paper_metadata(title_query: str, sleep_time: float = 0.15):
       url = f"https://api.openalex.org/works?search={title_query.replace(' ', '%20')}"
       headers = {"User-Agent": "YourProject (justin.zhang@wsu.edu)"}
       # ... (API call and data processing)
   ```

3. **FSRDC Relevance Check** (`is_fsrdc_related`)
   - Evaluates whether a paper's content aligns with FSRDC research criteria (initial assessment phase, followed by more rigorous evaluation criteria in subsequent analysis)
   - Searches for keywords in:
     - Paper title
     - Abstract
     - Institution names
     - Author affiliations
   - Uses a carefully curated list of FSRDC-related keywords based on:
     - Common FSRDC data sources and institutions
     - Restricted data access terminology
     - Federal statistical agencies
     - Research center nomenclature
   ```python
    fsrdc_keywords = [
    # Core FSRDC terms
    "census bureau", "cecus", "bureau", "fsrdc", "fsrdc data",
    "research data center", "rdc",
    
    # Federal statistical agencies
    "bea", "irs", "internal revenue service", "federal reserve",
    "nber", "cepr", "national bureau of economic research",
    
    # Data access terminology
    "restricted microdata", "confidential data", "restricted data",
    "microdata", "confidential data", "confidential microdata", "restricted",
    
    # Major Census datasets
    "annual survey of manufactures",
    "census of construction industries", 
    "census of agriculture",
    "census of retail trade",
    "census of manufacturing",
    "census of transportation",
    "census of population"
    ]
   ```
   
   These keywords were selected based on:
   - Analysis of known FSRDC publications from Projects 1 and 2
   - Common terminology in restricted data research
   - Major census datasets frequently used in FSRDC research
   - Variations in institutional names and abbreviations
   - Key federal statistical agencies involved in FSRDC

### 2.3.2 Main Processing Function

The `process_csv_and_find_citations` function orchestrates the entire data collection process through several key stages:

1. **Initial Setup and Data Loading**
   ```python
   def process_csv_and_find_citations(
       input_file: str,
       output_file: str,
       title_column: str,
       year_column: str = None,
       sleep_time: float = 0.15,
   ):
   ```
   - Loads and validates the input bibliography
   - Implements rate limiting (0.15s default) to respect API constraints
   - Sorts papers by year to prioritize recent research
   - Sets up CSV writer with comprehensive metadata fields

2. **Citation Network Traversal**
   - For each seed paper from the bibliography:
     ```python
     for idx, row in tqdm(list(papers_to_process.iterrows()), desc="Processing papers"):
         title = str(row[title_column]).strip()
         paper_data = get_paper_metadata(title, sleep_time)
     ```
     - Retrieves paper metadata from OpenAlex
     - Extracts all papers citing this work
     - Maintains a set of processed titles to avoid duplicates

3. **Citing Paper Analysis**
   For each citing paper:
   ```python
   for citing_work in citing_works:
       citing_title = citing_work.get("title")
       if citing_title in processed_citing_titles:
           continue
       citing_paper_data = get_paper_metadata(citing_title, sleep_time)
   ```
   - Checks for duplicate processing
   - Retrieves full metadata
   - Analyzes for FSRDC relevance

4. **FSRDC Relevance Assessment**
   - Searches for keywords across multiple fields:
     ```python
     # Check title
     if keyword.lower() in citing_paper_data.get("title", "").lower():
         matching_keywords.append(keyword)
     # Check abstract
     elif keyword.lower() in citing_paper_data.get("abstract", "").lower():
         matching_keywords.append(keyword)
     # Check institutions and affiliations...
     ```
   - Accumulates matching keywords for relevance scoring
   - Records all matches for later analysis

5. **Metadata Extraction and Organization**
   For relevant papers, collects:
   - Author information:
     ```python
     display_authors = []
     raw_authors = []
     for authorship in citing_paper_data.get("authorships", []):
         author = authorship.get("author", {})
         display_name = author.get("display_name")
         raw_name = authorship.get("raw_author_name")
     ```
   - Institution details:
     ```python
     institution_names = set()
     raw_affiliations = set()
     detailed_affiliations = set()
     ```
   - Complete paper metadata including:
     - Title and publication year
     - DOI and dataset information
     - Abstract and keywords
     - Institutional affiliations
     - Matching FSRDC criteria

6. **Output Generation**
   - Writes comprehensive records to CSV:
     ```python
     writer.writerow({
         "title": citing_paper_data.get("title", ""),
         "year": citing_paper_data.get("year", ""),
         "dataset": citing_paper_data.get("datasets", []),
         "display_author_names": "; ".join(display_authors),
         # ... additional fields
     })
     ```
   - Maintains progress tracking and statistics
   - Provides detailed logging of matches and processing steps

This systematic approach ensures:
- Complete coverage of citation networks
- Thorough metadata collection
- Accurate FSRDC relevance assessment
- Efficient processing with duplicate prevention
- Comprehensive documentation of matches and criteria

The process creates a rich dataset that captures not just paper titles, but the full context of FSRDC-related research, including institutional connections, author networks, and specific relevance criteria matches.

### 2.4 Acknowledgments Considerations

In our current API integration process using OpenAlex, we successfully retrieve comprehensive metadata for research papers—including title, publication year, DOI, dataset information, abstract, keywords, institutional affiliations, and matching FSRDC criteria. However, the Acknowledgments section is notably absent from the OpenAlex data.

The project requirements specified the integration of several APIs, such as:

- Dimensions API – Although this API is known for its comprehensive metadata (including potential Acknowledgments information), the free version requires an approval process. As noted by our instructor, the approval process is pending; therefore, we have not been able to utilize this API.

- NSF's PAR and NIH's PMC – These APIs could theoretically supply Acknowledgments details. However, our tests revealed that they return a very limited number of papers. Specifically, NSF's PAR mainly provides papers from NSF-funded projects in the physical sciences, engineering, and related fields (with some representation in the social sciences), while NIH's PMC is almost exclusively focused on biomedical and life sciences research. This specialized coverage does not meet the broader scope required for our dataset.

- Other APIs (ORCID, Preprint repositories, BASE API, CORE API) – While these sources offer valuable metadata, they do not reliably provide the Acknowledgments information required by the project.

Based on these considerations, we opted to use the OpenAlex API exclusively. This decision was made to ensure consistency and comprehensive coverage of the primary metadata fields for our analysis, despite the absence of Acknowledgments data. In future work, should the Dimensions API be approved or improved access be established for NSF's PAR and NIH's PMC, we may revisit the inclusion of Acknowledgments to further enrich the dataset.

## 3. Data Processing and Entity Resolution

After collecting data from web scraping and API integration, we implemented a comprehensive data processing pipeline in `data_processing.py`. This pipeline ensures data quality, relevance, and standardization by cleaning, deduplicating, enriching, and merging data from multiple sources before the final graph analysis stage.

### 3.1 Data Loading and Initial Cleaning

The pipeline begins by loading data from three primary sources:
- Web scraping results (`scraped_data.csv`)
- API integration results (`fsrdc5_related_papers_api_all.csv`)
- The original 2024 FSRDC dataset (`ResearchOutputs.xlsx`) and a cleaned version (`cleaned_biblio.csv`) used for deduplication and enrichment.

Initial cleaning involves handling missing values, standardizing basic formats, and logging operations for traceability.

### 3.2 Entity Resolution and Deduplication

A multi-stage deduplication process is applied to ensure uniqueness and avoid overlap with existing datasets:

1.  **API Data Deduplication (`process_api_data`)**:
    *   **Self-Deduplication**: Removes exact duplicates based on 'title' within the API-derived dataset.
    *   **Cross-Dataset Fuzzy Matching**: Compares API titles against `cleaned_biblio.csv` using fuzzy matching (`is_similar` function with threshold 80) to remove papers already present in the cleaned bibliography.
    ```python
    # Within process_api_data function
    deduplicate_self = df.drop_duplicates(subset=["title"], keep="first")
    # ... fuzzy matching logic against cleaned_biblio ...
    after_fuzzy_df = deduplicate_self[keep_rows].reset_index(drop=True)
    ```

2.  **Scraped Data Deduplication (`check_duplicates_with_research_outputs`)**:
    *   Compares scraped data titles against the original `ResearchOutputs.xlsx` using both exact and fuzzy matching to filter out duplicates.
    ```python
    def check_duplicates_with_research_outputs(scraped_data, research_outputs):
        """Check for duplicates between scraped data and ResearchOutputs.xlsx"""
        # ... exact and fuzzy matching logic ...
        deduplicated_data = scraped_data[keep_rows].reset_index(drop=True)
        # Save removed duplicates for review
        duplicate_data = scraped_data[~scraped_data.index.isin(deduplicated_data.index)]
        if not duplicate_data.empty:
            duplicate_data.to_csv('data/processed/duplicate_data.csv', index=False)
        return deduplicated_data
    ```

### 3.3 FSRDC Relevance Filtering (API Data)

The deduplicated API data undergoes a relevance check based on keywords identified during the API integration step:

```python
# Within process_api_data function
def count_keywords(keywords_str):
    """Calculate keyword count"""
    if pd.isna(keywords_str): return 0
    return len(str(keywords_str).split(", "))

# Filter records with 2 or more matching keywords
after_fuzzy_df_larger2 = after_fuzzy_df[
    after_fuzzy_df["match_rdc_criteria_keywords"].apply(count_keywords) >= 2
].reset_index(drop=True)
logger.info(f"Data count after keyword filtering: {len(after_fuzzy_df_larger2)}")
# Save this intermediate result
after_fuzzy_df_larger2.to_csv("data/processed/final_deduped_data.csv", index=False)
```
Only papers matching at least two FSRDC-related keywords are retained for further processing.

### 3.4 Data Enrichment via OpenAlex

To ensure comprehensive metadata, we enrich multiple datasets using the OpenAlex API:

1.  **API Data Enrichment (`process_api_data`)**: The filtered API data (`final_deduped_data.csv`) is further enriched by fetching and adding standardized keywords directly from OpenAlex.
    ```python
    # Within process_api_data function
    def fetch_openalex_data_by_title(title): # Fetches work object
    def get_openalex_keywords(work): # Extracts keywords from concepts
    # ... loop through after_fuzzy_df_larger2 ...
    work = fetch_openalex_data_by_title(row["title"])
    keywords = get_openalex_keywords(work)
    openalex_keywords.append(keywords)
    # ... add 'Keywords' column and save ...
    after_fuzzy_df_larger2.to_csv("data/processed/final_deduped_data_withkeyword.csv", index=False)
    ```

2.  **Cleaned Data Enrichment (`enrich_cleaned_data`)**: Enriches a separate cleaned dataset (`cleaned_data.csv`) with OpenAlex metadata (DOI, keywords, affiliations).
    ```python
    def enrich_cleaned_data(input_file: str, output_file: str, sleep_time: float = 0.15):
        """Enrich data from input CSV with metadata from OpenAlex"""
        # ... reads input, defines output fields ...
        for idx, row in tqdm(df.iterrows(), ...):
            metadata = get_paper_metadata(title, sleep_time) # Fetches DOI, keywords, affiliations
            if metadata:
                output_row.update(metadata)
            writer.writerow(output_row)
        # Saves to enriched_cleaned_data_openalex.csv
    ```

3.  **Scraped Data Enrichment (`enrich_scraped_data`)**: Enriches the deduplicated scraped data (`deduplicated_scraped_data.csv`) similarly with OpenAlex metadata.
    ```python
    def enrich_scraped_data(input_file: str, output_file: str, sleep_time: float = 0.15):
        """Enrich scraped data with metadata from OpenAlex"""
        # ... reads input, defines output fields ...
        for idx, row in tqdm(df.iterrows(), ...):
            metadata = get_paper_metadata(title, sleep_time) # Fetches DOI, keywords, affiliations
            if metadata:
                output_row.update(metadata)
            writer.writerow(output_row)
        # Saves to enriched_scraped_data_openalex.csv
    ```
This multi-pronged enrichment ensures that data from all sources benefits from the structured information available in OpenAlex.

### 3.5 Author and Institution Standardization

Standardization is applied consistently across datasets, primarily during the merging phase:

1.  **Author Name Standardization (`standardize_authors`)**: Consolidates author names from various fields (`authors`, `display_author_names`, `raw_author_names`) into a single semicolon-separated 'authors' column.
    ```python
    def standardize_authors(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize author information across different formats into a single authors column."""
        # ... handles list-like strings and combines author fields ...
        return df
    ```

2.  **Institution Name Normalization**: While a normalization function (`_normalize_institution_name`) exists, it's primarily used later during graph construction (`graph_analysis.py`) rather than directly in the main data processing flow of `data_processing.py`.

### 3.6 Final Data Merging and Refinement

The pipeline culminates in merging the three processed and enriched datasets, followed by a final author refinement step:

1.  **Merging Enriched Datasets (`merge_enriched_data`)**: Combines the enriched scraped, cleaned, and API datasets. This step also involves crucial column cleaning and renaming to create a unified schema.
    ```python
    def merge_enriched_data(
        enriched_scraped_path: str, # enriched_scraped_data_openalex.csv
        enriched_cleaned_path: str, # enriched_cleaned_data_openalex.csv
        enriched_api_path: str,     # final_deduped_data_withkeyword.csv
        output_path: str = "data/processed/merged_3_enriched_data.csv",
    ) -> pd.DataFrame:
        """Merge three enriched datasets while preserving all columns."""
        # ... reads dataframes ...
        # Standardize authors before merging
        scraped_df = standardize_authors(scraped_df)
        cleaned_df = standardize_authors(cleaned_df)
        api_df = standardize_authors(api_df)
        # Perform column drops and renames for standardization
        # e.g., drop 'datasets', rename 'project_end_year' to 'year', etc.
        # ...
        merged_df = pd.concat([scraped_df, cleaned_df, api_df], ignore_index=True)
        merged_df.to_csv(output_path, index=False)
        return merged_df
    ```

2.  **Bibliography Author Filling (`fill_authors_from_biblio`)**: After merging, this function attempts to fill any remaining missing author information in the merged dataset (`merged_3_enriched_data.csv`) by looking up titles in the `cleaned_biblio.csv`. The final, most complete dataset is saved as `New_And_Original_ResearchOutputs.csv`.
    ```python
    def fill_authors_from_biblio():
        """Fill authors from cleaned_biblio.csv into merged_3_enriched_data.csv"""
        merged_df = pd.read_csv("data/processed/merged_3_enriched_data.csv")
        biblio_df = pd.read_csv("data/raw/cleaned_biblio.csv")
        # ... processes biblio authors and creates title->author map ...
        for idx, row in merged_df.iterrows():
            title = row['title']
            if title in title_to_authors:
                # ... fills missing authors ...
        # Saves final dataset
        merged_df.to_csv("data/processed/New_And_Original_ResearchOutputs.csv", index=False)
    ```

### 3.7 Results and Output Files

The `data_processing.py` script generates several important output files throughout its processing stages:

1. **Deduplication Results**:
   - `deduplicate_self.csv`: Initial self-deduplicated API data
   - `duplicate_data.csv`: Records identified as duplicates during scraping deduplication
   - `final_deduped_data.csv`: API data after both deduplication and keyword filtering

2. **Enriched Data Files**:
   - `enriched_scraped_data_openalex.csv`: Scraped data enriched with OpenAlex metadata
   - `enriched_cleaned_data_openalex.csv`: Cleaned data enriched with OpenAlex metadata
   - `final_deduped_data_withkeyword.csv`: API data enriched with OpenAlex keywords

3. **Merged and Final Datasets**:
   - `merged_3_enriched_data.csv`: Combined dataset from all three enriched sources
   - `New_And_Original_ResearchOutputs.csv`: Final dataset with filled author information


Each output file serves a specific purpose:
- **Deduplication files** enable traceability of removed duplicates
- **Enriched files** preserve the enhanced metadata for each data source
- **Merged files** provide both intermediate and final combined datasets
- **Summary and log files** offer insights into the processing results and data quality

The processing pipeline maintains all these files to ensure:
- Complete data provenance tracking
- Ability to debug and validate each processing stage
- Flexibility to use different combinations of processed data
- Transparency in the deduplication and enrichment process

This comprehensive output structure supports both the immediate needs of graph analysis and potential future refinements or alternative analysis approaches.

## 4. Graph Construction and Analysis

In this section, we implemented a graph-based analysis of research outputs, constructing various types of graphs and computing network metrics to identify important nodes and clusters. We also employed Discrete Event Simulation (DES) techniques to model the lifecycle of research outputs.

### 4.1 Graph Construction Methodology

We utilized the NetworkX library to build multiple types of graphs, each capturing different relationships between research outputs:

#### 4.1.1 Main Graph

The main graph includes all research outputs as nodes, with edges established through shared features:

```python
def build_main_graph(self):
    """Build the main graph containing all nodes and multiple edge types"""
    # Add all nodes
    for _, row in self.data.iterrows():
        paper_id = str(row['paper_id'])
        self.G.add_node(paper_id, 
                       doi=row.get('doi'),
                       title=row.get('title'),
                       year=row.get('year'),
                       institution=row.get('institution_display_names'),
                       agency=row.get('Agency'),
                       keywords=row.get('keywords'),
                       abstract=row.get('abstract'))
    
    # Add various types of edges
    self._add_author_edges()
    self._add_keyword_edges()
    self._add_institution_edges()
```

We implemented three different types of edge connections:
- **Author-shared edges**: Connecting papers with common authors
- **Keyword-similarity edges**: Connecting papers with similar keywords based on TF-IDF vectorization
- **Institution-shared edges**: Connecting papers from the same institutions

#### 4.1.2 Institution Graph

The institution graph represents collaborations between research institutions:

```python
def build_institution_graph(self):
    """Build the institution subgraph"""
    # Preprocess institution names
    institution_map = {}
    normalized_institutions = {}
    
    # Standardize institution names and establish paper connections
    for _, row in self.data.iterrows():
        institutions = row['institution_display_names']
        paper_id = str(row['paper_id'])
        
        for institution in institutions:
            if pd.isna(institution):
                continue
                
            normalized_name = self._normalize_institution_name(institution)
            if not normalized_name:
                continue
                
            if normalized_name not in institution_map:
                institution_map[normalized_name] = set()
            institution_map[normalized_name].add(institution)
            
            if normalized_name not in normalized_institutions:
                normalized_institutions[normalized_name] = set()
            normalized_institutions[normalized_name].add(paper_id)
    
    # Add nodes and edges
    for norm_name, papers in normalized_institutions.items():
        if not self.institution_graph.has_node(norm_name):
            self.institution_graph.add_node(norm_name, 
                                          papers=papers,
                                          original_names=list(institution_map[norm_name]))
    
    # Add edges between institutions
    for i, (norm_name1, papers1) in enumerate(normalized_institutions.items()):
        for norm_name2, papers2 in list(normalized_institutions.items())[i+1:]:
            common_papers = len(papers1 & papers2)
            if common_papers > 0:
                self.institution_graph.add_edge(norm_name1, norm_name2, 
                                             weight=common_papers,
                                             common_papers=list(papers1 & papers2))
```

#### 4.1.3 Author Graph

The author graph represents collaborations between researchers:

```python
def build_author_graph(self):
    """Build the author collaboration network graph"""
    self.author_graph = nx.Graph()
    
    # Create author-paper mapping
    author_papers = defaultdict(list)
    for _, row in self.data.iterrows():
        paper_id = str(row['paper_id'])
        authors = row['authors']
        
        for author in authors:
            if pd.notna(author) and author:
                author_papers[author].append(paper_id)
    
    # Add author nodes
    for author, papers in author_papers.items():
        self.author_graph.add_node(author, papers=papers)
    
    # Add collaboration edges
    for i, (author1, papers1) in enumerate(author_papers.items()):
        for author2, papers2 in list(author_papers.items())[i+1:]:
            common_papers = set(papers1) & set(papers2)
            if common_papers:
                self.author_graph.add_edge(author1, author2, 
                                         weight=len(common_papers),
                                         common_papers=list(common_papers))
```

#### 4.1.4 Keyword Graph and Citation Graph

We also constructed keyword co-occurrence and citation networks:

```python
def build_keyword_graph(self):
    """Build the keyword co-occurrence network graph"""
    self.keyword_graph = nx.Graph()
    
    # Create keyword-paper mapping
    keyword_papers = defaultdict(list)
    for _, row in self.data.iterrows():
        paper_id = str(row['paper_id'])
        keywords = row['keywords']
        
        for keyword in keywords:
            if pd.notna(keyword) and keyword:
                keyword_papers[keyword].append(paper_id)
    
    # Add keyword nodes and co-occurrence edges
    # ...

def build_citation_graph(self):
    """Build the citation network graph"""
    self.citation_graph = nx.DiGraph()
    
    # Check for citation information in the data
    if 'references' not in self.data.columns or self.data['references'].isna().all():
        logging.warning("No citation information in the data, cannot build citation graph")
        return
    
    # Process citation information
    # ...
```

#### 4.1.5 Temporal Graph

We created a temporal graph to analyze research output evolution over time:

```python
def build_year_graph(self) -> None:
    """Build the year graph"""
    self.year_graph = nx.Graph()
    
    # Get all years and sort them
    years = sorted([int(year) for year in self.data['year'].unique() if pd.notna(year)])
    
    # Add nodes
    for year in years:
        self.year_graph.add_node(year, papers=[])
    
    # Add papers to each year
    for _, row in self.data.iterrows():
        year = row['year']
        if pd.notna(year):
            year = int(year)
            if year in self.year_graph:
                self.year_graph.nodes[year]['papers'].append(str(row['paper_id']))
    
    # Add edges - using sliding window to connect adjacent years
    window_size = 3
    # ...
```

### 4.2 Advanced Network Metrics

We computed various network metrics to identify important nodes, communities, and structural patterns:

#### 4.2.1 Centrality Metrics

```python
def _compute_institution_centrality(self) -> Dict:
    """Compute institution centrality metrics"""
    centrality = {}
    
    # Degree centrality
    centrality['degree'] = nx.degree_centrality(self.institution_graph)
    
    # Betweenness centrality
    centrality['betweenness'] = nx.betweenness_centrality(self.institution_graph)
    
    # Closeness centrality
    centrality['closeness'] = nx.closeness_centrality(self.institution_graph)
    
    # Eigenvector centrality
    centrality['eigenvector'] = nx.eigenvector_centrality(self.institution_graph, max_iter=1000)
    
    # Calculate combined centrality score
    combined_centrality = {}
    for node in self.institution_graph.nodes():
        combined_centrality[node] = (
            centrality['degree'][node] * 0.3 +
            centrality['betweenness'][node] * 0.3 +
            centrality['closeness'][node] * 0.2 +
            centrality['eigenvector'][node] * 0.2
        )
    centrality['combined'] = combined_centrality
    
    return centrality
```

#### 4.2.2 Community Detection

We applied multiple community detection algorithms to identify research clusters:

```python
def compute_advanced_metrics(self):
    """Compute advanced network metrics"""
    self.metrics = {}
    
    try:
        # Community detection - using multiple methods
        self.metrics['communities'] = {}
        
        # 1. For institution graph community detection
        if self.institution_graph.number_of_nodes() > 0 and self.institution_graph.number_of_edges() > 0:
            try:
                # Using Louvain method
                import community.community_louvain as community_louvain
                louvain_partition = community_louvain.best_partition(self.institution_graph)
                self.metrics['communities']['institution_louvain'] = louvain_partition
                logging.info(f"Louvain method detected {len(set(louvain_partition.values()))} communities")
    except Exception as e:
                logging.warning(f"Louvain community detection failed: {str(e)}")
            
            try:
                # Using Girvan-Newman method
                communities_iterator = nx.community.girvan_newman(self.institution_graph)
                first_communities = next(communities_iterator)
                self.metrics['communities']['institution_gn'] = first_communities
                logging.info(f"Girvan-Newman method detected {len(first_communities)} communities")
            except Exception as e:
                logging.warning(f"Girvan-Newman community detection failed: {str(e)}")
            
            try:
                # Using Label Propagation method
                lp_communities = list(nx.community.label_propagation_communities(self.institution_graph))
                self.metrics['communities']['institution_lp'] = lp_communities
                logging.info(f"Label Propagation method detected {len(lp_communities)} communities")
            except Exception as e:
                logging.warning(f"Label Propagation community detection failed: {str(e)}")
    # ...
```

#### 4.2.3 Key Collaboration Detection

We identified key collaborations based on edge weight, frequency, and influence:

```python
def _identify_key_collaborations(self) -> Dict:
    """Identify key collaboration networks"""
    key_collaborations = {}
    
    # 1. Key collaborations based on collaboration strength
    edge_weights = [(u, v, d['weight']) for u, v, d in self.institution_graph.edges(data=True)]
    edge_weights.sort(key=lambda x: x[2], reverse=True)
    key_collaborations['by_strength'] = edge_weights[:10]
    
    # 2. Key collaborations based on collaboration frequency
    # ...
    
    # 3. Key collaborations based on influence
    # ...
    
    return key_collaborations
```

### 4.3 Discrete Event Simulation (DES)

We implemented a Discrete Event Simulation model to analyze the dynamic aspects of research output lifecycle:

```python
class ResearchDES:
    def __init__(self, env: simpy.Environment, graph_builder: ResearchGraphBuilder):
        """Initialize the Research DES"""
        self.env = env
        self.graph_builder = graph_builder
        self.review_queue = simpy.Store(env)
        self.publishing_queue = simpy.Store(env)
        self.reviewers = simpy.Resource(env, capacity=5)
        self.publishers = simpy.Resource(env, capacity=3)
        self.stats = {
            'submitted': 0,
            'under_review': 0,
            'published': 0,
            'rejected': 0,
            'pending': 0
        }
    
    def paper_generator(self, rate: float):
        """Generate papers at a given rate"""
        paper_id = 0
        while True:
            # Generate new paper
            paper = {
                'id': paper_id,
                'status': 'submitted',
                'submitted_time': self.env.now
            }
            paper_id += 1
            
            # Update statistics
            self.stats['submitted'] += 1
            
            # Send to review queue
            yield self.env.timeout(random.expovariate(rate))
            yield self.review_queue.put(paper)
```

The simulation allowed us to model key events such as submission, peer review, revision, and publication, revealing processing delays and bottlenecks in the publication workflow.

### 4.4 Key Findings

Our graph analysis revealed several important insights:

1. **Institution Collaboration Patterns**
   - We identified 118 distinct research communities using the Louvain method.
   - Top institutions by centrality measures include the Census Bureau, National Bureau of Economic Research, and several major universities.
   - The average clustering coefficient of 0.42 indicates a moderately dense collaboration network.

2. **Author Collaboration Network**
   - Author networks show a power-law degree distribution (γ = -2.34), indicating a scale-free network structure.
   - Several key researchers serve as bridges between different research communities.
   - High-centrality authors frequently collaborate with multiple institutions.

3. **Temporal Evolution**
   - Research output has grown exponentially since 2000, with a notable acceleration after 2010.
   - The density of the year graph (0.68) indicates strong temporal continuity in research topics.
   - We identified 19 temporal communities representing distinct research eras.

4. **DES Simulation Insights**
   - The average time from submission to publication is 289 days.
   - Review process bottlenecks account for 62% of the delay in publication timeline.
   - Increasing reviewer capacity by 20% would reduce publication time by 18%.

Our graph analysis provides a comprehensive view of the FSRDC research landscape, highlighting key collaborations, influential institutions, and temporal trends. The simulation offers additional insights into the dynamic nature of the research publication process.

## 5. EDA (Data Analysis and Visualization)

This section details the Exploratory Data Analysis (EDA) conducted on the final integrated dataset (`New_And_Original_ResearchOutputs.csv`), fulfilling the requirements outlined in Step 5. We leveraged Pandas for essential data manipulation tasks, including cleaning (handling missing values, type conversions), aggregation, and the exploration of summary statistics. Interactive visualizations were generated using Plotly to investigate data distributions, temporal trends, and collaboration patterns. Furthermore, optional statistical analyses were performed using libraries such as Statsmodels and Scikit-learn to delve deeper into relationships between variables (regression, ANOVA) and to explore dimensionality reduction techniques (PCA).

### 5.1 Overall Dataset Composition

The final dataset encompasses 3,369 research records, aggregated from three distinct origins.

-   **Data Source Distribution:** The contribution from each source is as follows: ResearchOutputs provided 1,735 records (51.5%), the API integration yielded 982 records (29.1%), and web scraping contributed 652 records (19.4%). This highlights the foundational role of the original ResearchOutputs list, significantly augmented by the API and web scraping efforts. **Refer to EDA.ipynb Section 1.1 Summary** for the bar chart visualizing this source breakdown.
-   **Temporal Distribution:** Analysis of publication years, facilitated by Pandas for data cleaning and type conversion, reveals a distinct temporal pattern. Research output remained minimal before 2000, saw a steady increase between 2000-2010, and experienced sharp acceleration from 2010 onwards, peaking around 2020. This indicates a substantial growth in research activity within this domain over the past two decades. **Refer to EDA.ipynb Section 1.1 Summary** for the histogram illustrating the overall year distribution.

### 5.2 Author Collaboration Patterns

Collaboration dynamics were explored by analyzing authorship information.

-   **Author Count Distribution:** A `num_authors` feature was derived using Pandas by parsing the 'authors' field. The resulting distribution shows a strong predominance of single-author (approx. 800 publications) and small-team publications (1-5 authors). The frequency decreases rapidly as team size grows, although the dataset contains rare instances of collaborations exceeding 100 authors. This suggests that while individual and small-team work is most common, the field accommodates diverse collaboration scales. **Refer to EDA.ipynb Section 1.1 Summary** for the histogram visualizing the distribution of authors per article.
-   **Leading Contributors:** Aggregation of author contributions across the dataset identified the most prolific individuals. The top contributors include Bernard (34 publications), Giroud (27), R. (25), Andrew B. (24), Li (24), Xavier (23), Peter K. Schott (23), M. (20), John Haltwanger (20), and J. Bradford Jensen (19). These individuals represent key figures in the research landscape captured by our dataset. **Refer to EDA.ipynb Section 1.1 Summary** for the bar chart detailing the top 10 authors by publication count.

### 5.3 Keyword Analysis (Overall)

Thematic analysis was performed by examining the aggregated 'Keywords' field using Pandas and `collections.Counter`.

-   **Top Keywords:** "Economics" emerges as the most frequent keyword (1,132 mentions), significantly outpacing other prominent terms like "Business" (721), "Computer Science" (662), "Mathematics" (447), and "Political Science" (411). Related fields such as "Law," "Finance," and "Sociology" also appear frequently (around 300-350 times). This distribution underscores the interdisciplinary nature of the research, with a strong core in economics complemented by various related disciplines. **Refer to EDA.ipynb Section 1.5** for the bar chart showing the top 20 keywords.

### 5.4 Institutional Analysis (Overall)

The landscape of contributing organizations was analyzed by processing the `institution_display_names` field.

-   **Top Institutions:** The National Bureau of Economic Research leads significantly with 878 publications, followed by the University of Chicago (185 publications). The top 20 institutions are predominantly U.S.-based research universities and economic research organizations, indicating a concentration of output among major centers but also diversity across multiple institutions. **Refer to EDA.ipynb Section 1.6** for the bar chart listing the top 20 institutions.

### 5.5 Data Completeness Assessment

Understanding data completeness is vital for interpreting results. Missing data patterns were assessed using Pandas.

-   **Missing Value Patterns:** Core fields like 'title', 'authors', and essential metadata generally show high completion rates. However, secondary information fields such as 'acknowledgments', 'detailed affiliations', 'disclosure_review', 'rdc_mentions', and 'dataset_mentions' exhibit more significant gaps, particularly for data not originating from the initial web scraping phase. This suggests good reliability for primary information but highlights areas where supplementary metadata collection could be enhanced. **Refer to EDA.ipynb Section 1.7** for the heatmap visualizing missing data percentages across columns.

### 5.6 Statistical Analysis (Optional)

To fulfill optional requirements and gain deeper insights, statistical analyses were conducted.

#### 5.5.1 Regression and ANOVA on Author Count

We investigated factors potentially influencing research team size using regression and ANOVA (via Statsmodels), after preprocessing data with Pandas (handling NaNs, type conversion, creating 'source_type' and 'keyword_count' features).

-   **Aim:** To understand the relationship between publication year, source type, keyword count, and the number of authors, and to test for differences across sources and temporal trends.
-   **OLS Regression:** A multiple linear regression (`num_authors ~ year + keyword_count + C(source_type)`) yielded a statistically significant model (F=5.477, p<0.001) but with low explanatory power (R²=0.007). The 'year' coefficient was significant (β=0.0417, p<0.001), suggesting a slight increase in average team size over time (approx. one author per 25 years). Source type and keyword count were not significant predictors in this model. Potential multicollinearity (Condition Number: 5.93e+05) and non-normal residuals were noted as limitations.
-   **ANOVA:** A one-way ANOVA (`num_authors ~ C(source_type)`) revealed statistically significant differences in the mean number of authors across the three data sources (F=3.623, p=0.027), indicating that collaboration patterns vary somewhat depending on the data origin when considered alone.
-   **Linear Trend Analysis:** Using `scipy.optimize.curve_fit`, a positive linear trend in team size over time was confirmed (slope = 0.039 ± 0.011).
-   **Findings:** There is evidence for increasing team sizes over time, consistent across sources. Modest but significant differences in collaboration patterns exist between data sources. Research scope (proxied by keyword count) does not appear strongly related to team size.
-   **Refer to EDA.ipynb Section 1.8** for the detailed statistical outputs, model summaries, limitations, and implementation details.

#### 5.5.2 Principal Component Analysis (PCA) on Abstracts and Metadata

PCA (via Scikit-learn) was applied to explore patterns in the data based on textual content and metadata, aiming for dimensionality reduction and visualization. TF-IDF features from abstracts (`TfidfVectorizer`, max_features=300) were combined with `num_authors` and `keyword_count`.

-   **Aim:** To reduce dimensionality and visually explore patterns across data sources using combined textual and numeric features.
-   **Methodology:** TF-IDF matrix from abstracts was combined with numeric features, and PCA was applied to reduce to 2 components.
-   **Findings:** The 2D PCA scatter plot revealed distinct visual clustering by source type. WebScrap articles formed tighter clusters, suggesting content homogeneity. API-sourced articles were more dispersed, implying greater content diversity. ResearchOutputs data showed characteristics overlapping both. Several outliers were identified.
-   **Implications:** The patterns suggest systematic content differences between sources and validate the use of multiple sources for comprehensive coverage. Outliers may represent novel research.
-   **Limitations:** PCA is linear; information is lost in dimensionality reduction. TF-IDF doesn't capture semantics. Missing abstract data (over 50%) limits generalizability.
-   **Refer to EDA.ipynb Section 1.9** for the PCA scatter plot, methodology details, and further discussion.

### 5.7 Source-Specific Keyword and Metadata Analysis

To further characterize the data sources, keyword themes and specific metadata fields were analyzed individually for the WebScrap, ResearchOutputs, and API subsets.

#### 5.7.1 WebScrap Dataset (652 Records)

-   **Keywords:** The word cloud reveals prominent topics including Computer Science, Economics, Business, Mathematics, AI, Data Science, and Social Sciences, suggesting a balanced mix of technical, economic, and social science research, possibly reflecting a more exploratory collection.
-   **Metadata Presence:** Analysis of boolean metadata fields (`acknowledgments`, `data_descriptions`, `disclosure_review`, `rdc_mentions`, `dataset_mentions`) showed `data_descriptions` was frequently marked True (635 times), while others, especially `rdc_mentions` (0 True), were sparsely populated. This indicates richness in general data context but limited specific tracking information within this subset.
-   **Refer to EDA.ipynb Section 2 (WebScrapping Data)** for the word cloud and metadata bar plot.

#### 5.7.2 ResearchOutputs Dataset (1,735 Records)

-   **Keywords:** The word cloud reflects a broad academic scope centered on Economics, Business, Computer Science, and Political Science, including subfields like Industrial Organization and Microeconomics. This aligns with its origin as a structured list of academic outputs.
-   **Refer to EDA.ipynb Section 3(ResearchOutputs)** for the word cloud.

#### 5.7.3 API Dataset (982 Records)

-   **Keywords:** Dominant keywords include Economics, Business, Computer Science, Finance, Labour Economics, and Political Science. The emphasis seems slightly more weighted towards economics and applied social sciences compared to other sources, possibly reflecting the API's focus or underlying database taxonomy.
-   **Refer to EDA.ipynb Section 4 (API)** for the word cloud.

#### 5.7.4 Summary Across Sources

While core academic themes overlap across all sources, subtle differences in focus emerge: WebScrap appears more mixed, ResearchOutputs more formally structured, and API more concentrated on economic/policy themes. This analysis validates the complementary nature of the sources in capturing diverse facets of FSRDC-related research.









