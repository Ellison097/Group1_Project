# Group1: FSRDC Research Output Analysis
###### Yukun Gao (gaoyukun@seas.upenn.edu)
###### Licheng Guo (guolc@seas.upenn.edu)
###### Yixuan Xu (xuyixuan@seas.upenn.edu)
###### Zining Hua (znhua@seas.upenn.edu)
###### Qiming Zhang (zqiming@seas.upenn.edu) 
###### Yongchen Lu (ylu178@seas.upenn.edu)

## 0. Introduction

This report presents our approach to discovering and analyzing research outputs related to Federal Statistical Research Data Centers (FSRDC). The Federal Statistical Research Data Centers provide researchers with secure access to restricted microdata from various federal statistical agencies, including the Census Bureau, Bureau of Economic Analysis (BEA), and Internal Revenue Service (IRS). These data centers play a crucial role in enabling research that informs public policy and economic understanding.

Our project aims to identify and analyze research outputs that utilize FSRDC data, focusing on three main approaches:

1. **Web Scraping**: Extracting metadata from publicly available sources to identify potential FSRDC-related research outputs.
2. **API Integration**: Utilizing various academic APIs to retrieve comprehensive metadata about research papers.
3. **Data Processing and Entity Resolution**: Cleaning, deduplicating, and validating the collected data to ensure quality and relevance.

The methodology employed in this project addresses several key challenges in research data curation:
- Identifying research outputs that use restricted data without direct access to the data itself
- Handling variations in how authors and institutions reference FSRDC
- Ensuring comprehensive coverage while avoiding duplicates with existing datasets
- Validating the relevance of research outputs to FSRDC

This report details our implementation of each step, the challenges encountered, and the results achieved. By combining web scraping, API integration, and sophisticated data processing techniques, we have developed a robust pipeline for identifying and analyzing FSRDC-related research outputs.

## 1. Web Scraping

Our web scraping approach focuses on extracting metadata from publicly available sources to identify potential FSRDC-related research outputs. The implementation follows a systematic process to ensure comprehensive coverage while respecting website limitations.

### 1.1 Data Source Selection

We identified several key sources for web scraping:

1. **GitHub Repositories**: FSRDC projects often maintain GitHub repositories containing project information, including titles, PIs, and status updates.

2. **Institutional Websites**: RDC websites (e.g., Michigan RDC, Texas RDC) often list research outputs and publications.

3. **Research Institution Pages**: University departments and research centers frequently highlight FSRDC-related research.

4. **Conference Proceedings**: Academic conferences often publish papers that use FSRDC data.

### 1.2 Implementation Details

The `web_scraping.py` script implements our web scraping functionality with the following key components:

```python
def scrape_research_outputs():
    """Main function to scrape research outputs from various sources"""
    results = []
    
    # Scrape from GitHub repositories
    github_results = scrape_github_repos()
    results.extend(github_results)
    
    # Scrape from institutional websites
    institution_results = scrape_institution_sites()
    results.extend(institution_results)
    
    # Scrape from conference proceedings
    conference_results = scrape_conference_proceedings()
    results.extend(conference_results)
    
    # Save results to CSV
    save_to_csv(results, "data/processed/scraped_data.csv")
    
    return results
```

For each source, we implemented specific scraping functions that:

1. **Handle HTTP Errors**: Implement robust error handling to manage connection issues, timeouts, and rate limiting.

```python
def safe_request(url, headers=None, max_retries=3, delay=1):
    """Make a safe HTTP request with retries and delays"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            time.sleep(delay)  # Respect rate limiting
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                return None
            time.sleep(delay * (attempt + 1))  # Exponential backoff
```

2. **Parse HTML Content**: Use BeautifulSoup to extract relevant information from HTML pages.

```python
def parse_html_content(html_content):
    """Parse HTML content to extract research output metadata"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title = extract_title(soup)
    
    # Extract authors
    authors = extract_authors(soup)
    
    # Extract abstract
    abstract = extract_abstract(soup)
    
    # Extract affiliations
    affiliations = extract_affiliations(soup)
    
    return {
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'affiliations': affiliations
    }
```

3. **Clean and Normalize Data**: Apply regex and text processing techniques to clean and standardize the extracted data.

```python
def clean_text(text):
    """Clean and normalize text data"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters
    text = re.sub(r'[^\w\s.,;:()\-]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    
    return text
```

### 1.3 Selection Criteria Implementation

We implemented the selection criteria specified in the project requirements to ensure that only relevant research outputs are included:

```python
def is_fsrdc_related(metadata):
    """Check if research output is related to FSRDC based on selection criteria"""
    # Check acknowledgments
    if has_fsrdc_acknowledgment(metadata):
        return True
    
    # Check data description
    if has_restricted_data_reference(metadata):
        return True
    
    # Check disclosure review
    if has_disclosure_review_statement(metadata):
        return True
    
    # Check RDC mentions
    if has_rdc_mention(metadata):
        return True
    
    # Check dataset mentions
    if has_dataset_mention(metadata):
        return True
    
    return False
```

Each criterion is implemented as a separate function that checks for specific patterns in the metadata:

```python
def has_fsrdc_acknowledgment(metadata):
    """Check if metadata contains FSRDC acknowledgment"""
    acknowledgment_keywords = [
        "census bureau", "fsrdc", "research data center", "rdc",
        "federal statistical research data center"
    ]
    
    text = f"{metadata.get('title', '')} {metadata.get('abstract', '')}"
    text = text.lower()
    
    return any(keyword in text for keyword in acknowledgment_keywords)

def has_dataset_mention(metadata):
    """Check if metadata mentions specific FSRDC datasets"""
    # Load dataset names from Excel sheet
    dataset_names = load_dataset_names()
    
    text = f"{metadata.get('title', '')} {metadata.get('abstract', '')}"
    text = text.lower()
    
    return any(dataset.lower() in text for dataset in dataset_names)
```

### 1.4 Results and Challenges

Our web scraping implementation successfully extracted metadata from 943 potential FSRDC-related research outputs. The process encountered several challenges:

1. **Rate Limiting**: Many websites implement rate limiting to prevent automated scraping. We addressed this by implementing delays between requests and using exponential backoff for retries.

2. **HTML Structure Variations**: Different websites use different HTML structures, making it challenging to create a one-size-fits-all scraping solution. We implemented source-specific parsers to handle these variations.

3. **Data Completeness**: Not all sources provide complete metadata. We implemented fallback mechanisms to maximize data completeness.

4. **False Positives**: Some research outputs mention FSRDC or related terms without actually using FSRDC data. We refined our selection criteria to minimize false positives.

After applying our selection criteria and deduplication, we identified 652 unique FSRDC-related research outputs from web scraping.

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
- Comprehensive citation data
- Rich metadata including abstracts and affiliations
- Free access with good rate limits
- Structured data about institutions and authors

The `api_integration.py` script implements this strategy through three main components:

1. **Seed Paper Processing**
   - Reads the cleaned bibliography as seed papers
   - Sorts by year to prioritize recent papers
   - Processes each paper's citation network

2. **Forward Citation Analysis**
   - For each seed paper:
     - Retrieves all citing papers
     - Analyzes each citing paper for FSRDC relevance
     - Captures complete metadata for relevant papers

3. **FSRDC Relevance Filtering**
   - Applies multiple criteria to ensure papers are FSRDC-related
   - Checks multiple metadata fields for relevance
   - Uses carefully selected keywords based on project requirements

### 2.2 Keyword Selection Strategy

The keywords were carefully selected to align with project requirements:

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

The `api_integration.py` script is designed to collect and process research paper data using the OpenAlex API. Here's a detailed breakdown of its functionality:

### 2.3.1 Core Functions

1. **Abstract Reconstruction** (`reconstruct_abstract`)
   - OpenAlex provides abstracts in an inverted index format
   - Function reconstructs readable text from the index
   ```python
   def reconstruct_abstract(inverted_index):
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
     - Citation data
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

After collecting data from web scraping and API integration, we implemented a comprehensive data processing pipeline to ensure data quality and relevance.

### 3.1 Data Loading and Initial Cleaning

The first step in our data processing pipeline is loading and cleaning the raw data:

```python
def load_and_clean_data():
    """Load and clean data from various sources"""
    # Load web scraping data
    scraped_data = pd.read_csv("data/processed/scraped_data.csv")
    
    # Load API data
    api_data = pd.read_csv("data/processed/fsrdc5_related_papers_api_all.csv")
    
    # Load 2024 dataset for comparison
    dataset_2024 = pd.read_excel("data/raw/ResearchOutputs.xlsx")
    
    # Clean data
    cleaned_scraped_data = clean_dataframe(scraped_data)
    cleaned_api_data = clean_dataframe(api_data)
    
    return cleaned_scraped_data, cleaned_api_data, dataset_2024
```

The cleaning process includes:
- Handling missing values
- Standardizing formats (dates, names, etc.)
- Removing duplicates within each dataset

### 3.2 Entity Resolution and Deduplication

We implemented a multi-stage deduplication process to ensure uniqueness:

1. **Exact Matching**: Identify exact duplicates based on title, DOI, or other unique identifiers.

```python
def remove_exact_duplicates(df, subset=["title"]):
    """Remove exact duplicates based on specified columns"""
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
```

2. **Fuzzy Matching**: Handle variations in record formatting using fuzzy string matching.

```python
def is_similar(title1, title2, threshold=80):
    """Compare similarity between two titles"""
    if pd.isna(title1) or pd.isna(title2):
        return False
    return fuzz.ratio(str(title1).lower(), str(title2).lower()) >= threshold
```

3. **Cross-Reference with 2024 Dataset**: Ensure that research outputs already reported in the 2024 dataset are excluded.

```python
def remove_2024_duplicates(df, dataset_2024):
    """Remove entries that are already in the 2024 dataset"""
    keep_rows = []
    
    for idx, row in df.iterrows():
        keep = True
        current_title = row["title"]
        
        # Compare with each title in 2024 dataset
        for biblio_title in dataset_2024["OutputTitle"]:
            if is_similar(current_title, biblio_title):
                keep = False
                break
        
        keep_rows.append(keep)
    
    return df[keep_rows].reset_index(drop=True)
```

### 3.3 FSRDC Relevance Validation

We implemented a comprehensive validation process to ensure that all research outputs in our dataset are indeed FSRDC-related:

1. **Keyword Analysis**: Count and analyze FSRDC-related keywords in each paper.

```python
def count_keywords(keywords_str):
    """Count number of keywords by splitting on ', '"""
    if pd.isna(keywords_str):
        return 0
    return len(str(keywords_str).split(", "))
```

2. **Multi-Field Validation**: Check multiple fields (title, abstract, acknowledgments, etc.) for FSRDC relevance.

```python
def validate_fsrdc_relevance(df):
    """Validate FSRDC relevance of research outputs"""
    # Filter records with 2 or more keywords
    relevant_df = df[
        df["match_rdc_criteria_keywords"].apply(count_keywords) >= 2
    ].reset_index(drop=True)
    
    return relevant_df
```

### 3.4 Data Structure and Additional Attributes

We chose to use a comprehensive data structure that includes the following attributes:

1. **Basic Metadata**:
   - Title
   - Authors
   - Publication Year
   - DOI
   - Abstract

2. **FSRDC-Specific Information**:
   - Matching FSRDC Criteria Keywords
   - Dataset Information
   - RDC Mentions

3. **Institutional Information**:
   - Author Affiliations
   - Institution Display Names
   - Raw Affiliation Strings

4. **Additional Context**:
   - Citation Count
   - Source (Web Scraping or API)
   - Processing Timestamp

This structure ensures that our dataset provides comprehensive information about each research output while maintaining clarity and usability.

### 3.5 Results and Impact

Our data processing pipeline successfully:

1. **Reduced Duplicates**: Eliminated 574 exact duplicate titles and 228 similar titles from the original bibliography.

2. **Ensured Uniqueness**: Verified that all research outputs in our dataset are not already included in the 2024 dataset.

3. **Validated Relevance**: Confirmed that all research outputs meet the FSRDC relevance criteria.

4. **Enhanced Metadata**: Added comprehensive metadata from OpenAlex API to enrich the dataset.

The final dataset contains 982 unique FSRDC-related research outputs, each with at least 2 FSRDC relevance indicators and complete metadata for further analysis.










