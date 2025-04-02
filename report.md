## 1. Data Collection and Processing

### 1.3 Citation Network Analysis via API Integration

Our approach to discovering new FSRDC research outputs builds on Project 1's dataset by analyzing citation networks. Starting with **Donald Moratz's cleaned bibliography (cleaned_biblio.csv)** which is the cleaned verion of ResearchOutputs.xlsx provided , we trace forward citations to identify papers that cite known FSRDC research. This approach is based on the reasoning that:

1. Papers citing FSRDC research are likely to:
   - Use similar restricted datasets
   - Build upon FSRDC-based findings
   - Include authors familiar with FSRDC processes

2. By analyzing the entire citation network, we can:
   - Discover research outputs not captured in the 2024 dataset
   - Identify emerging research trends
   - Map the influence of FSRDC research

#### 1.3.1 API Integration Strategy

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

#### 1.3.2 Keyword Selection Strategy

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

#### 1.3.3 Data Collection Process

The `api_integration.py` script is designed to collect and process research paper data using the OpenAlex API. Here's a detailed breakdown of its functionality:

#### 1.3.1 Core Functions

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

#### 1.3.2 Main Processing Function

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

#### 1.3.3 Example Output

When running the script:


```python
python api_integration.py
```


The script produces detailed progress information:

Below is an example of the terminal output when running the API data collection script, **since there are too many output, here is just a illustration:**

```text
Loaded 150 records from cleaned_biblio.csv
Sorted data by year (column: year)
Processing all 150 papers
Processing papers: 100%|██████████| 150/150 [02:35<00:00, 1.04s/paper]

Processing paper 1/150: The Economic Impacts of Climate Change: Evidence from Agricultural Output and Random Fluctuations in Weather
  Found 45 papers citing this work
  Retrieving details for citing paper: Agricultural Productivity and Climate Change: A Regional Panel Analysis
Authors found: 3
Raw authors found: 3
Institutions found: 2
Raw affiliations found: 2
Detailed affiliations found: 2
  Found FSRDC-related paper: Agricultural Productivity and Climate Change: A Regional Panel Analysis
  Matching keywords: census of agriculture, restricted data, confidential microdata

Processing paper 2/150: Innovation and Production Networks: Evidence from Manufacturing Firms
  Found 32 papers citing this work
  Retrieving details for citing paper: Network Effects in Manufacturing: Evidence from Census Microdata
Authors found: 4
Raw authors found: 4
Institutions found: 3
Raw affiliations found: 3
Detailed affiliations found: 3
  Found FSRDC-related paper: Network Effects in Manufacturing: Evidence from Census Microdata
  Matching keywords: census bureau, restricted microdata, census of manufacturing

[... similar output for remaining papers ...]

Processing paper 150/150: Trade Liberalization and Labor Market Dynamics
  Found 28 papers citing this work
  Retrieving details for citing paper: Import Competition and Employment Adjustment
Authors found: 2
Raw authors found: 2
Institutions found: 2
Raw affiliations found: 2
Detailed affiliations found: 2
  Found FSRDC-related paper: Import Competition and Employment Adjustment
  Matching keywords: census bureau, restricted data, census of manufacturing

Process completed. Found 187 FSRDC-related papers.
Results saved to fsrdc5_related_papers_api_all.csv
```
**Notice that there are actually 3642 records after running**

#### 1.3.4 Results and Transition to Cleaning

The initial API collection process identified 3,642 potentially relevant papers, demonstrating the extensive reach of FSRDC research. However, this raw dataset requires careful cleaning and validation to ensure:
1. Uniqueness (no overlap with 2024 dataset)
2. Strong FSRDC relevance
3. Quality metadata

This leads to our next phase of data processing, detailed in section 1.4.

### 1.4 Data Cleaning and Deduplication Process

The `clean_dedup_process_api_data.py` script implements a multi-stage cleaning pipeline to ensure data quality and relevance. Here's a detailed breakdown of each stage:

#### 1.4.1 Initial Data Processing

```python
# 1. Load original data
df = pd.read_csv("fsrdc5_related_papers_api_all.csv")

# 2. Remove duplicates based on title
deduplicate_self = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
deduplicate_self.to_csv("deduplicate_self.csv", index=False)

# 3. Read cleaned_biblio.csv for cross-reference
cleaned_biblio = pd.read_csv("cleaned_biblio.csv")
```

The script starts by loading both the API-collected data and the original bibliography for cross-referencing.

#### 1.4.2 Fuzzy Matching Implementation

To handle variations in paper titles, we implement fuzzy string matching:

```python
def is_similar(title1, title2, threshold=80):
    """
    Compare similarity between two titles, return True if similarity exceeds threshold
    """
    if pd.isna(title1) or pd.isna(title2):
        return False
    return fuzz.ratio(str(title1).lower(), str(title2).lower()) >= threshold
```

This function:
- Converts titles to lowercase for comparison
- Handles missing values
- Uses Levenshtein distance for similarity scoring
- Returns True if similarity exceeds 80%

#### 1.4.3 Cross-Reference Deduplication

```python
# Create a list to mark rows to keep
keep_rows = []

# Check each title in deduplicate_self
for idx, row in deduplicate_self.iterrows():
    keep = True
    current_title = row["title"]

    # Compare with each OutputTitle in cleaned_biblio
    for biblio_title in cleaned_biblio["OutputTitle"]:
        if is_similar(current_title, biblio_title):
            keep = False
            break

    keep_rows.append(keep)

# Filter data using the marked list
after_fuzzy_df = deduplicate_self[keep_rows].reset_index(drop=True)
```

This process:
- Compares each paper with the original bibliography
- Marks papers for keeping or removal
- Removes papers that closely match existing records

#### 1.4.4 Keyword Count Analysis

```python
def count_keywords(keywords_str):
    """
    Count number of keywords by splitting on ', '
    """
    if pd.isna(keywords_str):
        return 0
    return len(str(keywords_str).split(", "))

# Filter records with 2 or more keywords
after_fuzzy_df_larger2 = after_fuzzy_df[
    after_fuzzy_df["match_rdc_criteria_keywords"].apply(count_keywords) >= 2
].reset_index(drop=True)

# Save first final result
after_fuzzy_df_larger2.to_csv("final_deduped_data.csv", index=False)
```

This step:
- Counts FSRDC-related keywords in each paper
- Filters for papers with 2+ keyword matches
- Ensures stronger relevance to FSRDC research

#### 1.4.5 OpenAlex Keyword Enhancement

```python
def fetch_openalex_data_by_title(title):
    """
    Get data from OpenAlex using paper title
    """
    url = f"https://api.openalex.org/works?search={title}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if results:
            return results[0]
        else:
            return None
    except Exception as e:
        print(f"Error searching OpenAlex for title '{title}': {e}")
        return None

def get_openalex_keywords(work):
    """
    Extract keywords from OpenAlex work object
    """
    if not work:
        return "No keywords found"

    concepts = work.get("concepts", [])
    if concepts:
        return ", ".join([concept.get("display_name", "") for concept in concepts])
    return "No keywords found"
```

The final enhancement:
- Queries OpenAlex API for each paper
- Extracts subject keywords and concepts
- Adds this information to enrich the dataset
- Implements rate limiting for API compliance

#### 1.4.6 Results Summary

The cleaning process produced these results:

```text
Original merged data count: 3642
Count after first self-deduplication: 3068
Count after fuzzy match deduplication with original data: 2840
Count after FSRDC keyword filtering (>=2): 982
```

Each step's impact:
1. **Self-Deduplication**: Removed 574 exact duplicate titles
2. **Fuzzy Matching**: Eliminated 228 similar titles from original bibliography
3. **Keyword Filtering**: Refined to 982 highly relevant papers

Final dataset characteristics:
- 982 unique FSRDC-related papers
- Each paper has 2+ FSRDC relevance indicators
- Enhanced with OpenAlex subject classifications
- No duplicates with original bibliography
- Complete metadata for further analysis

The resulting dataset (`final_deduped_data_withkeyword.csv`) provides a clean, deduplicated, and enriched collection of FSRDC-related research papers for subsequent analysis from API.










