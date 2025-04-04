import pandas as pd
import numpy as np
from thefuzz import fuzz
from typing import List, Dict, Any
import logging
import re
from datetime import datetime
import os
import json
import time
import requests
from tqdm import tqdm
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.threshold = 85  # Fuzzy matching threshold
        
        # Load existing research outputs
        self.existing_outputs = self._load_existing_outputs()

    def _load_existing_outputs(self) -> pd.DataFrame:
        """Load existing research outputs data"""
        try:
            return pd.read_excel('data/raw/ResearchOutputs.xlsx')
        except Exception as e:
            logger.error(f"Error loading existing outputs: {e}")
            return pd.DataFrame()

    def _clean_text(self, text: str) -> str:
        """Clean text data"""
        if pd.isna(text):
            return ""
        # Convert to lowercase
        text = str(text).lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    def _check_uniqueness(self, title: str) -> bool:
        """Check if research output exists in 2024 dataset"""
        if self.existing_outputs.empty:
            return True
        
        cleaned_title = self._clean_text(title)
        for _, row in self.existing_outputs.iterrows():
            existing_title = self._clean_text(row['OutputTitle'])
            # Use fuzzy matching
            similarity = fuzz.ratio(cleaned_title, existing_title)
            if similarity >= self.threshold:
                return False
        return True

    def _validate_fsrdc_criteria(self, row: pd.Series) -> bool:
        """Validate FSRDC criteria"""
        # Check if any criteria is met
        criteria_columns = [
            'acknowledgments',
            'data_descriptions',
            'disclosure_review',
            'rdc_mentions',
            'dataset_mentions'
        ]
        
        return any(row[col] for col in criteria_columns if col in row)

    def _merge_data(self, scraped_data: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
        """Merge scraped data and API data"""
        # Merge based on title
        merged_data = pd.merge(
            scraped_data,
            api_data,
            on='title',
            how='outer',
            suffixes=('_scraped', '_api')
        )
        return merged_data

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data"""
        # Remove duplicate rows
        df = df.drop_duplicates(subset=['title'])
        
        # Clean text columns
        text_columns = ['title', 'abstract', 'authors']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_text)
        
        # Handle missing values
        df = df.fillna({
            'abstract': '',
            'authors': '[]',
            'year': 0
        })
        
        return df

    def process(self, scraped_data: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
        """Process data and generate final dataset"""
        # Merge data
        merged_data = self._merge_data(scraped_data, api_data)
        
        # Clean data
        cleaned_data = self._clean_data(merged_data)
        
        # Validate uniqueness and FSRDC criteria
        final_data = []
        for _, row in cleaned_data.iterrows():
            if self._check_uniqueness(row['title']) and self._validate_fsrdc_criteria(row):
                final_data.append(row)
        
        # Convert to DataFrame
        final_df = pd.DataFrame(final_data)
        
        # Save processed data
        final_df.to_csv('data/processed/final_research_outputs.csv', index=False)
        
        # Output statistics
        logger.info(f"Total records processed: {len(cleaned_data)}")
        logger.info(f"Unique records after processing: {len(final_df)}")
        
        return final_df

    def _is_duplicate(self, title: str, authors: List[str]) -> bool:
        """Check if research output is duplicate"""
        if self.existing_outputs.empty:
            return False
            
        # Use fuzzy matching to check title
        title_similarities = [fuzz.ratio(title.lower(), t.lower()) 
                            for t in self.existing_outputs['Title']]
        
        # If title similarity exceeds 85%, consider as duplicate
        if max(title_similarities) > 85:
            return True
            
        # Check authors
        for _, row in self.existing_outputs.iterrows():
            existing_authors = str(row['Authors']).lower().split(',')
            if any(author.lower() in existing_authors for author in authors):
                return True
                
        return False

    def _standardize_authors(self, authors: List[str]) -> str:
        """Standardize author list"""
        if not authors:
            return ""
        return ", ".join([self._clean_text(author) for author in authors])

    def _extract_year(self, date_str: str) -> str:
        """Extract year from date string"""
        if pd.isna(date_str):
            return ""
        try:
            return str(pd.to_datetime(date_str).year)
        except:
            return ""

    def process_data(self, scraped_data: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
        """Process scraped and API data"""
        try:
            # Merge data
            all_data = pd.concat([scraped_data, api_data], ignore_index=True)
            
            # Clean data
            all_data['Title'] = all_data['Title'].apply(self._clean_text)
            all_data['Authors'] = all_data['Authors'].apply(self._standardize_authors)
            all_data['Abstract'] = all_data['Abstract'].apply(self._clean_text)
            all_data['Year'] = all_data['Year'].apply(self._extract_year)
            
            # Remove duplicates
            unique_outputs = []
            for _, row in all_data.iterrows():
                if not self._is_duplicate(row['Title'], row['Authors'].split(', ')):
                    unique_outputs.append(row)
                    
            # Convert to DataFrame
            processed_df = pd.DataFrame(unique_outputs)
            
            # Add processing timestamp
            processed_df['Processed_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save processed data
            output_file = 'data/processed/processed_research_outputs.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            processed_df.to_csv(output_file, index=False)
            
            logger.info(f"Successfully processed {len(processed_df)} unique research outputs")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return pd.DataFrame()

    def generate_summary(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data summary"""
        if processed_df.empty:
            return {}
            
        summary = {
            'total_outputs': len(processed_df),
            'unique_authors': len(processed_df['Authors'].unique()),
            'year_distribution': processed_df['Year'].value_counts().to_dict(),
            'source_distribution': processed_df['Source'].value_counts().to_dict(),
            'fsrdc_compliant': processed_df['fsrdc_compliant'].sum() if 'fsrdc_compliant' in processed_df.columns else 0
        }
        
        # Save summary
        summary_file = 'data/processed/research_outputs_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        return summary

def process_api_data():
    """Process API data"""
    try:
        logger.info("Starting API data processing...")
        
        # 1. Read original API data
        df = pd.read_csv("data/processed/fsrdc5_related_papers_api_all.csv")
        logger.info(f"Original API data count: {len(df)}")
        
        # 2. Deduplicate based on title
        deduplicate_self = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
        logger.info(f"Data count after title deduplication: {len(deduplicate_self)}")
        
        # Save first deduplication result
        deduplicate_self.to_csv("data/processed/deduplicate_self.csv", index=False)
        
        # 3. Read cleaned_biblio.csv
        cleaned_biblio = pd.read_csv("data/raw/cleaned_biblio.csv")
        logger.info(f"Read cleaned_biblio.csv, total {len(cleaned_biblio)} records")
        
        # 4. Use fuzzy matching for deduplication
        def is_similar(title1, title2, threshold=80):
            """Compare two titles for similarity, return True if similarity exceeds threshold"""
            if pd.isna(title1) or pd.isna(title2):
                return False
            return fuzz.ratio(str(title1).lower(), str(title2).lower()) >= threshold
        
        # Create mark list
        keep_rows = []
        
        # Check each title
        for idx, row in deduplicate_self.iterrows():
            # Default to keep this row
            keep = True
            current_title = row["title"]
            
            # Compare with each title in cleaned_biblio
            for biblio_title in cleaned_biblio["OutputTitle"]:
                if is_similar(current_title, biblio_title):
                    # If similar title found, mark as not to keep
                    keep = False
                    break
            
            keep_rows.append(keep)
        
        # Filter data using mark list
        after_fuzzy_df = deduplicate_self[keep_rows].reset_index(drop=True)
        logger.info(f"Data count after fuzzy matching deduplication: {len(after_fuzzy_df)}")
        
        # 5. Filter records containing 2 or more FSRDC keywords
        def count_keywords(keywords_str):
            """Calculate keyword count"""
            if pd.isna(keywords_str):
                return 0
            return len(str(keywords_str).split(", "))
        
        # Filter records
        after_fuzzy_df_larger2 = after_fuzzy_df[
            after_fuzzy_df["match_rdc_criteria_keywords"].apply(count_keywords) >= 2
        ].reset_index(drop=True)
        logger.info(f"Data count after keyword filtering: {len(after_fuzzy_df_larger2)}")
        
        # Save first final result (without OpenAlex keywords)
        after_fuzzy_df_larger2.to_csv("data/processed/final_deduped_data.csv", index=False)
        
        # 6. Get keywords from OpenAlex API
        def fetch_openalex_data_by_title(title):
            """Get data from OpenAlex"""
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
                logger.error(f"Error searching OpenAlex for '{title}': {e}")
                return None
        
        def get_openalex_keywords(work):
            """Extract keywords from OpenAlex work object"""
            if not work:
                return "No keywords found"
            
            concepts = work.get("concepts", [])
            if concepts:
                return ", ".join([concept.get("display_name", "") for concept in concepts])
            return "No keywords found"
        
        # Add OpenAlex keywords
        openalex_keywords = []
        logger.info("Starting to get keywords from OpenAlex...")
        
        for idx, row in after_fuzzy_df_larger2.iterrows():
            logger.info(f"Processing record {idx+1}/{len(after_fuzzy_df_larger2)}")
            work = fetch_openalex_data_by_title(row["title"])
            keywords = get_openalex_keywords(work)
            openalex_keywords.append(keywords)
            time.sleep(0.12)  # Avoid request overload
        
        after_fuzzy_df_larger2["Keywords"] = openalex_keywords
        
        # Save final result (with OpenAlex keywords)
        after_fuzzy_df_larger2.to_csv("data/processed/final_deduped_data_withkeyword.csv", index=False)
        
        # Output processing result statistics
        logger.info(f"Original merged data count: {len(df)}")
        logger.info(f"Data count after first deduplication: {len(deduplicate_self)}")
        logger.info(f"Data count after fuzzy matching deduplication: {len(after_fuzzy_df)}")
        logger.info(f"Data count after keyword filtering: {len(after_fuzzy_df_larger2)}")
        logger.info("OpenAlex keywords added")
        
        return after_fuzzy_df_larger2
        
    except Exception as e:
        logger.error(f"Error in API data processing: {str(e)}")
        raise

def check_duplicates_with_research_outputs(scraped_data: pd.DataFrame, research_outputs: pd.DataFrame) -> pd.DataFrame:
    """
    Check if scraped data is duplicate with ResearchOutputs.xlsx data, using exact matching and fuzzy matching
    
    Args:
        scraped_data: Data from web_scraping.py
        research_outputs: Data from ResearchOutputs.xlsx
    
    Returns:
        Deduplicated DataFrame
    """
    logger.info("Starting to check duplicate data...")
    
    # Ensure both DataFrames have title column
    if 'title' not in scraped_data.columns:
        logger.error("scraped_data has no title column")
        return scraped_data
    
    if 'OutputTitle' not in research_outputs.columns:
        logger.error("research_outputs has no OutputTitle column")
        return scraped_data
    
    # 1. First perform exact deduplication based on title
    scraped_data = scraped_data.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    logger.info(f"Data count after exact deduplication: {len(scraped_data)}")
    
    # 2. Use fuzzy matching for deduplication
    def is_similar(title1, title2, threshold=80):
        """Compare two titles for similarity, return True if similarity exceeds threshold"""
        if pd.isna(title1) or pd.isna(title2):
            return False
        return fuzz.ratio(str(title1).lower(), str(title2).lower()) >= threshold
    
    # Create mark list
    keep_rows = []
    
    # Check each title
    for idx, row in scraped_data.iterrows():
        # Default to keep this row
        keep = True
        current_title = row["title"]
        
        # Compare with each title in research_outputs
        for biblio_title in research_outputs["OutputTitle"]:
            if is_similar(current_title, biblio_title):
                # If similar title found, mark as not to keep
                keep = False
                break
        
        keep_rows.append(keep)
    
    # Filter data using mark list
    deduplicated_data = scraped_data[keep_rows].reset_index(drop=True)
    
    # Record deduplication result
    logger.info(f"Original data count: {len(scraped_data)}")
    logger.info(f"Data count after fuzzy matching deduplication: {len(deduplicated_data)}")
    
    # Save duplicate data to separate file
    duplicate_data = scraped_data[~scraped_data.index.isin(deduplicated_data.index)]
    if not duplicate_data.empty:
        output_file = 'data/processed/duplicate_data.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        duplicate_data.to_csv(output_file, index=False)
        logger.info(f"Duplicate data saved to {output_file}")
    
    return deduplicated_data

def get_paper_metadata(title_query: str, sleep_time: float = 0.15):
    """Get paper metadata based on paper title"""
    # Check if title_query is None or empty
    if not title_query or not isinstance(title_query, str):
        print(f"Invalid title query: {title_query}")
        return None

    # Implement rate limiting
    time.sleep(sleep_time)

    # Build search URL
    url = f"https://api.openalex.org/works?search={title_query.replace(' ', '%20')}"
    headers = {"User-Agent": "YourProject (your.email@domain.com)"}

    try:
        # Send GET request to OpenAlex API
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Check if results found
        if not data.get("results"):
            print(f"No results found for: {title_query}")
            return None

        # Use first search result
        work = data["results"][0]

        # Extract basic metadata and keywords
        metadata = {
            "doi": work.get("doi", "N/A"),
        }

        # Get keywords from concepts
        concepts = work.get("concepts", [])
        if concepts:
            metadata["keywords"] = ", ".join([concept.get("display_name", "") for concept in concepts])
        else:
            metadata["keywords"] = "No keywords found"

        # Extract institution information
        institution_names = set()
        raw_affiliations = set()
        detailed_affiliations = set()

        for authorship in work.get("authorships", []):
            # Institutions
            for inst in authorship.get("institutions", []):
                inst_name = inst.get("display_name")
                if inst_name:
                    institution_names.add(inst_name)

            # Raw affiliations
            raw_aff_strings = authorship.get("raw_affiliation_strings", [])
            raw_affiliations.update(raw_aff_strings)

            # Detailed affiliations
            for aff in authorship.get("affiliations", []):
                aff_string = aff.get("raw_affiliation_string")
                if aff_string:
                    detailed_affiliations.add(aff_string)

        metadata.update(
            {
                "institution_display_names": "; ".join(institution_names) if institution_names else "",
                "raw_affiliation_strings": "; ".join(raw_affiliations) if raw_affiliations else "",
                "detailed_affiliations": "; ".join(detailed_affiliations) if detailed_affiliations else "",
            }
        )

        return metadata

    except Exception as e:
        print(f"Error fetching metadata for {title_query}: {e}")
        return None

def enrich_cleaned_data(input_file: str, output_file: str, sleep_time: float = 0.15):
    """Enrich data from input CSV with metadata from OpenAlex"""
    logger.info(f"Starting to process cleaned data: {input_file}")
    
    # Read input CSV
    df = pd.read_csv(input_file)

    # Prepare output CSV
    fieldnames = [
        "title",
        "year",
        "datasets",
        "display_author_names",
        "raw_author_names",
        "doi",
        "abstract",
        "institution_display_names",
        "raw_affiliation_strings",
        "detailed_affiliations",
        "original_keywords",
        "original_agency",
        "keywords"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each paper
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
            title = str(row["Title"]).strip()
            print(f"\nProcessing paper {idx+1}/{len(df)}: {title}")

            # Create output row with default values
            output_row = {
                "title": title,
                "year": row.get("Year", ""),  # Use original year if available
                "datasets": "",
                "display_author_names": "",
                "raw_author_names": "",
                "doi": "",
                "abstract": "",
                "institution_display_names": "",
                "raw_affiliation_strings": "",
                "detailed_affiliations": "",
                "original_keywords": row.get("Keywords", ""),
                "original_agency": row.get("Agency", ""),
                "keywords": ""
            }

            # Only try to get metadata if title is not empty or nan
            if title and title.lower() != "nan":
                metadata = get_paper_metadata(title, sleep_time)
                if metadata:
                    # Only update year if not already present in original data
                    if not output_row["year"]:
                        output_row["year"] = metadata.get("year", "")
                    # Update other fields
                    metadata_without_year = {k: v for k, v in metadata.items() if k != "year"}
                    output_row.update(metadata_without_year)

            # Write to CSV
            writer.writerow(output_row)

    logger.info(f"Cleaned data enrichment completed, results saved to: {output_file}")

def enrich_scraped_data(input_file: str, output_file: str, sleep_time: float = 0.15):
    """Enrich scraped data with metadata from OpenAlex"""

    # Read input CSV
    df = pd.read_csv(input_file)

    # Prepare output CSV - keep original columns and add new ones
    original_columns = df.columns.tolist()
    new_columns = ["doi", "keywords", "institution_display_names", "raw_affiliation_strings", "detailed_affiliations"]

    all_columns = original_columns + new_columns

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_columns)
        writer.writeheader()

        # Process each paper
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
            title = str(row["title"]).strip()
            if not title or title.lower() == "nan":
                continue

            print(f"\nProcessing paper {idx+1}/{len(df)}: {title}")

            # Create output row with original data and default values for new fields
            output_row = {col: row[col] for col in original_columns}
            output_row.update(
                {
                    "doi": "N/A",
                    "keywords": "No keywords found",
                    "institution_display_names": "",
                    "raw_affiliation_strings": "",
                    "detailed_affiliations": "",
                }
            )

            # Get metadata from OpenAlex and update if found
            metadata = get_paper_metadata(title, sleep_time)
            if metadata:
                output_row.update(metadata)

            # Write to CSV
            writer.writerow(output_row)

    print(f"\nProcess completed. Results saved to {output_file}")

def standardize_authors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize author information across different formats into a single authors column.
    
    Args:
        df: DataFrame containing author information
    
    Returns:
        DataFrame with standardized authors column
    """
    # For scraped data with list-type authors
    if "authors" in df.columns:
        # Convert string representation of list to actual list if needed
        df["authors"] = df["authors"].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else x)
        # Convert list to semicolon-separated string
        df["authors"] = df["authors"].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)

    # For cleaned data and api data with display_author_names and raw_author_names
    author_cols = []
    if "display_author_names" in df.columns:
        author_cols.append("display_author_names")
    if "raw_author_names" in df.columns:
        author_cols.append("raw_author_names")

    if author_cols:
        # For each row, combine and deduplicate authors from both columns
        def combine_authors(row):
            authors = set()
            for col in author_cols:
                if pd.notna(row[col]) and row[col]:
                    authors.update(row[col].split("; "))
            return "; ".join(sorted(authors))

        df["authors"] = df.apply(combine_authors, axis=1)
        df = df.drop(columns=author_cols)

    return df


def merge_enriched_data(
    enriched_scraped_path: str,
    enriched_cleaned_path: str,
    enriched_api_path: str,
    output_path: str = "data/processed/merged_3_enriched_data.csv",
) -> pd.DataFrame:
    """
    Merge three enriched datasets while preserving all columns.

    Args:
        enriched_scraped_path: Path to enriched scraped data CSV
        enriched_cleaned_path: Path to enriched cleaned data CSV
        enriched_api_path: Path to enriched API data CSV
        output_path: Path to save merged data CSV

    Returns:
        Merged DataFrame
    """
    # Read the three datasets
    scraped_df = pd.read_csv(enriched_scraped_path)
    cleaned_df = pd.read_csv(enriched_cleaned_path)
    api_df = pd.read_csv(enriched_api_path)

    # Print initial counts
    print("Initial counts:")
    print(f"Scraped data: {len(scraped_df)} records")
    print(f"Cleaned data: {len(cleaned_df)} records")
    print(f"API data: {len(api_df)} records")

    # Drop datasets column from scraped and cleaned data
    if "datasets" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["datasets"])

    if "dataset" in api_df.columns:
        api_df = api_df.drop(columns=["dataset"])

    # Process scraped data
    if "project_start_year" in scraped_df.columns:
        scraped_df = scraped_df.drop(columns=["project_start_year"])
    if "project_end_year" in scraped_df.columns:
        scraped_df = scraped_df.rename(columns={"project_end_year": "year"})
        scraped_df["year"] = pd.to_numeric(scraped_df["year"], errors="coerce").astype("Int64")

    if "affiliations" in scraped_df.columns:
        scraped_df = scraped_df.drop(columns=["affiliations"])
    if "project_id" in scraped_df.columns:
        # to int
        scraped_df["project_id"] = pd.to_numeric(scraped_df["project_id"], errors="coerce").astype("Int64")
    if "project_rdc" in scraped_df.columns:
        # rename to Agency
        scraped_df = scraped_df.rename(columns={"project_rdc": "Agency"})
    if "citations" in scraped_df.columns:
        # drop citations column
        scraped_df = scraped_df.drop(columns=["citations"])
    # Rename project_abstract to abstract in scraped data
    if "project_abstract" in scraped_df.columns:
        # drop project_abstract column
        scraped_df = scraped_df.drop(columns=["project_abstract"])

    # Standardize authors in all datasets
    scraped_df = standardize_authors(scraped_df)
    cleaned_df = standardize_authors(cleaned_df)
    api_df = standardize_authors(api_df)

    # Rename columns to standardize
    if "keywords" in scraped_df.columns:
        scraped_df = scraped_df.rename(columns={"keywords": "Keywords"})

    cleaned_df = cleaned_df.rename(columns={"original_agency": "Agency", "original_keywords": "Keywords"})

    # Combine all DataFrames
    merged_df = pd.concat([scraped_df, cleaned_df, api_df], ignore_index=True)

    # Print column counts
    print("\nColumns in merged dataset:")
    for col in sorted(merged_df.columns):
        non_null_count = merged_df[col].count()
        print(f"{col}: {non_null_count} non-null values")

    print(f"\nTotal records in merged dataset: {len(merged_df)}")

    # Save to CSV
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to {output_path}")

    return merged_df

def process_data():
    """
    Main processing function
    """
    try:
        # 1. Read ResearchOutputs.xlsx
        logger.info("Reading ResearchOutputs.xlsx...")
        research_outputs = pd.read_excel('data/raw/ResearchOutputs.xlsx')
        logger.info(f"ResearchOutputs.xlsx data count: {len(research_outputs)}")
        
        # 2. Read web scraping data and deduplicate
        logger.info("Reading web scraping data...")
        web_data = pd.read_csv('data/processed/scraped_data.csv')
        logger.info(f"Original web scraping data count: {len(web_data)}")
        
        # Deduplicate web scraping data
        web_data_deduped = check_duplicates_with_research_outputs(web_data, research_outputs)
        logger.info(f"Web scraping data count after deduplication: {len(web_data_deduped)}")
        
        # 3. Process API data (including keyword deduplication)
        logger.info("Starting API data processing...")
        api_data_deduped = process_api_data()
        logger.info(f"API data count after deduplication: {len(api_data_deduped)}")
        
        # 4. Enrich metadata
        logger.info("Starting metadata enrichment...")
        enrich_cleaned_data(
            input_file="data/raw/cleaned_data.csv",
            output_file="data/processed/enriched_cleaned_data_openalex.csv",
            sleep_time=0.12
        )
        
        enrich_scraped_data(
            input_file="data/processed/deduplicated_scraped_data.csv",
            output_file="data/processed/enriched_scraped_data_openalex.csv",
            sleep_time=0.12
        )
        
        # 5. Merge all data
        logger.info("Starting data merging...")
        merged_df = merge_enriched_data(
            enriched_scraped_path="data/processed/enriched_scraped_data_openalex.csv",
            enriched_cleaned_path="data/processed/enriched_cleaned_data_openalex.csv",
            enriched_api_path="data/processed/final_deduped_data_withkeyword.csv",
            output_path="data/processed/merged_3_enriched_data.csv"
        )
        
        # 6. Print detailed statistics
        logger.info("\nData processing statistics:")
        logger.info(f"1. ResearchOutputs.xlsx data count: {len(research_outputs)}")
        logger.info(f"2. Original web scraping data count: {len(web_data)}")
        logger.info(f"3. Web scraping data count after deduplication: {len(web_data_deduped)}")
        logger.info(f"4. API data count after deduplication: {len(api_data_deduped)}")
        logger.info(f"5. Final merged data count: {len(merged_df)}")
        
        logger.info("Data processing completed!")
        
    except Exception as e:
        logger.error(f"Error occurred during data processing: {str(e)}")
        raise

def fill_authors_from_biblio():
    """
    Fill authors from cleaned_biblio.csv into merged_3_enriched_data.csv
    """
    try:
        # Read the merged enriched data
        merged_data_path = os.path.join("data/processed", "merged_3_enriched_data.csv")
        if not os.path.exists(merged_data_path):
            logging.error(f"Merged data file not found: {merged_data_path}")
            return 0
        
        merged_df = pd.read_csv(merged_data_path)
        logging.info(f"Read merged data with {len(merged_df)} rows")
        
        # Read the cleaned bibliography
        biblio_path = os.path.join("data/raw", "cleaned_biblio.csv")
        if not os.path.exists(biblio_path):
            logging.error(f"Bibliography file not found: {biblio_path}")
            return 0
        
        biblio_df = pd.read_csv(biblio_path)
        logging.info(f"Read bibliography with {len(biblio_df)} rows")
        
        # Process Authors column: replace commas with semicolons and remove 'and' words
        biblio_df['Authors'] = biblio_df['Authors'].apply(lambda x: 
            str(x).replace(',', ';').replace(' and ', ';').replace(' and', ';').replace('and ', ';') 
            if pd.notna(x) else '')
        
        # Create a mapping from title to authors
        title_to_authors = dict(zip(biblio_df['OutputTitle'], biblio_df['Authors']))
        
        # Count how many titles match
        matched_titles = 0
        filled_authors = 0
        
        # Fill authors for matching titles
        for idx, row in merged_df.iterrows():
            title = row['title']
            if title in title_to_authors:
                matched_titles += 1
                if pd.isna(merged_df.at[idx, 'authors']) or merged_df.at[idx, 'authors'] == '':
                    merged_df.at[idx, 'authors'] = title_to_authors[title]
                    filled_authors += 1
        
        logging.info(f"Matched {matched_titles} titles between the two datasets")
        logging.info(f"Filled {filled_authors} empty author fields")
        
        # Save the updated dataframe
        output_path = os.path.join("data/processed", "New_And_Original_ResearchOutputs.csv")
        merged_df.to_csv(output_path, index=False)
        logging.info(f"Saved updated data to {output_path}")
        
        return filled_authors
        
    except Exception as e:
        logging.error(f"Error filling authors from bibliography: {str(e)}")
        return 0

if __name__ == "__main__":
    process_data() 
    filled_authors = fill_authors_from_biblio()
    logging.info(f"Step 7: Added {filled_authors} authors from bibliography") 