import csv
import time
import os
from typing import List, Dict, Any
import pandas as pd
import requests
from tqdm import tqdm  # Add progress bar support
from datetime import datetime


def reconstruct_abstract(inverted_index):
    """Reconstruct abstract text from inverted index"""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return "No abstract available."

    # Find total word count by getting max position
    max_index = max(pos for positions in inverted_index.values() for pos in positions)
    words = [None] * (max_index + 1)

    # Place each word in its corresponding position
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word

    # Join words to form complete abstract
    return " ".join(word for word in words if word is not None)


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
    headers = {"User-Agent": "YourProject (justin.zhang@wsu.edu)"}

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

        # Extract basic metadata
        title = work.get("title", "N/A")
        year = work.get("publication_year", "N/A")
        doi = work.get("doi", "N/A")
        citation_count = work.get("cited_by_count", "N/A")
        dataset = work.get("datasets", [])

        # Get keywords
        concepts = work.get("concepts", [])
        if concepts:
            keywords = ", ".join([concept.get("display_name", "") for concept in concepts])
        else:
            keywords = "No keywords available"

        # Extract author info - keep complete authorships data
        authorships = work.get("authorships", [])  # Save entire authorships array

        # Extract author information
        author_names = []
        raw_author_names = []
        author_institutions = []
        raw_affiliation_strings = []
        all_institutions = set()
        all_raw_affiliations = set()

        for auth in work.get("authorships", []):
            # Author name
            author_name = auth.get("author", {}).get("display_name", "Unknown")
            author_names.append(author_name)

            # Raw author name
            raw_name = auth.get("raw_author_name", "Unknown")
            raw_author_names.append(raw_name)

            # All institutions for this author
            institutions = [inst.get("display_name", "Unknown") for inst in auth.get("institutions", [])]
            author_institutions.append(institutions)

            # Add to all unique institutions set
            for inst in institutions:
                all_institutions.add(inst)

            # Raw affiliation strings for this author
            affiliations = auth.get("raw_affiliation_strings", [])
            raw_affiliation_strings.append(affiliations)

            # Add to all unique raw affiliations set
            for aff in affiliations:
                all_raw_affiliations.add(aff)

        # Convert sets back to lists
        unique_institutions = list(all_institutions)
        unique_raw_affiliations = list(all_raw_affiliations)

        # Reconstruct abstract
        abstract = None
        inv_index = work.get("abstract_inverted_index")
        if inv_index:
            abstract = reconstruct_abstract(inv_index)
        else:
            abstract = "No abstract available."

        # Get citation info - papers citing this paper (forward citations)
        cited_by_api_url = work.get("cited_by_api_url")
        citing_works = []

        if cited_by_api_url:
            # Implement rate limiting
            time.sleep(sleep_time)

            citing_response = requests.get(cited_by_api_url, headers=headers)
            citing_data = citing_response.json()

            for citing_work in citing_data.get("results", []):
                citing_works.append(
                    {
                        "title": citing_work.get("title", "Unknown"),
                        "year": citing_work.get("publication_year", "Unknown"),
                        "authors": [
                            a.get("author", {}).get("display_name", "Unknown")
                            for a in citing_work.get("authorships", [])
                        ],
                        "doi": citing_work.get("doi", "N/A"),
                    }
                )

        # Organize results - ensure complete authorships included
        return {
            "title": title,
            "year": year,
            "datasets": dataset,
            "doi": doi,
            "keywords": keywords,
            "citation_count": citation_count,
            "abstract": abstract,
            "authorships": authorships,  # Return complete authorships array
            "author_names": author_names,
            "raw_author_names": raw_author_names,
            "author_institutions": author_institutions,
            "raw_affiliation_strings": raw_affiliation_strings,
            "unique_institutions": unique_institutions,
            "unique_raw_affiliations": unique_raw_affiliations,
            "citing_works": citing_works,
        }
    except Exception as e:
        print(f"Error processing {title_query}: {str(e)}")
        return None


def is_fsrdc_related(paper_data, keywords):
    """Check if the paper is related to FSRDC"""
    if not paper_data:
        return False

    # Convert to lowercase for case-insensitive comparison
    keywords_lower = [k.lower() for k in keywords]

    # Check title
    title = paper_data.get("title", "").lower()
    for keyword in keywords_lower:
        if keyword in title:
            return True

    # Check abstract
    abstract = paper_data.get("abstract", "").lower()
    for keyword in keywords_lower:
        if keyword in abstract:
            return True

    # Check institutions
    for institutions in paper_data.get("author_institutions", []):
        for inst in institutions:
            inst_lower = inst.lower()
            for keyword in keywords_lower:
                if keyword in inst_lower:
                    return True

    # Check raw affiliations
    for affiliations in paper_data.get("raw_affiliation_strings", []):
        for aff in affiliations:
            aff_lower = aff.lower()
            for keyword in keywords_lower:
                if keyword in aff_lower:
                    return True

    # Check keywords
    keywords_str = paper_data.get("keywords", "").lower()
    for keyword in keywords_lower:
        if keyword in keywords_str:
            return True

    return False


def process_csv_and_find_citations(
    input_file: str,
    output_file: str,
    title_column: str,
    year_column: str = None,
    sleep_time: float = 0.15,
):
    """Process CSV file and find citation relationships"""
    # Read CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records from {input_file}")
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return

    # Ensure column exists
    if title_column not in df.columns:
        print(f"Error: Column '{title_column}' not found in CSV.")
        return

    # Sort by year (if provided)
    if year_column and year_column in df.columns:
        try:
            df[year_column] = pd.to_numeric(df[year_column], errors="coerce")
            sorted_df = df.sort_values(by=year_column, ascending=False)
            print(f"Sorted data by year (column: {year_column})")
        except Exception as e:
            print(f"Warning: Could not sort by year ({str(e)}). Processing without sorting.")
            sorted_df = df
    else:
        print("No valid year column provided. Processing without sorting.")
        sorted_df = df

    # Remove the max_papers parameter and process all papers
    papers_to_process = sorted_df
    print(f"Processing all {len(papers_to_process)} papers")

    # Define FSRDC related keywords
    fsrdc_keywords = [
        "census bureau",
        "cecus",
        "bureau",
        "fsrdc",
        "fsrdc data",
        "research data center",
        "rdc",
        "bea",
        "restricted microdata",
        "confidential data",
        "annual survey of manufactures",
        "census of construction industries",
        "census of agriculture",
        "census of retail trade",
        "census of manufacturing",
        "census of transportation",
        "census of population",
        "restricted data",
        "microdata",
        "confidential data",
        "confidential microdata",
        "restricted",
        "irs",
        "internal revenue service",
        "federal reserve",
        "nber",
        "cepr",
        "national bureau of economic research",
    ]

    # Prepare to write results to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "title",
            "year",
            "dataset",
            "display_author_names",
            "raw_author_names",
            "doi",
            "abstract",
            "institution_display_names",
            "raw_affiliation_strings",
            "detailed_affiliations",
            "match_rdc_criteria_keywords",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Track processed citing papers to avoid duplicates
        processed_citing_titles = set()
        total_relevant_papers = 0

        # Add progress bar using tqdm
        for idx, row in tqdm(list(papers_to_process.iterrows()), desc="Processing papers"):
            title = str(row[title_column]).strip()
            if not title or title.lower() == "nan":
                continue

            print(f"\nProcessing paper {idx+1}/{len(papers_to_process)}: {title}")

            # Get paper metadata
            paper_data = get_paper_metadata(title, sleep_time)

            if not paper_data:
                continue

            # Process papers citing this work
            citing_works = paper_data.get("citing_works", [])
            print(f"  Found {len(citing_works)} papers citing this work")

            for citing_work in citing_works:
                citing_title = citing_work.get("title")

                # Skip already processed citing papers
                if citing_title in processed_citing_titles:
                    continue

                processed_citing_titles.add(citing_title)

                # Get details for the citing paper
                print(f"  Retrieving details for citing paper: {citing_title}")
                citing_paper_data = get_paper_metadata(citing_title, sleep_time)

                if not citing_paper_data:
                    continue

                # Check if related to FSRDC
                matching_keywords = []
                for keyword in fsrdc_keywords:
                    # Check title
                    if keyword.lower() in citing_paper_data.get("title", "").lower():
                        matching_keywords.append(keyword)
                    # Check abstract
                    elif keyword.lower() in citing_paper_data.get("abstract", "").lower():
                        matching_keywords.append(keyword)
                    # Check institutions and affiliations
                    else:
                        found = False
                        # Check institutions
                        for institutions in citing_paper_data.get("author_institutions", []):
                            for inst in institutions:
                                if keyword.lower() in inst.lower():
                                    matching_keywords.append(keyword)
                                    found = True
                                    break
                            if found:
                                break
                        # Check affiliations
                        if not found:
                            for affiliations in citing_paper_data.get("raw_affiliation_strings", []):
                                for aff in affiliations:
                                    if keyword.lower() in aff.lower():
                                        matching_keywords.append(keyword)
                                        found = True
                                        break
                                if found:
                                    break

                if matching_keywords:
                    total_relevant_papers += 1
                    print(f"  Found FSRDC-related paper: {citing_title}")
                    print(f"  Matching keywords: {', '.join(set(matching_keywords))}")

                    # Prepare author information
                    display_authors = []
                    raw_authors = []
                    for authorship in citing_paper_data.get("authorships", []):
                        # Get correct field names from API response
                        author = authorship.get("author", {})
                        display_name = author.get("display_name")
                        raw_name = authorship.get("raw_author_name")

                        if display_name:
                            display_authors.append(display_name)
                        if raw_name:
                            raw_authors.append(raw_name)

                    # Prepare all types of institution/affiliation information
                    institution_names = set()
                    raw_affiliations = set()
                    detailed_affiliations = set()

                    for authorship in citing_paper_data.get("authorships", []):
                        # Get display_name from institutions array
                        for inst in authorship.get("institutions", []):
                            inst_name = inst.get("display_name")
                            if inst_name:
                                institution_names.add(inst_name)

                        # Get directly from raw_affiliation_strings array
                        raw_aff_strings = authorship.get("raw_affiliation_strings", [])
                        raw_affiliations.update(raw_aff_strings)

                        # Get raw_affiliation_string from affiliations array
                        for aff in authorship.get("affiliations", []):
                            aff_string = aff.get("raw_affiliation_string")
                            if aff_string:
                                detailed_affiliations.add(aff_string)

                    # Debug print to verify data
                    print(f"Authors found: {len(display_authors)}")
                    print(f"Raw authors found: {len(raw_authors)}")
                    print(f"Institutions found: {len(institution_names)}")
                    print(f"Raw affiliations found: {len(raw_affiliations)}")
                    print(f"Detailed affiliations found: {len(detailed_affiliations)}")

                    # Write to CSV
                    writer.writerow(
                        {
                            "title": citing_paper_data.get("title", ""),
                            "year": citing_paper_data.get("year", ""),
                            "dataset": citing_paper_data.get("datasets", []),
                            "display_author_names": "; ".join(display_authors) if display_authors else "",
                            "raw_author_names": "; ".join(raw_authors) if raw_authors else "",
                            "doi": citing_paper_data.get("doi", ""),
                            "abstract": citing_paper_data.get("abstract", ""),
                            "institution_display_names": "; ".join(institution_names) if institution_names else "",
                            "raw_affiliation_strings": "; ".join(raw_affiliations) if raw_affiliations else "",
                            "detailed_affiliations": "; ".join(detailed_affiliations) if detailed_affiliations else "",
                            "match_rdc_criteria_keywords": ", ".join(set(matching_keywords)),
                        }
                    )

    print(f"\nProcess completed. Found {total_relevant_papers} FSRDC-related papers.")
    print(f"Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    process_csv_and_find_citations(
        input_file="cleaned_biblio.csv",
        output_file="data/processed/fsrdc5_related_papers_api_all.csv",  # Updated output filename
        title_column="OutputTitle",
        year_column="year",
        sleep_time=0.12,
    )
