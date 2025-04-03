import os
import logging
from web_scraping import WebScraper
from api_integration import process_csv_and_find_citations
from data_processing import process_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_file_exists(file_path):
    """Check if file exists"""
    if os.path.exists(file_path):
        logger.info(f"File {file_path} exists")
        return True
    logger.info(f"File {file_path} does not exist")
    return False

def check_data_files():
    """Check if all required data files exist"""
    required_files = [
        'data/raw/ResearchOutputs.xlsx',
        'data/raw/cleaned_biblio.csv',
        'data/raw/cleaned_data.csv',
        'data/processed/scraped_data.csv',
        'data/processed/fsrdc5_related_papers_api_all.csv',
        'data/processed/enriched_scraped_data_openalex.csv',
        'data/processed/enriched_cleaned_data_openalex.csv',
        'data/processed/merged_3_enriched_data.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("\nMissing required data files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True

def main():
    """Main function"""
    try:
        # Check data files
        if not check_data_files():
            print("\nPlease ensure all required data files exist before running the program.")
            return
        
        # If all files exist, return directly
        print("\nAll data files exist, no need to reprocess.")
        return
        
        # The following code will not execute due to the return above
        # Run data processing
        # from data_processing import process_data
        # process_data()
        
        # print("\nData processing completed!")
        
    except Exception as e:
        print(f"\nProgram error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 