import os
import logging
from web_scraping import WebScraper
from api_integration import process_csv_and_find_citations
from data_processing import process_data
from graph_analysis import ResearchGraphBuilder
from visualization import ResearchGraphVisualizer

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
    print("\n=== Research Network Analysis System ===")
    print("Starting analysis process...\n")
    
    try:
        # Step 1: Data file check
        print("Step 1: Checking data files")
        print("-" * 50)
        if check_data_files():
            print("✓ All required data files exist")
        else:
            print("✗ Missing required data files, please ensure all files are prepared")
            return
        print()
        
        # Step 2: Data preprocessing
        print("Step 2: Data preprocessing")
        print("-" * 50)
        processed_file = 'data/processed/New_And_Original_ResearchOutputs.csv'
        if os.path.exists(processed_file):
            print(f"✓ Preprocessed data file exists: {processed_file}")
            print("  Skipping preprocessing step")
        else:
            print("Starting data preprocessing...")
            process_data()
            print("✓ Data preprocessing completed")
        print()
        
        # Step 3: Graph analysis
        print("Step 3: Graph analysis")
        print("-" * 50)
        pkl_path = "output/analysis_results.pkl"
        if os.path.exists(pkl_path):
            print("Found saved analysis results, loading directly...")
            graph_builder = ResearchGraphBuilder.load_from_file(pkl_path)
            print("✓ Successfully loaded saved analysis results")
            
            print("Recalculating network metrics...")
            graph_builder.compute_advanced_metrics()
            print("✓ Network metrics calculation completed")
        else:
            print("Starting new graph analysis...")
            graph_builder = ResearchGraphBuilder(processed_file)
            
            print("Building research network...")
            graph_builder.build_main_graph()
            graph_builder.build_institution_graph()
            graph_builder.build_year_graph()
            
            # Add calls to build author, keyword and citation graphs
            print("Building author collaboration network...")
            graph_builder.build_author_graph()
            
            print("Building keyword co-occurrence network...")
            graph_builder.build_keyword_graph()
            
            print("Building citation network...")
            graph_builder.build_citation_graph()
            
            print("Calculating network metrics...")
            graph_builder.compute_advanced_metrics()
            
            print("Saving analysis results...")
            graph_builder.save_to_file()
            print("✓ Graph analysis completed and results saved")
        print()
        
        # Step 4: Visualization analysis
        print("Step 4: Visualization analysis")
        print("-" * 50)
        print("Initializing visualizer...")
        visualizer = ResearchGraphVisualizer(graph_builder)
        print("✓ Visualizer initialization completed")
        print()
        
        # Step 5: Generate visualization results
        print("Step 5: Generate visualization results")
        print("-" * 50)
        output_dir = 'output/visualizations'
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"Found existing visualization results in: {output_dir}")
            print("Do you want to regenerate visualization results? (y/n)")
            response = input().lower()
            if response != 'y':
                print("Skipping visualization result generation")
                return
        
        print("Generating visualization results...")
        visualizer.save_all_plots()
        print(f"✓ Visualization results have been saved to: {output_dir}")
        print()
        
        print("=== Analysis Process Completed ===")
        print("All steps have been successfully executed!")
        
    except Exception as e:
        print("\n✗ Error: An error occurred during execution")
        print(f"Error message: {str(e)}")
        raise

if __name__ == "__main__":
    main() 