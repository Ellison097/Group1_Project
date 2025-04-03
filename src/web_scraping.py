import pandas as pd
import requests
import re
import time
import logging
import os
import random
from typing import List, Dict, Any
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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

class WebScraper:
    def __init__(self):
        self.base_url = "https://www.fsrdc.org/research-outputs/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.output_file = 'data/processed/scraped_data.csv'
        self.failed_projects_file = 'data/processed/failed_projects.csv'
        self.failed_projects = []

    def get_page_content(self, url: str) -> str:
        """Get page content with retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get page content after {max_retries} attempts: {e}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def parse_project_page(self, url: str) -> Dict[str, Any]:
        """Parse individual project page"""
        try:
            content = self.get_page_content(url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract project information
            project_info = {}
            
            # Title
            title_elem = soup.find('h1', class_='entry-title')
            project_info['title'] = title_elem.text.strip() if title_elem else ''
            
            # Abstract
            abstract_elem = soup.find('div', class_='entry-content')
            project_info['abstract'] = abstract_elem.text.strip() if abstract_elem else ''
            
            # Authors
            authors_elem = soup.find('div', class_='authors')
            project_info['authors'] = authors_elem.text.strip() if authors_elem else ''
            
            # Year
            year_elem = soup.find('span', class_='year')
            project_info['year'] = year_elem.text.strip() if year_elem else ''
            
            # Keywords
            keywords_elem = soup.find('div', class_='keywords')
            project_info['keywords'] = keywords_elem.text.strip() if keywords_elem else ''
            
            return project_info
            
        except Exception as e:
            logger.error(f"Error parsing project page {url}: {e}")
            self.failed_projects.append({'url': url, 'error': str(e)})
            return None

    def scrape_all(self):
        """Scrape all project pages"""
        try:
            # Get main page content
            content = self.get_page_content(self.base_url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find all project links
            project_links = soup.find_all('a', class_='project-link')
            
            # Store all project data
            all_projects = []
            
            # Process each project
            for link in project_links:
                project_url = link.get('href')
                if project_url:
                    logger.info(f"Processing project: {project_url}")
                    project_data = self.parse_project_page(project_url)
                    if project_data:
                        all_projects.append(project_data)
                    time.sleep(1)  # Avoid request overload
            
            # Convert to DataFrame
            df = pd.DataFrame(all_projects)
            
            # Save to CSV
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            df.to_csv(self.output_file, index=False)
            logger.info(f"Scraped data saved to {self.output_file}")
            
            # Save failed projects
            if self.failed_projects:
                failed_df = pd.DataFrame(self.failed_projects)
                failed_df.to_csv(self.failed_projects_file, index=False)
                logger.info(f"Failed projects saved to {self.failed_projects_file}")
            
        except Exception as e:
            logger.error(f"Error in scraping process: {e}")
            raise

def scrape_data():
    """Main function to run web scraping"""
    scraper = WebScraper()
    scraper.scrape_all()

if __name__ == "__main__":
    scrape_data() 