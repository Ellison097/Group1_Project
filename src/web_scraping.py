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

class WebScraper:
    def __init__(self):
        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/web_scraping.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set up list of user agents, randomly select to reduce chance of being blocked
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.80 Mobile/15E148 Safari/604.1'
        ]
        
        # Set up retry strategy
        self.retry_strategy = Retry(
            total=3,  # Set retry count back to 3
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        self.adapter = HTTPAdapter(max_retries=self.retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", self.adapter)
        self.session.mount("http://", self.adapter)
        
        # Add HTTP proxy configuration
        self.proxies = [
            None,  # Sometimes better not to use a proxy
            # Example proxies below, need to be replaced with real working ones
            # 'http://your-proxy-1.com:8080',
            # 'http://your-proxy-2.com:8080',
        ]
        
        # FSRDC criteria
        self.fsrdc_criteria = {
            'acknowledgments': [
                r'Census Bureau',
                r'FSRDC',
                r'Federal Statistical Research Data Center'
            ],
            'data_descriptions': [
                r'Census',
                r'IRS',
                r'BEA',
                r'microdata'
            ],
            'disclosure_review': [
                r'disclosure review',
                r'confidentiality review',
                r'disclosure avoidance'
            ],
            'rdc_mentions': [
                r'Michigan RDC',
                r'Texas RDC',
                r'California RDC',
                r'New York RDC'
            ]
        }
        
        # Track failed projects
        self.failed_projects = []
        
        # Recovery mechanism: record last processed index
        self.last_processed_index = 0
        self.checkpoint_file = 'data/processed/scraping_checkpoint.txt'
    
    def _get_random_delay(self) -> float:
        """Get a random delay time (6-15 seconds)"""
        delay = random.uniform(6, 15)
        time.sleep(delay)
        return delay
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent"""
        return random.choice(self.user_agents)
    
    def _get_random_proxy(self) -> str:
        """Get a random proxy"""
        return random.choice(self.proxies)
    
    def _load_datasets(self, excel_file: str) -> List[str]:
        """Load dataset names list from Datasets sheet"""
        try:
            df = pd.read_excel(excel_file, sheet_name='Datasets')
            datasets = df['Data Name'].dropna().unique().tolist()
            self.logger.info(f"Loaded {len(datasets)} unique dataset names")
            return datasets
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            return []

    def _check_criteria(self, text: str, datasets: List[str]) -> Dict[str, bool]:
        """Check if text meets FSRDC criteria"""
        results = {}
        
        if not text:
            # If text is empty, no criteria are met
            for criterion in self.fsrdc_criteria:
                results[criterion] = False
            results['dataset_mentions'] = False
            return results
        
        # Check basic criteria
        for criterion, patterns in self.fsrdc_criteria.items():
            results[criterion] = any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
        
        # Check dataset names
        results['dataset_mentions'] = any(dataset.lower() in text.lower() for dataset in datasets)
        
        return results

    def _fetch_nber_papers(self, query: str, datasets: List[str]) -> List[Dict[str, Any]]:
        """Get research papers from NBER (National Bureau of Economic Research)"""
        results = []
        try:
            # Prepare headers and proxy
            headers = {'User-Agent': self._get_random_user_agent()}
            proxy = self._get_random_proxy()
            
            # Build query URL
            search_url = f"https://www.nber.org/papers?q={query.replace(' ', '+')}"
            
            # Send request
            self.logger.info(f"Fetching NBER papers for: {query}")
            response = self.session.get(
                search_url,
                headers=headers,
                proxies={'http': proxy, 'https': proxy} if proxy else None,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse response
            soup = BeautifulSoup(response.text, 'html.parser')
            papers = soup.select('div.search-result')
            
            for paper in papers[:5]:  # Only process first 5 results to reduce requests
                try:
                    title_elem = paper.select_one('h3.title')
                    authors_elem = paper.select_one('div.authors')
                    abstract_elem = paper.select_one('div.search-result__abstract')
                    
                    if not title_elem or not authors_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    authors = [author.strip() for author in authors_elem.text.strip().split(',')]
                    abstract = abstract_elem.text.strip() if abstract_elem else ""
                    
                    # Extract paper detail page link
                    link = title_elem.find('a')['href'] if title_elem.find('a') else None
                    full_link = f"https://www.nber.org{link}" if link else None
                    
                    # Integrate metadata
                    metadata = {
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'affiliations': [],
                        'citations': 0,
                        'source': 'NBER',
                        'url': full_link
                    }
                    
                    # Check FSRDC criteria
                    criteria_results = self._check_criteria(
                        metadata['abstract'] + " " + metadata['title'],
                        datasets
                    )
                    metadata.update(criteria_results)
                    
                    if any(criteria_results.values()):
                        results.append(metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error processing NBER paper: {str(e)}")
                    continue  # Skip this paper and continue with next
                
                # Add delay after processing each result
                self._get_random_delay()
            
        except Exception as e:
            self.logger.error(f"Error fetching from NBER: {str(e)}")
        
        return results

    def _fetch_ideas_repec(self, query: str, datasets: List[str]) -> List[Dict[str, Any]]:
        """Get research papers from IDEAS/RePEc"""
        results = []
        try:
            # Prepare headers and proxy
            headers = {'User-Agent': self._get_random_user_agent()}
            proxy = self._get_random_proxy()
            
            # Build query URL
            search_url = f"https://ideas.repec.org/cgi-bin/htsearch?q={query.replace(' ', '+')}"
            
            # Send request
            self.logger.info(f"Fetching IDEAS/RePEc papers for: {query}")
            response = self.session.get(
                search_url,
                headers=headers,
                proxies={'http': proxy, 'https': proxy} if proxy else None,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse response
            soup = BeautifulSoup(response.text, 'html.parser')
            papers = soup.select('li.list-group-item')
            
            for paper in papers[:5]:  # Only process first 5 results to reduce requests
                try:
                    title_elem = paper.select_one('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    link = title_elem['href']
                    full_link = f"https://ideas.repec.org{link}" if not link.startswith('http') else link
                    
                    # Get paper detail page
                    details_response = self.session.get(
                        full_link,
                        headers={'User-Agent': self._get_random_user_agent()},
                        proxies={'http': proxy, 'https': proxy} if proxy else None,
                        timeout=30
                    )
                    details_response.raise_for_status()
                    
                    details_soup = BeautifulSoup(details_response.text, 'html.parser')
                    
                    # Extract authors
                    authors_elem = details_soup.select('meta[name="citation_author"]')
                    authors = [author['content'] for author in authors_elem]
                    
                    # Extract abstract
                    abstract_elem = details_soup.select_one('div#abstract')
                    abstract = abstract_elem.text.strip() if abstract_elem else ""
                    
                    # Integrate metadata
                    metadata = {
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'affiliations': [],
                        'citations': 0,
                        'source': 'IDEAS/RePEc',
                        'url': full_link
                    }
                    
                    # Check FSRDC criteria
                    criteria_results = self._check_criteria(
                        metadata['abstract'] + " " + metadata['title'],
                        datasets
                    )
                    metadata.update(criteria_results)
                    
                    if any(criteria_results.values()):
                        results.append(metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error processing IDEAS/RePEc paper: {str(e)}")
                    continue  # Skip this paper and continue with next
                
                # Add delay after processing each result
                self._get_random_delay()
            
        except Exception as e:
            self.logger.error(f"Error fetching from IDEAS/RePEc: {str(e)}")
        
        return results

    def _fetch_arxiv_papers(self, query: str, datasets: List[str]) -> List[Dict[str, Any]]:
        """Get research papers from arXiv"""
        results = []
        try:
            # Prepare headers and proxy
            headers = {'User-Agent': self._get_random_user_agent()}
            proxy = self._get_random_proxy()
            
            # Build query URL (using arXiv API)
            search_url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&start=0&max_results=5"
            
            # Send request
            self.logger.info(f"Fetching arXiv papers for: {query}")
            response = self.session.get(
                search_url,
                headers=headers,
                proxies={'http': proxy, 'https': proxy} if proxy else None,
                timeout=30
            )
            response.raise_for_status()
            
            # Parse XML response
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            # arXiv XML namespace
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Extract paper data
            for entry in root.findall('./atom:entry', ns):
                try:
                    title = entry.find('./atom:title', ns).text.strip()
                    abstract = entry.find('./atom:summary', ns).text.strip()
                    authors = [author.find('./atom:name', ns).text.strip() 
                             for author in entry.findall('./atom:author', ns)]
                    url = entry.find('./atom:id', ns).text.strip()
                    
                    # Integrate metadata
                    metadata = {
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'affiliations': [],
                        'citations': 0,
                        'source': 'arXiv',
                        'url': url
                    }
                    
                    # Check FSRDC criteria
                    criteria_results = self._check_criteria(
                        metadata['abstract'] + " " + metadata['title'],
                        datasets
                    )
                    metadata.update(criteria_results)
                    
                    if any(criteria_results.values()):
                        results.append(metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error processing arXiv paper: {str(e)}")
                    continue  # Skip this paper and continue with next
                
                # Add delay after processing each result
                self._get_random_delay()
            
        except Exception as e:
            self.logger.error(f"Error fetching from arXiv: {str(e)}")
        
        return results

    def _save_intermediate_results(self, results: List[Dict[str, Any]]) -> None:
        """Save intermediate results"""
        try:
            df = pd.DataFrame(results)
            output_file = 'data/processed/scraped_data_intermediate.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(df)} intermediate results")
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {str(e)}")

    def _save_checkpoint(self, index: int) -> None:
        """Save checkpoint, record processed index"""
        try:
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                f.write(str(index))
            self.logger.info(f"Saved checkpoint at index {index}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    def _load_checkpoint(self) -> int:
        """Load checkpoint, get last processed index"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    index = int(f.read().strip())
                self.logger.info(f"Loaded checkpoint at index {index}")
                return index
            return 0
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return 0

    def _save_failed_projects(self) -> None:
        """Save list of failed projects"""
        try:
            if self.failed_projects:
                output_file = 'data/processed/failed_projects.csv'
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df = pd.DataFrame(self.failed_projects)
                df.to_csv(output_file, index=False)
                self.logger.info(f"Saved {len(self.failed_projects)} failed projects")
        except Exception as e:
            self.logger.error(f"Error saving failed projects: {str(e)}")

    def scrape(self, test_mode: bool = False) -> pd.DataFrame:
        """Scrape research data from web, default is full run"""
        start_time = datetime.now()
        self.logger.info(f"Starting web scraping at {start_time}")
        
        results = []
        excel_file = 'data/raw/ProjectsAllMetadata.xlsx'
        
        # Load dataset names
        datasets = self._load_datasets(excel_file)
        if not datasets:
            self.logger.error("Failed to load datasets")
            return pd.DataFrame()
        
        # Read project information
        try:
            df = pd.read_excel(excel_file, sheet_name='All Metadata')
            pis = df['PI'].dropna().unique().tolist()
            self.logger.info(f"Found {len(pis)} unique PIs in Excel file")
            
            # If in test mode, only process first 10 projects
            if test_mode:
                df = df.head(10)
                self.logger.info("Running in test mode - processing first 10 projects")
            
            # Check if there's a checkpoint
            start_index = self._load_checkpoint()
            
            # Search research outputs for each project
            for i, (_, project) in enumerate(tqdm(list(df.iterrows())[start_index:], desc="Searching research outputs", total=len(df)-start_index)):
                current_index = start_index + i
                try:
                    if pd.isna(project['PI']) or pd.isna(project['Title']):
                        continue
                    
                    query = f"{project['PI']} {project['Title']}"
                    self.logger.info(f"Processing project {current_index+1}/{len(df)}: {project['Proj ID']}")
                    
                    # Get research outputs from multiple sources
                    try:
                        nber_results = self._fetch_nber_papers(query, datasets)
                    except Exception as e:
                        self.logger.error(f"Failed to fetch NBER papers: {str(e)}")
                        nber_results = []
                        
                    try:
                        ideas_repec_results = self._fetch_ideas_repec(query, datasets)
                    except Exception as e:
                        self.logger.error(f"Failed to fetch IDEAS/RePEc papers: {str(e)}")
                        ideas_repec_results = []
                        
                    try:
                        arxiv_results = self._fetch_arxiv_papers(query, datasets)
                    except Exception as e:
                        self.logger.error(f"Failed to fetch arXiv papers: {str(e)}")
                        arxiv_results = []
                    
                    # Merge results and add project information
                    project_results = []
                    for result in nber_results + ideas_repec_results + arxiv_results:
                        result.update({
                            'project_id': project['Proj ID'],
                            'project_pi': project['PI'],
                            'project_rdc': project['RDC'],
                            'project_status': project['Status'],
                            'project_start_year': project['Start Year'],
                            'project_end_year': project['End Year'],
                            'project_abstract': project['Abstract'] if 'Abstract' in project else None
                        })
                        project_results.append(result)
                    
                    if not project_results:
                        self.logger.warning(f"No research outputs found for project: {project['Proj ID']}")
                        # Record projects with no results
                        self.failed_projects.append({
                            'project_id': project['Proj ID'],
                            'project_pi': project['PI'],
                            'project_title': project['Title'],
                            'reason': 'No research outputs found'
                        })
                    
                    results.extend(project_results)
                    
                    # Save intermediate results and checkpoint every 10 projects
                    if (current_index + 1) % 10 == 0:
                        self._save_intermediate_results(results)
                        self._save_checkpoint(current_index + 1)
                        self._save_failed_projects()
                        
                except Exception as e:
                    self.logger.error(f"Error processing project {project['Proj ID']}: {str(e)}")
                    # Record failed projects
                    self.failed_projects.append({
                        'project_id': project['Proj ID'],
                        'project_pi': project['PI'],
                        'project_title': project['Title'],
                        'reason': str(e)
                    })
                    # Save checkpoint to continue from here next time
                    self._save_checkpoint(current_index)
                    self._save_intermediate_results(results)
                    self._save_failed_projects()
                
        except Exception as e:
            self.logger.error(f"Error processing Excel file: {str(e)}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results) if results else pd.DataFrame()
        
        # Save raw data
        output_file = 'data/processed/scraped_data.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_results.to_csv(output_file, index=False)
        
        # Save failed projects
        self._save_failed_projects()
        
        # Clear checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"Web scraping completed at {end_time}")
        self.logger.info(f"Total duration: {duration}")
        self.logger.info(f"Saved {len(df_results)} research outputs to scraped_data.csv")
        
        return df_results

if __name__ == "__main__":
    scraper = WebScraper()
    df = scraper.scrape(test_mode=False)  # Default is full run
    print(f"Scraped {len(df)} research outputs")