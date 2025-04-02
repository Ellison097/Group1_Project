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
        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("logs/web_scraping.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 设置请求头列表，随机选择以减少被封的可能性
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/91.0.4472.80 Mobile/15E148 Safari/604.1'
        ]
        
        # 设置重试策略
        self.retry_strategy = Retry(
            total=3,  # 将重试次数改回为3次
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        self.adapter = HTTPAdapter(max_retries=self.retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", self.adapter)
        self.session.mount("http://", self.adapter)
        
        # 添加 HTTP 代理配置
        self.proxies = [
            None,  # 有时候不使用代理可能更好
            # 以下是一些示例代理，需要替换为真实可用的代理
            # 'http://your-proxy-1.com:8080',
            # 'http://your-proxy-2.com:8080',
        ]
        
        # FSRDC 标准
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
        
        # 上次失败的项目跟踪
        self.failed_projects = []
        
        # 恢复机制：记录最后处理的索引
        self.last_processed_index = 0
        self.checkpoint_file = 'data/processed/scraping_checkpoint.txt'
    
    def _get_random_delay(self) -> float:
        """获取一个随机延迟时间（6-15秒）"""
        delay = random.uniform(6, 15)
        time.sleep(delay)
        return delay
    
    def _get_random_user_agent(self) -> str:
        """获取一个随机用户代理"""
        return random.choice(self.user_agents)
    
    def _get_random_proxy(self) -> str:
        """获取一个随机代理"""
        return random.choice(self.proxies)
    
    def _load_datasets(self, excel_file: str) -> List[str]:
        """从Datasets sheet加载数据集名称列表"""
        try:
            df = pd.read_excel(excel_file, sheet_name='Datasets')
            datasets = df['Data Name'].dropna().unique().tolist()
            self.logger.info(f"Loaded {len(datasets)} unique dataset names")
            return datasets
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            return []

    def _check_criteria(self, text: str, datasets: List[str]) -> Dict[str, bool]:
        """检查文本是否满足FSRDC标准"""
        results = {}
        
        if not text:
            # 如果文本为空，所有标准都不符合
            for criterion in self.fsrdc_criteria:
                results[criterion] = False
            results['dataset_mentions'] = False
            return results
        
        # 检查基本标准
        for criterion, patterns in self.fsrdc_criteria.items():
            results[criterion] = any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
        
        # 检查数据集名称
        results['dataset_mentions'] = any(dataset.lower() in text.lower() for dataset in datasets)
        
        return results

    def _fetch_nber_papers(self, query: str, datasets: List[str]) -> List[Dict[str, Any]]:
        """从 NBER (National Bureau of Economic Research) 获取研究论文"""
        results = []
        try:
            # 准备请求头和代理
            headers = {'User-Agent': self._get_random_user_agent()}
            proxy = self._get_random_proxy()
            
            # 构建查询URL
            search_url = f"https://www.nber.org/papers?q={query.replace(' ', '+')}"
            
            # 发送请求
            self.logger.info(f"Fetching NBER papers for: {query}")
            response = self.session.get(
                search_url,
                headers=headers,
                proxies={'http': proxy, 'https': proxy} if proxy else None,
                timeout=30
            )
            response.raise_for_status()
            
            # 解析响应
            soup = BeautifulSoup(response.text, 'html.parser')
            papers = soup.select('div.search-result')
            
            for paper in papers[:5]:  # 只处理前5个结果以减少请求量
                try:
                    title_elem = paper.select_one('h3.title')
                    authors_elem = paper.select_one('div.authors')
                    abstract_elem = paper.select_one('div.search-result__abstract')
                    
                    if not title_elem or not authors_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    authors = [author.strip() for author in authors_elem.text.strip().split(',')]
                    abstract = abstract_elem.text.strip() if abstract_elem else ""
                    
                    # 提取论文详情页链接
                    link = title_elem.find('a')['href'] if title_elem.find('a') else None
                    full_link = f"https://www.nber.org{link}" if link else None
                    
                    # 整合元数据
                    metadata = {
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'affiliations': [],
                        'citations': 0,
                        'source': 'NBER',
                        'url': full_link
                    }
                    
                    # 检查FSRDC标准
                    criteria_results = self._check_criteria(
                        metadata['abstract'] + " " + metadata['title'],
                        datasets
                    )
                    metadata.update(criteria_results)
                    
                    if any(criteria_results.values()):
                        results.append(metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error processing NBER paper: {str(e)}")
                    continue  # 跳过此论文继续处理下一个
                
                # 每处理一个结果后添加延迟
                self._get_random_delay()
            
        except Exception as e:
            self.logger.error(f"Error fetching from NBER: {str(e)}")
        
        return results

    def _fetch_ideas_repec(self, query: str, datasets: List[str]) -> List[Dict[str, Any]]:
        """从 IDEAS/RePEc 获取研究论文"""
        results = []
        try:
            # 准备请求头和代理
            headers = {'User-Agent': self._get_random_user_agent()}
            proxy = self._get_random_proxy()
            
            # 构建查询URL
            search_url = f"https://ideas.repec.org/cgi-bin/htsearch?q={query.replace(' ', '+')}"
            
            # 发送请求
            self.logger.info(f"Fetching IDEAS/RePEc papers for: {query}")
            response = self.session.get(
                search_url,
                headers=headers,
                proxies={'http': proxy, 'https': proxy} if proxy else None,
                timeout=30
            )
            response.raise_for_status()
            
            # 解析响应
            soup = BeautifulSoup(response.text, 'html.parser')
            papers = soup.select('li.list-group-item')
            
            for paper in papers[:5]:  # 只处理前5个结果以减少请求量
                try:
                    title_elem = paper.select_one('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    link = title_elem['href']
                    full_link = f"https://ideas.repec.org{link}" if not link.startswith('http') else link
                    
                    # 获取论文详情页
                    details_response = self.session.get(
                        full_link,
                        headers={'User-Agent': self._get_random_user_agent()},
                        proxies={'http': proxy, 'https': proxy} if proxy else None,
                        timeout=30
                    )
                    details_response.raise_for_status()
                    
                    details_soup = BeautifulSoup(details_response.text, 'html.parser')
                    
                    # 提取作者
                    authors_elem = details_soup.select('meta[name="citation_author"]')
                    authors = [author['content'] for author in authors_elem]
                    
                    # 提取摘要
                    abstract_elem = details_soup.select_one('div#abstract')
                    abstract = abstract_elem.text.strip() if abstract_elem else ""
                    
                    # 整合元数据
                    metadata = {
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'affiliations': [],
                        'citations': 0,
                        'source': 'IDEAS/RePEc',
                        'url': full_link
                    }
                    
                    # 检查FSRDC标准
                    criteria_results = self._check_criteria(
                        metadata['abstract'] + " " + metadata['title'],
                        datasets
                    )
                    metadata.update(criteria_results)
                    
                    if any(criteria_results.values()):
                        results.append(metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error processing IDEAS/RePEc paper: {str(e)}")
                    continue  # 跳过此论文继续处理下一个
                
                # 每处理一个结果后添加延迟
                self._get_random_delay()
            
        except Exception as e:
            self.logger.error(f"Error fetching from IDEAS/RePEc: {str(e)}")
        
        return results

    def _fetch_arxiv_papers(self, query: str, datasets: List[str]) -> List[Dict[str, Any]]:
        """从 arXiv 获取研究论文"""
        results = []
        try:
            # 准备请求头和代理
            headers = {'User-Agent': self._get_random_user_agent()}
            proxy = self._get_random_proxy()
            
            # 构建查询URL (使用arXiv API)
            search_url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&start=0&max_results=5"
            
            # 发送请求
            self.logger.info(f"Fetching arXiv papers for: {query}")
            response = self.session.get(
                search_url,
                headers=headers,
                proxies={'http': proxy, 'https': proxy} if proxy else None,
                timeout=30
            )
            response.raise_for_status()
            
            # 解析XML响应
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            # arXiv XML命名空间
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # 提取论文数据
            for entry in root.findall('./atom:entry', ns):
                try:
                    title = entry.find('./atom:title', ns).text.strip()
                    abstract = entry.find('./atom:summary', ns).text.strip()
                    authors = [author.find('./atom:name', ns).text.strip() 
                             for author in entry.findall('./atom:author', ns)]
                    url = entry.find('./atom:id', ns).text.strip()
                    
                    # 整合元数据
                    metadata = {
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'affiliations': [],
                        'citations': 0,
                        'source': 'arXiv',
                        'url': url
                    }
                    
                    # 检查FSRDC标准
                    criteria_results = self._check_criteria(
                        metadata['abstract'] + " " + metadata['title'],
                        datasets
                    )
                    metadata.update(criteria_results)
                    
                    if any(criteria_results.values()):
                        results.append(metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error processing arXiv paper: {str(e)}")
                    continue  # 跳过此论文继续处理下一个
                
                # 每处理一个结果后添加延迟
                self._get_random_delay()
            
        except Exception as e:
            self.logger.error(f"Error fetching from arXiv: {str(e)}")
        
        return results

    def _save_intermediate_results(self, results: List[Dict[str, Any]]) -> None:
        """保存中间结果"""
        try:
            df = pd.DataFrame(results)
            output_file = 'data/processed/scraped_data_intermediate.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(df)} intermediate results")
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {str(e)}")

    def _save_checkpoint(self, index: int) -> None:
        """保存检查点，记录处理到的索引"""
        try:
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                f.write(str(index))
            self.logger.info(f"Saved checkpoint at index {index}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    def _load_checkpoint(self) -> int:
        """加载检查点，获取上次处理到的索引"""
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
        """保存失败的项目列表"""
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
        """从网页爬取研究数据，默认为全量运行"""
        start_time = datetime.now()
        self.logger.info(f"Starting web scraping at {start_time}")
        
        results = []
        excel_file = 'data/raw/ProjectsAllMetadata.xlsx'
        
        # 加载数据集名称
        datasets = self._load_datasets(excel_file)
        if not datasets:
            self.logger.error("Failed to load datasets")
            return pd.DataFrame()
        
        # 读取项目信息
        try:
            df = pd.read_excel(excel_file, sheet_name='All Metadata')
            pis = df['PI'].dropna().unique().tolist()
            self.logger.info(f"Found {len(pis)} unique PIs in Excel file")
            
            # 如果是测试模式，只处理前10个项目
            if test_mode:
                df = df.head(10)
                self.logger.info("Running in test mode - processing first 10 projects")
            
            # 检查是否有检查点
            start_index = self._load_checkpoint()
            
            # 对每个项目搜索研究输出
            for i, (_, project) in enumerate(tqdm(list(df.iterrows())[start_index:], desc="Searching research outputs", total=len(df)-start_index)):
                current_index = start_index + i
                try:
                    if pd.isna(project['PI']) or pd.isna(project['Title']):
                        continue
                    
                    query = f"{project['PI']} {project['Title']}"
                    self.logger.info(f"Processing project {current_index+1}/{len(df)}: {project['Proj ID']}")
                    
                    # 从多个来源获取研究输出
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
                    
                    # 合并结果并添加项目信息
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
                        # 记录没有找到结果的项目
                        self.failed_projects.append({
                            'project_id': project['Proj ID'],
                            'project_pi': project['PI'],
                            'project_title': project['Title'],
                            'reason': 'No research outputs found'
                        })
                    
                    results.extend(project_results)
                    
                    # 每处理10个项目后保存一次中间结果和检查点
                    if (current_index + 1) % 10 == 0:
                        self._save_intermediate_results(results)
                        self._save_checkpoint(current_index + 1)
                        self._save_failed_projects()
                        
                except Exception as e:
                    self.logger.error(f"Error processing project {project['Proj ID']}: {str(e)}")
                    # 记录处理失败的项目
                    self.failed_projects.append({
                        'project_id': project['Proj ID'],
                        'project_pi': project['PI'],
                        'project_title': project['Title'],
                        'reason': str(e)
                    })
                    # 保存检查点，以便下次从这里继续
                    self._save_checkpoint(current_index)
                    self._save_intermediate_results(results)
                    self._save_failed_projects()
                
        except Exception as e:
            self.logger.error(f"Error processing Excel file: {str(e)}")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results) if results else pd.DataFrame()
        
        # 保存原始数据
        output_file = 'data/processed/scraped_data.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_results.to_csv(output_file, index=False)
        
        # 保存失败的项目
        self._save_failed_projects()
        
        # 清除检查点文件
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
    df = scraper.scrape(test_mode=False)  # 默认为全量运行
    print(f"Scraped {len(df)} research outputs") 