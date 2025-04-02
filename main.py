import os
from src.web_scraping import WebScraper
from src.api_integration import APIIntegrator
from src.data_processing import DataProcessor
from dotenv import load_dotenv

def main():
    # 加载环境变量
    load_dotenv()
    
    # 初始化各个模块
    web_scraper = WebScraper()
    api_integrator = APIIntegrator()
    data_processor = DataProcessor()
    
    # 1. 网络爬虫
    print("开始网络爬虫...")
    scraped_data = web_scraper.scrape()
    
    # 2. API集成
    print("开始API集成...")
    api_data = api_integrator.integrate(scraped_data)
    
    # 3. 数据处理和实体解析
    print("开始数据处理和实体解析...")
    final_data = data_processor.process(scraped_data, api_data)
    
    print("处理完成！")

if __name__ == "__main__":
    main() 