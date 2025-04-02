import os
import logging
from web_scraping import WebScraper
from api_integration import process_csv_and_find_citations
from data_processing import process_data

# 配置日志
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
    """检查文件是否存在"""
    if os.path.exists(file_path):
        logger.info(f"文件 {file_path} 已存在")
        return True
    logger.info(f"文件 {file_path} 不存在")
    return False

def main():
    """主函数"""
    try:
        # 定义文件路径
        scraped_data_path = 'data/processed/scraped_data.csv'
        api_data_path = 'data/processed/fsrdc5_related_papers_api_all.csv'
        final_data_path = 'data/processed/final_combined_data.csv'

        # 初始化爬虫
        scraper = WebScraper()

        # 检查并执行网页爬取
        if not check_file_exists(scraped_data_path):
            logger.info("开始网页爬取...")
            scraper.scrape_all()
        else:
            logger.info("跳过网页爬取，使用现有数据")

        # 检查并执行API数据获取
        if not check_file_exists(api_data_path):
            logger.info("开始API数据获取...")
            process_csv_and_find_citations(
                input_file='data/raw/ResearchOutputs.xlsx',
                output_file=api_data_path,
                title_column='Title',
                year_column='Year',
                sleep_time=0.15
            )
        else:
            logger.info("跳过API数据获取，使用现有数据")

        # 检查并执行数据处理
        if not check_file_exists(final_data_path):
            logger.info("开始数据处理...")
            process_data()
        else:
            logger.info("跳过数据处理，使用现有数据")

        logger.info("所有处理完成")

    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 