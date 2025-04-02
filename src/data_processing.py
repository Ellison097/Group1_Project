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

# 设置日志
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
        self.threshold = 85  # 模糊匹配阈值
        
        # 加载已有的研究成果
        self.existing_outputs = self._load_existing_outputs()

    def _load_existing_outputs(self) -> pd.DataFrame:
        """加载已有的研究成果数据"""
        try:
            return pd.read_excel('data/raw/ResearchOutputs.xlsx')
        except Exception as e:
            logger.error(f"Error loading existing outputs: {e}")
            return pd.DataFrame()

    def _clean_text(self, text: str) -> str:
        """清理文本数据"""
        if pd.isna(text):
            return ""
        # 转换为小写
        text = str(text).lower()
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 移除多余空格
        text = ' '.join(text.split())
        return text

    def _check_uniqueness(self, title: str) -> bool:
        """检查研究产出是否在2024年数据集中"""
        if self.existing_outputs.empty:
            return True
        
        cleaned_title = self._clean_text(title)
        for _, row in self.existing_outputs.iterrows():
            existing_title = self._clean_text(row['OutputTitle'])
            # 使用模糊匹配
            similarity = fuzz.ratio(cleaned_title, existing_title)
            if similarity >= self.threshold:
                return False
        return True

    def _validate_fsrdc_criteria(self, row: pd.Series) -> bool:
        """验证FSRDC标准"""
        # 检查是否满足任一标准
        criteria_columns = [
            'acknowledgments',
            'data_descriptions',
            'disclosure_review',
            'rdc_mentions',
            'dataset_mentions'
        ]
        
        return any(row[col] for col in criteria_columns if col in row)

    def _merge_data(self, scraped_data: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
        """合并爬虫数据和API数据"""
        # 基于标题进行合并
        merged_data = pd.merge(
            scraped_data,
            api_data,
            on='title',
            how='outer',
            suffixes=('_scraped', '_api')
        )
        return merged_data

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        # 删除重复行
        df = df.drop_duplicates(subset=['title'])
        
        # 清理文本列
        text_columns = ['title', 'abstract', 'authors']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_text)
        
        # 处理缺失值
        df = df.fillna({
            'abstract': '',
            'authors': '[]',
            'year': 0
        })
        
        return df

    def process(self, scraped_data: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
        """处理数据并生成最终数据集"""
        # 合并数据
        merged_data = self._merge_data(scraped_data, api_data)
        
        # 清理数据
        cleaned_data = self._clean_data(merged_data)
        
        # 验证唯一性和FSRDC标准
        final_data = []
        for _, row in cleaned_data.iterrows():
            if self._check_uniqueness(row['title']) and self._validate_fsrdc_criteria(row):
                final_data.append(row)
        
        # 转换为DataFrame
        final_df = pd.DataFrame(final_data)
        
        # 保存处理后的数据
        final_df.to_csv('data/processed/final_research_outputs.csv', index=False)
        
        # 输出统计信息
        logger.info(f"Total records processed: {len(cleaned_data)}")
        logger.info(f"Unique records after processing: {len(final_df)}")
        
        return final_df

    def _is_duplicate(self, title: str, authors: List[str]) -> bool:
        """检查是否是重复的研究成果"""
        if self.existing_outputs.empty:
            return False
            
        # 使用模糊匹配检查标题
        title_similarities = [fuzz.ratio(title.lower(), t.lower()) 
                            for t in self.existing_outputs['Title']]
        
        # 如果标题相似度超过85%，认为是重复
        if max(title_similarities) > 85:
            return True
            
        # 检查作者
        for _, row in self.existing_outputs.iterrows():
            existing_authors = str(row['Authors']).lower().split(',')
            if any(author.lower() in existing_authors for author in authors):
                return True
                
        return False

    def _standardize_authors(self, authors: List[str]) -> str:
        """标准化作者列表"""
        if not authors:
            return ""
        return ", ".join([self._clean_text(author) for author in authors])

    def _extract_year(self, date_str: str) -> str:
        """从日期字符串中提取年份"""
        if pd.isna(date_str):
            return ""
        try:
            return str(pd.to_datetime(date_str).year)
        except:
            return ""

    def process_data(self, scraped_data: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
        """处理爬虫和API数据"""
        try:
            # 合并数据
            all_data = pd.concat([scraped_data, api_data], ignore_index=True)
            
            # 数据清洗
            all_data['Title'] = all_data['Title'].apply(self._clean_text)
            all_data['Authors'] = all_data['Authors'].apply(self._standardize_authors)
            all_data['Abstract'] = all_data['Abstract'].apply(self._clean_text)
            all_data['Year'] = all_data['Year'].apply(self._extract_year)
            
            # 去重
            unique_outputs = []
            for _, row in all_data.iterrows():
                if not self._is_duplicate(row['Title'], row['Authors'].split(', ')):
                    unique_outputs.append(row)
                    
            # 转换为DataFrame
            processed_df = pd.DataFrame(unique_outputs)
            
            # 添加处理时间戳
            processed_df['Processed_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存处理后的数据
            output_file = 'data/processed/processed_research_outputs.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            processed_df.to_csv(output_file, index=False)
            
            logger.info(f"Successfully processed {len(processed_df)} unique research outputs")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return pd.DataFrame()

    def generate_summary(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """生成数据摘要"""
        if processed_df.empty:
            return {}
            
        summary = {
            'total_outputs': len(processed_df),
            'unique_authors': len(processed_df['Authors'].unique()),
            'year_distribution': processed_df['Year'].value_counts().to_dict(),
            'source_distribution': processed_df['Source'].value_counts().to_dict(),
            'fsrdc_compliant': processed_df['fsrdc_compliant'].sum() if 'fsrdc_compliant' in processed_df.columns else 0
        }
        
        # 保存摘要
        summary_file = 'data/processed/research_outputs_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        return summary

def process_api_data():
    """处理API数据"""
    try:
        logger.info("开始处理API数据...")
        
        # 1. 读取原始API数据
        df = pd.read_csv("data/processed/fsrdc5_related_papers_api_all.csv")
        logger.info(f"原始API数据量: {len(df)}")
        
        # 2. 基于标题去重
        deduplicate_self = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
        logger.info(f"标题去重后数据量: {len(deduplicate_self)}")
        
        # 保存第一次去重结果
        deduplicate_self.to_csv("data/processed/deduplicate_self.csv", index=False)
        
        # 3. 读取cleaned_biblio.csv
        cleaned_biblio = pd.read_csv("data/processed/cleaned_biblio.csv")
        logger.info(f"读取cleaned_biblio.csv，共 {len(cleaned_biblio)} 条记录")
        
        # 4. 使用模糊匹配进行去重
        def is_similar(title1, title2, threshold=80):
            """比较两个标题的相似度，如果超过阈值则返回True"""
            if pd.isna(title1) or pd.isna(title2):
                return False
            return fuzz.ratio(str(title1).lower(), str(title2).lower()) >= threshold
        
        # 创建标记列表
        keep_rows = []
        
        # 检查每个标题
        for idx, row in deduplicate_self.iterrows():
            # 默认保留该行
            keep = True
            current_title = row["title"]
            
            # 与cleaned_biblio中的每个标题比较
            for biblio_title in cleaned_biblio["OutputTitle"]:
                if is_similar(current_title, biblio_title):
                    # 如果找到相似标题，标记为不保留
                    keep = False
                    break
            
            keep_rows.append(keep)
        
        # 使用标记列表过滤数据
        after_fuzzy_df = deduplicate_self[keep_rows].reset_index(drop=True)
        logger.info(f"模糊匹配去重后数据量: {len(after_fuzzy_df)}")
        
        # 5. 过滤包含2个或更多FSRDC关键词的记录
        def count_keywords(keywords_str):
            """计算关键词数量"""
            if pd.isna(keywords_str):
                return 0
            return len(str(keywords_str).split(", "))
        
        # 过滤记录
        after_fuzzy_df_larger2 = after_fuzzy_df[
            after_fuzzy_df["match_rdc_criteria_keywords"].apply(count_keywords) >= 2
        ].reset_index(drop=True)
        logger.info(f"关键词过滤后数据量: {len(after_fuzzy_df_larger2)}")
        
        # 保存第一次最终结果（不含OpenAlex关键词）
        after_fuzzy_df_larger2.to_csv("data/processed/final_deduped_data.csv", index=False)
        
        # 6. 从OpenAlex API获取关键词
        def fetch_openalex_data_by_title(title):
            """从OpenAlex获取数据"""
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
                logger.error(f"搜索OpenAlex时出错 '{title}': {e}")
                return None
        
        def get_openalex_keywords(work):
            """从OpenAlex工作对象中提取关键词"""
            if not work:
                return "No keywords found"
            
            concepts = work.get("concepts", [])
            if concepts:
                return ", ".join([concept.get("display_name", "") for concept in concepts])
            return "No keywords found"
        
        # 添加OpenAlex关键词
        openalex_keywords = []
        logger.info("开始从OpenAlex获取关键词...")
        
        for idx, row in after_fuzzy_df_larger2.iterrows():
            logger.info(f"处理第 {idx+1}/{len(after_fuzzy_df_larger2)} 条记录")
            work = fetch_openalex_data_by_title(row["title"])
            keywords = get_openalex_keywords(work)
            openalex_keywords.append(keywords)
            time.sleep(0.12)  # 避免请求过快
        
        after_fuzzy_df_larger2["Keywords"] = openalex_keywords
        
        # 保存最终结果（包含OpenAlex关键词）
        after_fuzzy_df_larger2.to_csv("data/processed/final_deduped_data_withkeyword.csv", index=False)
        
        # 输出处理结果统计
        logger.info(f"原始合并数据量: {len(df)}")
        logger.info(f"第一次去重后数据量: {len(deduplicate_self)}")
        logger.info(f"模糊匹配去重后数据量: {len(after_fuzzy_df)}")
        logger.info(f"关键词过滤后数据量: {len(after_fuzzy_df_larger2)}")
        logger.info("已添加OpenAlex关键词列")
        
        return after_fuzzy_df_larger2
        
    except Exception as e:
        logger.error(f"处理API数据时出错: {str(e)}")
        raise

def check_duplicates_with_research_outputs(scraped_data: pd.DataFrame, research_outputs: pd.DataFrame) -> pd.DataFrame:
    """
    检查爬取的数据是否与ResearchOutputs.xlsx中的数据重复，使用精确匹配和模糊匹配
    
    Args:
        scraped_data: 从web_scraping.py获取的数据
        research_outputs: 从ResearchOutputs.xlsx读取的数据
    
    Returns:
        去重后的DataFrame
    """
    logger.info("开始检查重复数据...")
    
    # 确保两个DataFrame都有title列
    if 'title' not in scraped_data.columns:
        logger.error("scraped_data中没有title列")
        return scraped_data
    
    if 'OutputTitle' not in research_outputs.columns:
        logger.error("research_outputs中没有OutputTitle列")
        return scraped_data
    
    # 1. 首先进行基于标题的精确去重
    scraped_data = scraped_data.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    logger.info(f"精确去重后数据量: {len(scraped_data)}")
    
    # 2. 使用模糊匹配进行去重
    def is_similar(title1, title2, threshold=80):
        """比较两个标题的相似度，如果超过阈值则返回True"""
        if pd.isna(title1) or pd.isna(title2):
            return False
        return fuzz.ratio(str(title1).lower(), str(title2).lower()) >= threshold
    
    # 创建标记列表
    keep_rows = []
    
    # 检查每个标题
    for idx, row in scraped_data.iterrows():
        # 默认保留该行
        keep = True
        current_title = row["title"]
        
        # 与ResearchOutputs中的每个标题比较
        for biblio_title in research_outputs["OutputTitle"]:
            if is_similar(current_title, biblio_title):
                # 如果找到相似标题，标记为不保留
                keep = False
                break
        
        keep_rows.append(keep)
    
    # 使用标记列表过滤数据
    deduplicated_data = scraped_data[keep_rows].reset_index(drop=True)
    
    # 记录去重结果
    logger.info(f"原始数据量: {len(scraped_data)}")
    logger.info(f"模糊匹配去重后数据量: {len(deduplicated_data)}")
    
    # 保存重复数据到单独的文件
    duplicate_data = scraped_data[~scraped_data.index.isin(deduplicated_data.index)]
    if not duplicate_data.empty:
        output_file = 'data/processed/duplicate_data.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        duplicate_data.to_csv(output_file, index=False)
        logger.info(f"重复数据已保存到 {output_file}")
    
    return deduplicated_data

def check_duplicates_between_sources(web_data: pd.DataFrame, api_data: pd.DataFrame) -> pd.DataFrame:
    """
    检查web数据和API数据之间的重复，使用精确匹配和模糊匹配，并智能合并列
    
    Args:
        web_data: web爬取的数据
        api_data: API获取的数据
    
    Returns:
        去重后的DataFrame
    """
    logger.info("开始检查web数据和API数据之间的重复...")
    
    # 打印两个数据集的列名和示例数据
    logger.info("\nWeb数据列名和示例:")
    logger.info(f"列名: {web_data.columns.tolist()}")
    logger.info("\n前3行数据示例:")
    logger.info(web_data.head(3).to_string())
    
    logger.info("\nAPI数据列名和示例:")
    logger.info(f"列名: {api_data.columns.tolist()}")
    logger.info("\n前3行数据示例:")
    logger.info(api_data.head(3).to_string())
    
    # 分析可能的列名对应关系
    def find_similar_columns(df1_cols, df2_cols, threshold=80):
        """查找相似的列名"""
        similar_cols = {}
        for col1 in df1_cols:
            for col2 in df2_cols:
                similarity = fuzz.ratio(str(col1).lower(), str(col2).lower())
                if similarity >= threshold:
                    similar_cols[col1] = col2
        return similar_cols
    
    similar_columns = find_similar_columns(web_data.columns, api_data.columns)
    logger.info("\n可能的列名对应关系:")
    for web_col, api_col in similar_columns.items():
        logger.info(f"{web_col} <-> {api_col}")
    
    # 确保两个DataFrame都有必要的列
    required_columns = ['title', 'authors']
    for col in required_columns:
        if col not in web_data.columns or col not in api_data.columns:
            logger.error(f"缺少必要的列: {col}")
            return pd.concat([web_data, api_data], ignore_index=True)
    
    def is_similar(title1, title2, threshold=80):
        """比较两个标题的相似度"""
        if pd.isna(title1) or pd.isna(title2):
            return False
        return fuzz.ratio(str(title1).lower(), str(title2).lower()) >= threshold
    
    def authors_overlap(authors1, authors2):
        """检查作者是否有重叠"""
        if pd.isna(authors1) or pd.isna(authors2):
            return False
        authors1 = set(str(authors1).lower().split(', '))
        authors2 = set(str(authors2).lower().split(', '))
        return bool(authors1.intersection(authors2))
    
    # 创建标记列表
    keep_rows = []
    
    # 检查每个web数据条目
    for idx, web_row in web_data.iterrows():
        # 默认保留该行
        keep = True
        current_title = web_row["title"]
        current_authors = web_row["authors"]
        
        # 与API数据中的每个条目比较
        for _, api_row in api_data.iterrows():
            api_title = api_row["title"]
            api_authors = api_row["authors"]
            
            # 如果标题相似或作者有重叠，标记为不保留
            if is_similar(current_title, api_title) or authors_overlap(current_authors, api_authors):
                keep = False
                break
        
        keep_rows.append(keep)
    
    # 使用标记列表过滤web数据
    web_data_filtered = web_data[keep_rows].reset_index(drop=True)
    
    # 智能合并列
    # 1. 找出共同的列
    common_columns = list(set(web_data_filtered.columns) & set(api_data.columns))
    logger.info(f"\n共同列: {common_columns}")
    
    # 2. 找出web数据独有的列
    web_only_columns = list(set(web_data_filtered.columns) - set(api_data.columns))
    logger.info(f"Web数据独有列: {web_only_columns}")
    
    # 3. 找出API数据独有的列
    api_only_columns = list(set(api_data.columns) - set(web_data_filtered.columns))
    logger.info(f"API数据独有列: {api_only_columns}")
    
    # 4. 创建合并后的DataFrame
    merged_data = pd.DataFrame()
    
    # 5. 添加共同列（优先使用API数据中的值）
    for col in common_columns:
        merged_data[col] = api_data[col]
        # 对于web数据中独有的行，使用web数据的值
        merged_data.loc[web_data_filtered.index, col] = web_data_filtered[col]
    
    # 6. 添加web数据独有的列
    for col in web_only_columns:
        merged_data[col] = web_data_filtered[col]
        # 对于API数据中的行，填充空值
        merged_data.loc[api_data.index, col] = None
    
    # 7. 添加API数据独有的列
    for col in api_only_columns:
        merged_data[col] = api_data[col]
        # 对于web数据中的行，填充空值
        merged_data.loc[web_data_filtered.index, col] = None
    
    # 记录去重结果
    logger.info(f"\nWeb数据原始量: {len(web_data)}")
    logger.info(f"API数据原始量: {len(api_data)}")
    logger.info(f"去重后Web数据量: {len(web_data_filtered)}")
    logger.info(f"最终合并数据量: {len(merged_data)}")
    
    # 保存重复数据到单独的文件
    duplicate_data = web_data[~web_data.index.isin(web_data_filtered.index)]
    if not duplicate_data.empty:
        output_file = 'data/processed/web_api_duplicates.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        duplicate_data.to_csv(output_file, index=False)
        logger.info(f"重复数据已保存到 {output_file}")
    
    # 保存列名信息
    column_info = {
        'common_columns': common_columns,
        'web_only_columns': web_only_columns,
        'api_only_columns': api_only_columns,
        'similar_columns': similar_columns
    }
    with open('data/processed/column_info.json', 'w') as f:
        json.dump(column_info, f, indent=4)
    
    return merged_data

def process_data():
    """
    主处理函数
    """
    try:
        # 1. 读取ResearchOutputs.xlsx
        logger.info("正在读取ResearchOutputs.xlsx...")
        research_outputs = pd.read_excel('data/raw/ResearchOutputs.xlsx')
        logger.info(f"ResearchOutputs.xlsx数据量: {len(research_outputs)}")
        
        # 2. 读取web scraping数据并去重
        logger.info("正在读取web scraping数据...")
        web_data = pd.read_csv('data/processed/scraped_data.csv')
        logger.info(f"Web scraping原始数据量: {len(web_data)}")
        
        # 对web scraping数据进行去重
        web_data_deduped = check_duplicates_with_research_outputs(web_data, research_outputs)
        logger.info(f"Web scraping去重后数据量: {len(web_data_deduped)}")
        
        # 3. 处理API数据（包含关键词去重）
        logger.info("开始处理API数据...")
        api_data_deduped = process_api_data()
        logger.info(f"API去重后数据量: {len(api_data_deduped)}")
        
        # 4. 检查web数据和API数据之间的重复
        logger.info("正在检查web数据和API数据之间的重复...")
        combined_data = check_duplicates_between_sources(web_data_deduped, api_data_deduped)
        logger.info(f"合并去重后数据量: {len(combined_data)}")
        
        # 5. 保存最终结果
        output_file = 'data/processed/final_combined_data.csv'
        combined_data.to_csv(output_file, index=False)
        logger.info(f"已保存最终结果到: {output_file}")
        
        # 6. 打印详细的统计信息
        logger.info("\n数据处理统计:")
        logger.info(f"1. ResearchOutputs.xlsx数据量: {len(research_outputs)}")
        logger.info(f"2. Web scraping原始数据量: {len(web_data)}")
        logger.info(f"3. Web scraping去重后数据量: {len(web_data_deduped)}")
        logger.info(f"4. API去重后数据量: {len(api_data_deduped)}")
        logger.info(f"5. 最终合并去重后数据量: {len(combined_data)}")
        
        logger.info("数据处理完成！")
        
    except Exception as e:
        logger.error(f"数据处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    process_data() 