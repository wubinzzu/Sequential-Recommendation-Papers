import arxiv
import json
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from zai import ZhipuAiClient

class PaperUpdater:
    def __init__(self, paper_path):
        self.existing_papers = set()
        self.keywords = self.load_keywords()
        self.paper_path = paper_path
        self.client = self.get_paper_classify_agent()

    def load_keywords(self) -> List[str]:
        """加载查询关键词"""
        with open('scripts/keywords.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def get_paper_classify_agent(self):
        """初始化GLM API客户端"""
        key = os.environ.get("ZHIPUAI_API_KEY")
        if not key:
            print("API key Missing")
            exit(0)
        return ZhipuAiClient(api_key=key)# 请替换为您的API Key
        

    def classify_paper_with_llm(self, paper_info: Dict) -> Tuple[bool, str, str]:
        """使用GLM API对论文进行分类"""
        system_prompt = """
你是一位专注于推荐系统领域的资深论文分类专家。

为了更准确地判断论文是否属于“序列推荐”领域，请参考以下背景知识：

序列推荐是指利用用户的历史行为序列，捕捉项目之间的顺序模式、时序动态或短期偏好，从而预测用户下一次可能交互的项目的研究方向。

典型特征包括但不限于：
1. 显式地对用户行为序列进行建模，如使用RNN（GRU4Rec）、Transformer（SASRec, BERT4Rec）、CNN（Caser）或图神经网络（GCN用于序列建模）；
2. 关注时间戳信息、行为的时间顺序或时序动态；
3. 涉及会话推荐，重点在于捕捉会话内的序列依赖关系；
4. 引入注意力机制来捕捉序列中关键项目的影响（如DIN, DIEN等，侧重序列行为建模）；
5. 建模长期兴趣与短期意图的演变。

不属于序列推荐的情况包括：
- 基于大语言模型（LLM）的生成式推荐（Generative Recommendation），例如Tiger、Generative Retrieval等将推荐视为纯文本生成任务的研究；
- 传统协同过滤（如矩阵分解MF），未考虑行为的时间顺序；
- 仅关注社交关系或知识图谱，未涉及行为序列的时序建模；
- 单纯的评分预测，不涉及序列模式挖掘。

如果论文核心在于对用户行为序列进行建模以预测后续行为，请将其归类为“序列推荐”。
"""
        prompt = f"""
请对以下学术论文进行分类。论文信息：

标题：{paper_info['title']}
摘要：{paper_info['summary']}

请判断该论文是否属于序列推荐领域。

请严格按以下格式返回结果：

是否属于序列推荐：是/否

其中冒号前的内容为固定中文字符串，冒号后的内容为分类结果。
不要输出多余信息。
因为后续处理流程会使用 line.split('：')[1].strip()=='是' 来判断是否属于序列推荐。
"""
        try:
            response = self.client.chat.completions.create(
                model="glm-4.5-air",
                messages=[
                    {"role": "system", "content": system_prompt},  # 添加System消息
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10000,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            #print("GLM API返回结果:", result)
            return ("是" == result.split('：')[1].strip())
            
        except Exception as e:
            print(f"GLM API调用失败: {e}")
            return False

    def load_existing_papers(self) -> None:
        """加载已有论文信息用于去重"""
        with open(self.paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        arxiv_pattern = r'arxiv\.org/abs/(\d+\.\d+)'
        self.existing_papers = set(re.findall(arxiv_pattern, content, re.IGNORECASE))

    def query_new_papers(self) -> List[Dict]:
        """查询arXiv最新论文"""
        new_papers = []
        client = arxiv.Client()
        
        for keyword in self.keywords:
            search_query = f'ti:"{keyword}" OR abs:"{keyword}"'
            search = arxiv.Search(
                query=search_query,
                max_results=200,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            try:
                for result in client.results(search):
                    # 日期过滤：只看7天内的
                    if result.published.date() < (datetime.now() - timedelta(days=100)).date():
                        continue
                    
                    arxiv_id = result.entry_id.split('/')[-1]
                    for i in range(1,10):
                        arxiv_id = arxiv_id.replace(f'v{i}','')
                    if arxiv_id in self.existing_papers:
                        continue

                    # 构建论文信息字典，补充 journal_ref 和 comment
                    paper_info = {
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'arxiv_id': arxiv_id,
                        'pdf_url': result.pdf_url,
                        'year': result.published.year,
                        'summary': result.summary,
                        'primary_category': str(result.primary_category),
                        'journal_ref': result.journal_ref,  # 新增
                        'comment': result.comment            # 新增
                    }
                    paper_info['venue'] = str(self.determine_venue(paper_info)).strip()
                    if paper_info['venue'] == 'Arxiv':
                        continue
                    # 使用GLM进行分类
                    is_right = self.classify_paper_with_llm(paper_info)
                    
                    if is_right:
                        new_papers.append(paper_info)
                        self.existing_papers.add(arxiv_id)

                    
            except Exception as e:
                print(f"查询关键词 '{keyword}' 时出错: {e}")
                continue
                
        return new_papers

    def format_paper_entry(self, paper: Dict) -> str:
        """格式化论文条目"""
        year = paper['year']
        venue = paper['venue']
        
        if len(venue.split(' ')) > 1:
            venue,year = venue.split(' ')
        abs_url = paper['pdf_url'].replace('pdf','abs')
        for i in range(1,10):
            abs_url = abs_url.replace(f'v{i}','')
        entry = f"- `{venue}({year})`{paper['title']} **[[PDF]({abs_url})]**\n"
        
        return entry
    def determine_venue(self, paper: Dict) -> str:
        target_venues = [
                "NeurIPS", "ICML", "ICLR", "AAAI", "IJCAI", "TOIS", "NAACL"
                "ACL", "EMNLP", "WSDM", "TMLR", "GenRec", "ICDE"
                "KDD", "WWW", "SIGIR-AP","SIGIR", "TKDE", "TORS", "CIKM", "RecSys",
                "JMLR", "TPAMI", "TIP", "NIPS", "EMNLP"
            ]
        def extract_venue(desc):
            if not desc:
                return 'Arxiv'
            for venue in target_venues:
                if venue.lower() in desc.lower():
                    # 尝试提取年份
                    # 正则逻辑：查找 Venue 名称后的年份
                    year_match = re.search(rf'{venue}.*?((?:19|20)\d{{2}})', desc, re.IGNORECASE)
                    if year_match:
                        return f"{venue} {year_match.group(1)}"
                    # 如果没找到年份，仅返回会议名
                    return venue
            return 'Arxiv'
        """根据论文信息确定会议/期刊"""
        # 1. 检查是否有正式的期刊引用
        if paper.get('journal_ref'):
            venue = extract_venue(paper['journal_ref'].strip())
            if venue != 'Arxiv':
                return venue
        
        # 2. 检查 Comment 字段中的会议信息
        comment = paper.get('comment', '')
        return extract_venue(comment)

    def update_readme(self, new_papers: List[Dict]) -> bool:
        """更新主README.md文件"""
        if not new_papers:
            print("没有发现新论文")
            return False

        # 读取现有README内容
        with open(self.paper_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 找到Generative Recommendation部分
        pattern = r'(### Sequential Recommendation\n)(.*?)(?=\n###|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            print("未找到标题部分")
            return False

        # 构建新内容
        new_entries = []
        for paper in new_papers:
            new_entries.append(self.format_paper_entry(paper))

        updated_section = match.group(1) + ''.join(new_entries) + match.group(2)
        new_content = content.replace(match.group(0), updated_section)

        # 写回文件
        with open(self.paper_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"成功添加 {len(new_papers)} 篇新论文")
        return True

    def main(self):
        """主执行函数"""
        self.load_existing_papers()
        new_papers = self.query_new_papers()
        
        if self.update_readme(new_papers):
            commit_message = f"Auto-update: Add {len(new_papers)} new papers - {datetime.now().strftime('%Y-%m-%d')}"
            print(commit_message)
        else:
            print("无需更新")

if __name__ == "__main__":
    updater = PaperUpdater(paper_path='README.md')
    updater.main()
