import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
import tiktoken
import sys
import time
# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field, SecretStr
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain.chains import TransformChain, SequentialChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import yaml

# ==================== 配置加载 ====================

def load_config(config_file="new_create_translation_issue.yaml"):
    """从YAML文件加载配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('translation_agent', {})
    except FileNotFoundError:
        print(f"配置文件 {config_file} 不存在")
        raise
    except yaml.YAMLError as e:
        print(f"解析配置文件时发生错误: {e}")
        raise

# 加载配置
_config = load_config()

# ==================== 配置常量 ====================

BACKEND_TYPE = _config.get('backend', {}).get('type', 'siliconflow')
SILICONFLOW_API_KEY = _config.get('backend', {}).get('siliconflow', {}).get('api_key', '')
SILICONFLOW_API_BASE = _config.get('backend', {}).get('siliconflow', {}).get('api_base', 'https://api.siliconflow.cn/v1')
OLLAMA_BASE_URL = _config.get('backend', {}).get('ollama', {}).get('base_url', 'http://localhost:11434')
MODEL_NAME = _config.get('model', {}).get('name', 'Qwen/Qwen3-8B')
MODEL_TEMPERATURE = _config.get('model', {}).get('temperature', 0.1)
MODEL_MAX_RETRY = _config.get('model', {}).get('max_retry', 5)
MODEL_MAX_RETRY_OLLAMA = _config.get('model', {}).get('max_retry_ollama', 1)
PROCESSING_MAX_WORKERS = _config.get('processing', {}).get('max_workers', 8)
SINGLE_FILE_TIMEOUT = _config.get('processing', {}).get('single_file_timeout', 180)
TOTAL_SUMMARY_TIMEOUT = _config.get('processing', {}).get('total_summary_timeout', 300)
LOGGING_LEVEL = _config.get('logging', {}).get('level', 'INFO')

# 配置日志
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL.upper()))
logger = logging.getLogger(__name__)

# ==================== 数据模型定义 ====================

class SingleFileSummary(BaseModel):
    """单个文件摘要的结构化输出"""
    file_path: str = Field(description="文件路径", default="")
    change_type: Literal["仅涉及标点符号的修改", "涉及到中英文文本内容的修改", "涉及到代码内容的修改", "涉及到其他内容的修改"] = Field(description="改动类型")
    potential_impact: str = Field(description="改动对其他文件潜在的影响")
    summary: str = Field(description="改动的详细摘要")
    lines_added: int = Field(description="新增行数", default=0)
    lines_deleted: int = Field(description="删除行数", default=0)

class FileChangeInfo(BaseModel):
    """文件改动信息"""
    file_path: str = Field(description="文件路径")
    change_type: Literal["仅涉及标点符号的修改", "涉及到中英文文本内容的修改", "涉及到代码内容的修改", "涉及到其他内容的修改"] = Field(description="改动类型")
    lines_changed: int = Field(description="改动行数")

class TotalSummary(BaseModel):
    """总摘要的结构化输出"""
    total_files_changed: int = Field(description="总共修改的文件数量", default=0)
    total_lines_changed: int = Field(description="总共修改的行数", default=0)
    overall_potential_impact: str = Field(description="整体改动对其他文件潜在的影响")
    overall_summary: str = Field(description="整体改动的详细摘要")
    change_type_list: List[str] = Field(description="所有文件包含的改动种类列表", default=[])
    file_changes: List[FileChangeInfo] = Field(description="每个修改文件的详细信息列表", default=[])

@dataclass
class DiffFileInfo:
    """单个文件的diff信息"""
    file_path: str
    diff_content: str
    lines_added: int
    lines_deleted: int

@dataclass
class ProcessingResult:
    """处理结果"""
    file_summaries: List[SingleFileSummary]
    total_summary: Optional[TotalSummary]
    processed_files: int
    total_files: int
    error: Optional[str] = None

# ==================== Token 统计工具 ====================

class TokenCounter:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        """初始化tokenizer"""
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.warning("无法初始化tokenizer，将不会计算token数量")

    def _encode(self, text: str) -> List[int]:
        """编码文本"""
        if not isinstance(text, str):
            return []
        if self.tokenizer is None:
            # 如果没有tokenizer，使用简单的估算方法
            return [0] * (len(text) // 4)
        try:
            return self.tokenizer.encode(text)
        except Exception as e:
            logger.warning(f"编码文本时发生错误: {e}")
            # 如果编码失败，使用简单的估算方法
            return [0] * (len(text) // 4)

    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self._encode(text))

    def count_prompt(self, prompt: str) -> int:
        """计算prompt的token数量"""
        tokens = self._count_tokens(prompt)
        self.prompt_tokens += tokens
        self.total_tokens += tokens
        return tokens

    def count_completion(self, completion: str) -> int:
        """计算completion的token数量"""
        tokens = self._count_tokens(completion)
        self.completion_tokens += tokens
        self.total_tokens += tokens
        return tokens

    def get_stats(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

# ==================== 工具函数 ====================

class DiffParser:
    """Git Diff 解析器"""
    
    @staticmethod
    def parse_git_diff(diff_content: str) -> List[DiffFileInfo]:
        """
        解析git diff内容，提取每个文件的改动信息
        
        Args:
            diff_content: git diff的原始内容
 
        Returns:
            包含文件路径和对应diff内容的列表
        """

        files = []
        current_file = None
        current_diff = []
        
        lines = diff_content.strip().split('\n')
        
        for line in lines:
            # 匹配文件路径行
            if line.startswith('diff --git'):
                # 保存前一个文件的信息
                if current_file and current_diff:
                    diff_info = DiffParser._create_diff_file_info(current_file, current_diff)
                    if diff_info:
                        files.append(diff_info)
                
                # 提取文件路径
                parts = line.split()
                if len(parts) >= 3:
                    # 格式: diff --git a/path/to/file b/path/to/file
                    file_path = parts[2].replace('a/', '', 1)
                    current_file = file_path
                    current_diff = [line]
                else:
                    current_file = None
                    current_diff = []
            elif current_file:
                current_diff.append(line)
        
        # 添加最后一个文件
        if current_file and current_diff:
            diff_info = DiffParser._create_diff_file_info(current_file, current_diff)
            if diff_info:
                files.append(diff_info)
        
        return files
    
    @staticmethod
    def _create_diff_file_info(file_path: str, diff_lines: List[str]) -> Optional[DiffFileInfo]:
        """创建DiffFileInfo对象"""
        diff_content = '\n'.join(diff_lines)
        lines_added, lines_deleted = DiffParser._count_lines_changed(diff_content)
        
        return DiffFileInfo(
            file_path=file_path,
            diff_content=diff_content,
            lines_added=lines_added,
            lines_deleted=lines_deleted
        )
    
    @staticmethod
    def _count_lines_changed(diff_content: str) -> Tuple[int, int]:
        """统计git diff中改动的行数"""
        lines_added, lines_deleted = 0, 0
        lines = diff_content.strip().split('\n')

        for line in lines:
            # 统计新增行（以+开头，但不是+++）
            if line.startswith('+') and not line.startswith('+++'):
                lines_added += 1
            # 统计删除行（以-开头，但不是---）
            elif line.startswith('-') and not line.startswith('---'):
                lines_deleted += 1

        return lines_added, lines_deleted

# ==================== LangChain 组件 ====================

class LLMFactory:
    """LLM工厂类"""
    
    @staticmethod
    def create_chat_llm(model_name: str = None, base_url: str = None):
        """创建LLM实例"""
        if model_name is None:
            model_name = MODEL_NAME
        if base_url is None:
            base_url = OLLAMA_BASE_URL
            
        if BACKEND_TYPE == "ollama":
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=MODEL_TEMPERATURE
            )
        elif BACKEND_TYPE == "siliconflow":
            return ChatOpenAI(
                model=model_name,
                api_key=SecretStr(SILICONFLOW_API_KEY),
                base_url=SILICONFLOW_API_BASE,
                temperature=MODEL_TEMPERATURE
            )
        else:
            raise ValueError(f"不支持的后端类型: {BACKEND_TYPE}")
    
    @staticmethod
    def create_llm(model_name: str = None, base_url: str = None):
        """创建LLM实例"""
        if model_name is None:
            model_name = MODEL_NAME
        if base_url is None:
            base_url = OLLAMA_BASE_URL
            
        if BACKEND_TYPE == "ollama":
            return Ollama(
                model=model_name,
                base_url=base_url,
                temperature=MODEL_TEMPERATURE
            )
        elif BACKEND_TYPE == "siliconflow":
            return ChatOpenAI(
                model=model_name,
                api_key=SecretStr(SILICONFLOW_API_KEY),
                base_url=SILICONFLOW_API_BASE,
                temperature=MODEL_TEMPERATURE
            )
        else:
            raise ValueError(f"不支持的后端类型: {BACKEND_TYPE}")

class PromptTemplates:
    """提示模板集合"""
    
    @staticmethod
    def get_single_file_prompt() -> ChatPromptTemplate:
        """获取单文件分析提示模板"""
        return ChatPromptTemplate.from_messages([
            ("system", """
你是一个专业的Git维护专家，擅长总结社区文档的改动，请分析以下git diff中单个文件的改动，并生成结构化的摘要。

请仔细分析这个文件的改动，并按照以下要求生成摘要：

1. 改动类型判断（必须选择以下四种之一）：
   - "仅涉及标点符号的修改"：只修改了标点符号（如逗号、分号、括号、感叹号等）
   - "涉及到中英文文本内容的修改"：修改了注释、文档、字符串、变量名等文本内容
   - "涉及到代码内容的修改"：修改了函数逻辑、变量定义、控制流等代码结构
   - "涉及到其他内容的修改"：修改了其他内容，如新增二进制文件、新增依赖库等

2. 潜在影响分析：
   - 分析这个文件的改动可能对其他文件或整体系统造成的影响
   - 考虑依赖关系、接口变化、数据流等
   - 如果是配置文件的修改，考虑对系统配置的影响
   - 如果对其他文件无潜在影响，请说明无潜在影响及原因

3. 详细摘要：
   - 提炼出摘要改动文件所属的板块，并解释板块作用
   - 结合文件名和改动细节，用详细的语言描述具体的改动内容，要求准确全面，且改动内容要做到具体
   - 突出重要的改动点和影响范围，包括修改内容主要针对的对象、文档的分类等
   - 结合文件名、改动类型、潜在影响分析，对摘要做进一步补充

4. 输出格式：
  - 请用中文生成摘要
  - 要求改动类型、潜在影响、改动内容总结都包含在摘要中，不能存在空字段
  - 严格检查你的输出，对“新增”、“删除”、“修改”等字眼要严格检查，确保没有出现语义错误

            """),
            ("human", """
文件路径: {file_path}

Git Diff 内容:
{diff_content}

            """)
        ])
    
    @staticmethod
    def get_total_summary_prompt() -> ChatPromptTemplate:
        """获取总摘要生成提示模板"""
        return ChatPromptTemplate.from_messages([
            ("system", """
你是一个专业的Git维护专家，擅长总结社区文档的改动，请基于以下各个文件的改动摘要，生成整个git diff的总摘要。

请分析所有文件的改动，并生成一个总摘要，要求：

1. 整体潜在影响分析：
   - 逐个总结所有文件的改动内容，并进行详细的列举，尽量涵盖所有修改内容
   - 综合分析所有文件改动对系统的整体影响
   - 考虑文件间的依赖关系和系统架构影响
   - 评估改动的风险等级和影响范围
   - 如果对其他文件无潜在影响，请说明无潜在影响及原因

2. 整体摘要详细列举：
   - 提炼出所有摘要改动文件所属的板块，并解释板块作用
   - 用详细的语言分条概括每个摘要文件的核心内容，需要具体到文件，这一部分要占到最大的篇幅，不要遗漏任何摘要文件的内容
   - 突出重要的改动点，包括修改内容主要针对的对象、文档的分类等
   - 注意：整体摘要需要总结所有文件的内容；整体摘要需要尽可能详细

3. 输出格式：
   - 请用中文生成摘要，整体摘要内容字段务必全面详细
   - 要求整体潜在影响、整体摘要都包含在摘要中，不能存在空字段
   - 整体摘要必须满足以下格式："本次更改涉及到XXX等文件，这些文件分别属于社区中的XXX模块。涉及到XXX的修改，可能会对XXX造成影响。总的来说，这次更改主要是XXX。"
   - 严格检查你的输出，对“新增”、“删除”、“修改”等字眼要严格检查，确保没有出现语义错误

            """),
            ("human", """
各个文件的改动摘要:
{file_changes}

总文件数: {total_files}
            """)
        ])

class SingleFileAnalysisChain:
    """单文件分析任务链"""
    
    def __init__(self, llm: ChatOllama | ChatOpenAI, token_counter: TokenCounter):
        self.llm = llm
        self.token_counter = token_counter
        
        # 创建输出解析器
        self.output_parser = JsonOutputParser(pydantic_object=SingleFileSummary)
        
        # 根据后端类型选择不同的链构建方式
        if BACKEND_TYPE == "ollama":
            self.prompt = PromptTemplates.get_single_file_prompt()
            self.chain = self.prompt | self.llm.with_structured_output(SingleFileSummary)
        else:
            # 为硅基流动平台添加输出格式说明
            format_instructions = """
请以JSON格式输出，包含以下字段：
{{
    "change_type": "改动类型（必须是以下之一：仅涉及标点符号的修改、涉及到中英文文本内容的修改、涉及到代码内容的修改、涉及到其他内容的修改）",
    "potential_impact": "改动对其他文件潜在的影响",
    "summary": "改动的详细摘要"
}}
"""
            # 创建新的prompt模板
            system_template = """
你是一个专业的Git维护专家，擅长总结社区文档的改动，请分析以下git diff中单个文件的改动，并生成结构化的摘要。

请仔细分析这个文件的改动，并按照以下要求生成摘要：

1. 改动类型判断（必须选择以下四种之一）：
   - "仅涉及标点符号的修改"：只修改了标点符号（如逗号、分号、括号、感叹号等）
   - "涉及到中英文文本内容的修改"：修改了注释、文档、字符串、变量名等文本内容
   - "涉及到代码内容的修改"：修改了函数逻辑、变量定义、控制流等代码结构
   - "涉及到其他内容的修改"：修改了其他内容，如新增二进制文件、新增依赖库等

2. 潜在影响分析：
   - 分析这个文件的改动可能对其他文件或整体系统造成的影响
   - 考虑依赖关系、接口变化、数据流等
   - 如果是配置文件的修改，考虑对系统配置的影响
   - 如果对其他文件无潜在影响，请说明无潜在影响及原因

3. 详细摘要：
   - 提炼出摘要改动文件所属的板块，并解释板块作用
   - 结合文件名和改动细节，用详细的语言描述具体的改动内容，要求准确全面，且改动内容要做到具体
   - 突出重要的改动点和影响范围，包括修改内容主要针对的对象、文档的分类等
   - 结合文件名、改动类型、潜在影响分析，对摘要做进一步补充

4. 输出格式：
  - 请用中文生成摘要
  - 要求改动类型、潜在影响、改动内容总结都包含在摘要中，不能存在空字段
  - 严格检查你的输出，对“新增”、“删除”、“修改”等字眼要严格检查，确保没有出现语义错误

{format_instructions}
"""
            human_template = """
文件路径: {file_path}

Git Diff 内容:
{diff_content}
"""
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_template.format(format_instructions=format_instructions)),
                ("human", human_template)
            ])
            self.chain = self.prompt | self.llm | self.output_parser
    
    def analyze(self, diff_file_info: DiffFileInfo) -> Optional[SingleFileSummary]:
        """分析单个文件的改动"""
        max_retry = MODEL_MAX_RETRY_OLLAMA if BACKEND_TYPE == "ollama" else MODEL_MAX_RETRY
        for attempt in range(1, max_retry + 1):
            try:
                # 构造prompt字符串
                prompt_args = {
                    "file_path": diff_file_info.file_path,
                    "diff_content": diff_file_info.diff_content
                }
                try:
                    messages = self.prompt.format_messages(**prompt_args)
                    if messages and len(messages) > 0:
                        message = messages[0]
                        if hasattr(message, 'content') and message.content:
                            prompt_str = str(message.content)
                            if prompt_str:
                                self.token_counter.count_prompt(prompt_str)
                except Exception as e:
                    logger.warning(f"格式化prompt时发生错误: {e}")
                
                # 使用线程池执行器为每个分析任务添加超时控制
                with ThreadPoolExecutor(max_workers=1) as timeout_executor:
                    invoke_args = {
                        "file_path": diff_file_info.file_path,
                        "diff_content": diff_file_info.diff_content,
                        "lines_added": diff_file_info.lines_added,
                        "lines_deleted": diff_file_info.lines_deleted
                    }
                    if BACKEND_TYPE != "ollama":
                        # 为 SiliconFlow 添加 response_format 参数
                        invoke_args["response_format"] = {"type": "json_object"}
                    
                    # 提交任务并设置超时
                    future = timeout_executor.submit(self.chain.invoke, invoke_args)
                    try:
                        result = future.result(timeout=SINGLE_FILE_TIMEOUT)
                    except TimeoutError:
                        logger.error(f"分析文件 {diff_file_info.file_path} 超时（{SINGLE_FILE_TIMEOUT}秒），第{attempt}次尝试")
                        if attempt < max_retry:
                            continue
                        else:
                            logger.error(f"分析文件 {diff_file_info.file_path} 连续{max_retry}次均超时，放弃。")
                            return None
                # 统计completion token
                if isinstance(result, (dict, SingleFileSummary)):
                    # 如果是dict（来自JsonOutputParser），转换为SingleFileSummary
                    if isinstance(result, dict):
                        result = SingleFileSummary(**result)
                    try:
                        if result and hasattr(result, 'summary'):
                            summary = result.summary
                            if summary:
                                completion_str = str(summary)
                                if completion_str:
                                    self.token_counter.count_completion(completion_str)
                    except Exception as e:
                        logger.warning(f"计算completion tokens时发生错误: {e}")
                    # 将result的lines_added和lines_deleted重新赋值为准确值，避免出现错误
                    result.file_path = diff_file_info.file_path
                    result.lines_added = diff_file_info.lines_added
                    result.lines_deleted = diff_file_info.lines_deleted
                    return result
                else:
                    logger.error(f"分析文件 {diff_file_info.file_path} 时返回类型错误: {type(result)}，第{attempt}次尝试")
            except Exception as e:
                err_str = str(e)
                # 检查是否为HTTP错误（如404、5xx），常见关键字有status code、HTTP、response等
                is_http_error = False
                for code in ["404", "500", "502", "503", "504"]:
                    if code in err_str:
                        is_http_error = True
                        break
                if ("status code" in err_str or "HTTP" in err_str or "response" in err_str) and any(code in err_str for code in ["404", "500", "502", "503", "504"]):
                    is_http_error = True
                if is_http_error:
                    logger.error(f"分析文件 {diff_file_info.file_path} 时发生HTTP错误: {e}，第{attempt}次尝试，10秒后重试...")
                    if attempt < max_retry:
                        time.sleep(10)
                        continue
                else:
                    logger.error(f"分析文件 {diff_file_info.file_path} 时发生错误: {e}，第{attempt}次尝试")
                # 其它异常直接进入下一次重试
            # 如果不是最后一次，等待片刻后重试（可选，防止接口限流）
        logger.error(f"分析文件 {diff_file_info.file_path} 连续{max_retry}次均未获得结构化输出，放弃。")
        return None

class TotalSummaryChain:
    """总摘要生成任务链"""
    
    def __init__(self, llm: ChatOllama | ChatOpenAI, token_counter: TokenCounter):
        self.llm = llm
        self.token_counter = token_counter
        
        # 创建输出解析器
        self.output_parser = JsonOutputParser(pydantic_object=TotalSummary)
        
        # 根据后端类型选择不同的链构建方式
        if BACKEND_TYPE == "ollama":
            self.prompt = PromptTemplates.get_total_summary_prompt()
            self.chain = self.prompt | self.llm.with_structured_output(TotalSummary)
        else:
            # 为硅基流动平台添加输出格式说明
            format_instructions = """
请以JSON格式输出，包含以下字段：
{{
    "overall_potential_impact": "整体改动对其他文件潜在的影响",
    "overall_summary": "整体改动的详细摘要"
}}
"""
            # 创建新的prompt模板
            system_template = """
你是一个专业的Git维护专家，擅长总结社区文档的改动，请基于以下各个文件的改动摘要，生成整个git diff的总摘要。

请分析所有文件的改动，并生成一个总摘要，要求：

1. 整体潜在影响分析：
   - 逐个总结所有文件的改动内容，并进行详细的列举，尽量涵盖所有修改内容
   - 综合分析所有文件改动对系统的整体影响
   - 考虑文件间的依赖关系和系统架构影响
   - 评估改动的风险等级和影响范围
   - 如果对其他文件无潜在影响，请说明无潜在影响及原因

2. 整体摘要详细列举：
   - 提炼出所有摘要改动文件所属的板块，并解释板块作用
   - 用详细的语言分条概括每个摘要文件的核心内容，需要具体到文件，这一部分要占到最大的篇幅，不要遗漏任何摘要文件的内容
   - 突出重要的改动点，包括修改内容主要针对的对象、文档的分类等
   - 注意：整体摘要需要总结所有文件的内容；整体摘要需要尽可能详细

3. 输出格式：
   - 请用中文生成摘要，整体摘要内容字段务必全面详细
   - 要求整体潜在影响、整体摘要都包含在摘要中，不能存在空字段
   - 整体摘要必须满足以下格式："本次更改涉及到XXX等文件，这些文件分别属于社区中的XXX模块。涉及到XXX的修改，可能会对XXX造成影响。总的来说，这次更改主要是XXX。"
   - 严格检查你的输出，对“新增”、“删除”、“修改”等字眼要严格检查，确保没有出现语义错误

{format_instructions}
"""
            human_template = """
各个文件的改动摘要:
{file_changes}

总文件数: {total_files}
"""
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_template.format(format_instructions=format_instructions)),
                ("human", human_template)
            ])
            self.chain = self.prompt | self.llm | self.output_parser
    
    def generate(self, file_summaries: List[SingleFileSummary]) -> Optional[TotalSummary]:
        """生成总摘要"""
        try:
            total_files = len(file_summaries)
            total_lines = sum(s.lines_added + s.lines_deleted for s in file_summaries)
            file_changes_info = []
            # 收集所有改动类型
            all_change_types = list(set(s.change_type for s in file_summaries))
            
            for summary in file_summaries:
                file_changes_info.append({
                    'file_path': summary.file_path,
                    'change_type': summary.change_type,
                    'potential_impact': summary.potential_impact,
                    'summary': summary.summary
                })
            
            # 构造prompt字符串
            prompt_args = {
                "file_changes": json.dumps(file_changes_info, ensure_ascii=False, indent=2),
                "total_files": total_files
            }
            try:
                messages = self.prompt.format_messages(**prompt_args)
                if messages and len(messages) > 0:
                    message = messages[0]
                    if hasattr(message, 'content') and message.content:
                        prompt_str = str(message.content)
                        if prompt_str:
                            self.token_counter.count_prompt(prompt_str)
            except Exception as e:
                logger.warning(f"格式化prompt时发生错误: {e}")
            
            # 使用线程池执行器为总摘要生成添加超时控制
            with ThreadPoolExecutor(max_workers=1) as timeout_executor:
                invoke_args = {
                    "file_changes": json.dumps(file_changes_info, ensure_ascii=False, indent=2),
                    "total_files": total_files,
                    "total_lines": total_lines
                }
                if BACKEND_TYPE != "ollama":
                    # 为 SiliconFlow 添加 response_format 参数
                    invoke_args["response_format"] = {"type": "json_object"}
                
                # 提交任务并设置超时
                future = timeout_executor.submit(self.chain.invoke, invoke_args)
                try:
                    result = future.result(timeout=TOTAL_SUMMARY_TIMEOUT)
                except TimeoutError:
                    logger.error(f"生成总摘要超时（{TOTAL_SUMMARY_TIMEOUT}秒），放弃生成总摘要")
                    return None
            
            # 处理结果
            if isinstance(result, (dict, TotalSummary)):
                # 如果是dict（来自JsonOutputParser），转换为TotalSummary
                if isinstance(result, dict):
                    result = TotalSummary(**result)
                try:
                    if result and hasattr(result, 'overall_summary'):
                        summary = result.overall_summary
                        if summary:
                            completion_str = str(summary)
                            if completion_str:
                                self.token_counter.count_completion(completion_str)
                except Exception as e:
                    logger.warning(f"计算completion tokens时发生错误: {e}")
                return TotalSummary(
                    total_files_changed=total_files,
                    total_lines_changed=total_lines,
                    overall_potential_impact=result.overall_potential_impact,
                    overall_summary=result.overall_summary,
                    change_type_list=all_change_types,
                    file_changes=[
                        FileChangeInfo(
                            file_path=summary.file_path,
                            change_type=summary.change_type,
                            lines_changed=summary.lines_added + summary.lines_deleted
                        )
                        for summary in file_summaries
                    ]
                )
            else:
                logger.error(f"生成总摘要时返回类型错误: {type(result)}")
                return None
        except Exception as e:
            logger.error(f"生成总摘要时发生错误: {e}")
            return None

# ==================== 主处理类 ====================

class GitDiffSummarizer:
    """Git Diff 摘要生成器"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        if model_name is None:
            model_name = MODEL_NAME
        if base_url is None:
            base_url = OLLAMA_BASE_URL
        self.token_counter = TokenCounter(model_name)
        self.llm = LLMFactory.create_chat_llm(model_name, base_url)
        self.single_file_chain = SingleFileAnalysisChain(self.llm, self.token_counter)
        self.total_summary_chain = TotalSummaryChain(self.llm, self.token_counter)
    
    def process_git_diff(self, diff_content: str, max_workers: int = None) -> ProcessingResult:
        if max_workers is None:
            max_workers = PROCESSING_MAX_WORKERS
            
        logger.info("开始解析git diff...")
        files = DiffParser.parse_git_diff(diff_content)
        logger.info(f"解析到 {len(files)} 个文件的改动")
        if not files:
            logger.warning("未找到任何文件改动")
            return ProcessingResult(
                file_summaries=[],
                total_summary=None,
                processed_files=0,
                total_files=0,
                error='未找到任何文件改动'
            )
        logger.info("开始并行处理各个文件的改动...")
        file_summaries = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.single_file_chain.analyze, file_info): file_info.file_path
                for file_info in files
            }
            for future in as_completed(future_to_file, timeout=SINGLE_FILE_TIMEOUT + 30):
                file_path = future_to_file[future]
                try:
                    summary = future.result(timeout=5)  # 额外的5秒缓冲时间
                    if summary:
                        file_summaries.append(summary)
                        logger.info(f"完成文件 {file_path} 的摘要生成")
                    else:
                        logger.warning(f"文件 {file_path} 的摘要生成失败")
                except TimeoutError:
                    logger.error(f"文件 {file_path} 的摘要生成超时，跳过该文件")
                    future.cancel()  # 尝试取消超时的任务
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时发生异常: {e}")
        logger.info(f"成功生成 {len(file_summaries)} 个文件的摘要")
        logger.info("开始生成总摘要...")
        total_summary = None
        if file_summaries:
            total_summary = self.total_summary_chain.generate(file_summaries)
            if total_summary:
                logger.info("总摘要生成成功")
            else:
                logger.warning("总摘要生成失败")
        return ProcessingResult(
            file_summaries=file_summaries,
            total_summary=total_summary,
            processed_files=len(file_summaries),
            total_files=len(files)
        )

# ==================== 主函数 ====================

def get_agent_summary(sample_diff):

    summarizer = GitDiffSummarizer()
    result = summarizer.process_git_diff(sample_diff)

    if result.error:
        print(f"错误: {result.error}")
    print("\n=== 单文件摘要 ===")
    for summary in result.file_summaries:
        print(f"文件: {summary.file_path}")
        print(f"改动类型: {summary.change_type}")
        print(f"新增行数: {summary.lines_added}")
        print(f"删除行数: {summary.lines_deleted}")
        print(f"潜在影响: {summary.potential_impact}")
        print(f"摘要: {summary.summary}")
        print("-" * 50)
    print("=== 处理结果 ===")
    print(f"总文件数: {result.total_files}")
    print(f"成功处理文件数: {result.processed_files}")
    if result.total_summary:
        print("\n=== 总摘要 ===")
        total = result.total_summary
        print(f"总文件数: {total.total_files_changed}")
        print(f"总改动行数: {total.total_lines_changed}")
        print(f"改动类型列表: {total.change_type_list}")
        print(f"整体潜在影响: {total.overall_potential_impact}")
        print(f"整体摘要: {total.overall_summary}")
        print("\n=== 文件改动列表 ===")
        for file_change in total.file_changes:
            print(f"- {file_change.file_path}: {file_change.change_type} ({file_change.lines_changed} 行)")
            
    # 输出token统计
    stats = summarizer.token_counter.get_stats()
    print("\n=== Token消耗统计 ===")
    print(f"Prompt tokens: {stats['prompt_tokens']}")
    print(f"Completion tokens: {stats['completion_tokens']}")
    print(f"Total tokens: {stats['total_tokens']}")
    # exit()
    return result

if __name__ == "__main__":
    # 微服务接口逻辑： 传递进来的就是 sample_diff 的内容
    sample_diff = sys.argv[1]
    result = get_agent_summary(sample_diff) 
    print(result)