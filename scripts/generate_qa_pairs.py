import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
import openai
from dotenv import load_dotenv

load_dotenv()


class CodeQAProcessor:
    """代码问答对处理器"""

    def __init__(self, business_rules_path: str):
        with open(business_rules_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.business_rules = data['rules']

        self.templates = {
            'business_operation': "函数 {name} 的主要业务功能是什么？",
            'api_business_operation': "API端点 {name} 的主要业务作用是什么？",
            'business_entity': "{name} 模型在系统中代表什么业务实体？",
            'code_value_check': "这段代码中实现了什么业务规则验证？",
            'validation_function': "验证函数 {name} 检查什么业务规则？",
            'field_validation': "{name} 字段有什么业务约束？"
        }

    def extract_data_processing(self, rule: Dict) -> Dict[str, Any]:
        """从代码中提取数据处理信息"""
        code_context = rule.get('context', '')
        docstring = rule.get('docstring', '')

        processing_info = {
            'operations': [],
            'database_operations': [],
            'validation_rules': [],
            'data_transformations': [],
            'external_calls': []
        }

        # 1. 分析数据库操作
        db_patterns = {
            'session.add': '数据库添加操作',
            'session.commit': '数据库提交操作',
            'session.query': '数据库查询操作',
            'session.delete': '数据库删除操作',
            'session.update': '数据库更新操作',
            'db.session': '数据库会话操作',
            'get_db_session': '获取数据库会话',
            'create_all': '创建数据库表'
        }

        for pattern, description in db_patterns.items():
            if pattern in code_context:
                processing_info['database_operations'].append(description)
                processing_info['operations'].append(f"数据库操作: {description}")

        # 2. 分析验证规则
        validation_patterns = {
            r'if\s+not\s+': '空值检查',
            r'len\([^)]+\)\s*[<>]=?\s*\d+': '长度验证',
            r'isinstance\([^)]+\)': '类型检查',
            r'assert\s+': '断言验证',
            r'raise\s+': '异常抛出',
            r'Form\(\.\.\.\)': '表单验证',
            r'Field\([^)]*min_length': '最小长度验证',
            r'Field\([^)]*max_length': '最大长度验证',
            r'Field\([^)]*gt\s*=': '大于验证',
            r'Field\([^)]*lt\s*=': '小于验证'
        }

        for pattern, description in validation_patterns.items():
            if re.search(pattern, code_context, re.IGNORECASE):
                processing_info['validation_rules'].append(description)
                processing_info['operations'].append(f"验证规则: {description}")

        # 3. 分析数据转换
        transformation_patterns = {
            r'int\([^)]+\)': '整数转换',
            r'str\([^)]+\)': '字符串转换',
            r'float\([^)]+\)': '浮点数转换',
            r'datetime\.': '时间处理',
            r'json\.': 'JSON处理',
            r'\.lower\(\)': '转小写',
            r'\.upper\(\)': '转大写',
            r'\.strip\(\)': '去除空白'
        }

        for pattern, description in transformation_patterns.items():
            if re.search(pattern, code_context):
                processing_info['data_transformations'].append(description)
                processing_info['operations'].append(f"数据转换: {description}")

        # 4. 分析外部调用
        external_patterns = {
            r'logger\.': '日志记录',
            r'request\.': 'HTTP请求处理',
            r'response\.': 'HTTP响应处理',
            r'RedirectResponse': '重定向响应',
            r'TemplateResponse': '模板响应'
        }

        for pattern, description in external_patterns.items():
            if re.search(pattern, code_context):
                processing_info['external_calls'].append(description)

        # 5. 从文档字符串中提取信息
        if docstring:
            # 简单的关键词提取
            if '验证' in docstring or '检查' in docstring:
                processing_info['validation_rules'].append('文档中描述的验证逻辑')

            if '数据库' in docstring or '存储' in docstring:
                processing_info['database_operations'].append('文档中描述的数据库操作')

        return processing_info

    def extract_code_context(self, rule: Dict) -> Dict[str, Any]:
        """提取代码上下文信息"""
        code_context = rule.get('context', '')

        # 提取相关行数
        lines = code_context.split('\n') if code_context else []
        relevant_lines = []

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                if len(relevant_lines) < 10:  # 最多10行
                    relevant_lines.append(line)

        return {
            'file_path': rule['file'],
            'line_number': rule.get('line_number'),
            'code_snippet': '\n'.join(relevant_lines),
            'has_code': bool(code_context.strip())
        }


class LLMAnswerGenerator:
    """使用LLM生成答案的生成器"""

    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4/"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "glm-4"

    def generate_answer(self, rule: Dict, question: str, data_processing_info: Dict) -> str:
        """使用LLM生成答案（包含数据处理信息）"""

        system_prompt = """你是一个专业的软件架构师和业务分析师。请根据提供的代码上下文和数据处理信息，清晰、专业地回答问题。
        要求：
        1. 答案要基于代码的实际功能
        2. 重点解释数据处理部分（数据库操作、验证规则、数据转换等）
        3. 保持简洁明了
        4. 使用中文回答"""

        user_prompt = self._build_prompt(rule, question, data_processing_info)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=700,
                timeout=30
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成答案失败: {e}")
            return self._generate_fallback_answer(rule, data_processing_info)

    def _build_prompt(self, rule: Dict, question: str, data_processing_info: Dict) -> str:
        """构建提示词（包含数据处理信息）"""
        prompt = f"问题：{question}\n\n"

        # 代码信息
        prompt += f"代码信息：\n- 文件：{rule['file']}\n- 规则类型：{rule['type']}\n"

        name = rule.get('function_name') or rule.get('class_name') or rule.get('endpoint') or rule.get('field')
        if name:
            prompt += f"- 名称：{name}\n"

        if rule.get('line_number'):
            prompt += f"- 行号：{rule['line_number']}\n"

        if rule.get('docstring'):
            prompt += f"- 文档：{rule['docstring'][:200]}...\n"

        # 代码上下文
        if rule.get('context'):
            prompt += f"\n代码上下文：\n```python\n{rule['context'][:800]}\n```\n"

        # 数据处理信息
        prompt += f"\n数据处理分析：\n"

        if data_processing_info['database_operations']:
            db_ops = ', '.join(data_processing_info['database_operations'])
            prompt += f"- 数据库操作：{db_ops}\n"

        if data_processing_info['validation_rules']:
            validations = ', '.join(data_processing_info['validation_rules'])
            prompt += f"- 验证规则：{validations}\n"

        if data_processing_info['data_transformations']:
            transforms = ', '.join(data_processing_info['data_transformations'])
            prompt += f"- 数据转换：{transforms}\n"

        if data_processing_info['external_calls']:
            externals = ', '.join(data_processing_info['external_calls'])
            prompt += f"- 外部调用：{externals}\n"

        if not any(data_processing_info.values()):
            prompt += "- 未检测到明显的数据处理操作\n"

        prompt += "\n请基于以上信息，重点解释数据处理部分，给出详细回答："
        return prompt

    def _generate_fallback_answer(self, rule: Dict, data_processing_info: Dict) -> str:
        """生成备用答案（包含数据处理信息）"""
        rule_type = rule['type']
        name = rule.get('function_name') or rule.get('class_name') or rule.get('endpoint') or '该元素'

        answer = f"{name} 实现了相关的业务逻辑。"

        # 添加数据处理信息
        if data_processing_info['database_operations']:
            db_ops = '、'.join(data_processing_info['database_operations'])
            answer += f"\n\n数据处理部分包含数据库操作：{db_ops}。"

        if data_processing_info['validation_rules']:
            validations = '、'.join(data_processing_info['validation_rules'])
            answer += f"\n包含验证规则：{validations}。"

        if data_processing_info['data_transformations']:
            transforms = '、'.join(data_processing_info['data_transformations'])
            answer += f"\n涉及数据转换：{transforms}。"

        return answer


def generate_qa_pairs():
    """生成问答对（包含数据处理信息）"""
    BUSINESS_RULES_PATH = "../data/business_rules_enhanced.json"
    API_KEY = os.getenv("ZHIPUAI_API_KEY")

    print("开始生成问答对（包含数据处理信息）...")

    processor = CodeQAProcessor(BUSINESS_RULES_PATH)
    llm_generator = LLMAnswerGenerator(API_KEY) if API_KEY else None

    all_qa_pairs = []

    for i, rule in enumerate(processor.business_rules):
        try:
            rule_type = rule['type']

            name = rule.get('function_name') or rule.get('class_name') or rule.get('endpoint') or rule.get('field')
            display_name = name if name else f"规则{i+1}"

            if rule_type in processor.templates:
                question = processor.templates[rule_type].format(name=display_name)
            else:
                question = f"请解释这段代码的业务逻辑？"

            print(f"处理 {i+1}/{len(processor.business_rules)}: {rule_type} - {display_name}")

            # 提取数据处理信息
            data_processing_info = processor.extract_data_processing(rule)

            # 生成答案（包含数据处理信息）
            if llm_generator:
                answer = llm_generator.generate_answer(rule, question, data_processing_info)
                time.sleep(0.5)
            else:
                answer = llm_generator._generate_fallback_answer(rule, data_processing_info) if llm_generator else "无LLM生成的答案"

            # 提取代码上下文
            code_context = processor.extract_code_context(rule)

            # 创建完整的问答对
            qa_pair = {
                'id': f"qa_{i:03d}",
                'rule_type': rule_type,
                'file': rule['file'],
                'line_number': rule.get('line_number'),
                'question': question,
                'answer': answer,

                # 原文的代码
                'original_code': {
                    'file_path': rule['file'],
                    'line_number': rule.get('line_number'),
                    'code_snippet': code_context['code_snippet'],
                    'full_context': rule.get('context', '')[:500]  # 完整上下文前500字符
                },

                # 数据处理信息（核心部分）
                'data_processing': {
                    'summary': {
                        'total_operations': len(data_processing_info['operations']),
                        'has_database_operations': bool(data_processing_info['database_operations']),
                        'has_validation_rules': bool(data_processing_info['validation_rules']),
                        'has_data_transformations': bool(data_processing_info['data_transformations'])
                    },
                    'details': data_processing_info,
                    'detected_patterns': data_processing_info['operations']
                },

                'metadata': {
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'has_llm': llm_generator is not None,
                    'confidence': rule.get('confidence', 'medium')
                }
            }

            all_qa_pairs.append(qa_pair)

        except Exception as e:
            print(f"处理失败: {e}")
            continue

    return all_qa_pairs, llm_generator is not None


def save_results(qa_pairs: List[Dict], use_llm: bool):
    """保存结果"""
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    if use_llm:
        filename = "scene1_qa_with_data_processing_llm.json"
    else:
        filename = "scene1_qa_with_data_processing_rule.json"

    output_path = output_dir / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'total_pairs': len(qa_pairs),
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'has_llm': use_llm,
                'contains_data_processing': True,
                'data_processing_fields': ['original_code', 'data_processing']
            },
            'qa_pairs': qa_pairs
        }, f, indent=2, ensure_ascii=False)

    print(f"保存到: {output_path}")

    # JSONL格式
    jsonl_path = output_path.with_suffix('.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    print(f"JSONL格式: {jsonl_path}")

    return output_path


def print_statistics(qa_pairs: List[Dict]):
    """打印统计信息"""
    print(f"\n生成统计:")
    print(f"  总共生成: {len(qa_pairs)} 个问答对")

    # 规则类型统计
    rule_types = {}
    for qa in qa_pairs:
        rule_type = qa['rule_type']
        rule_types[rule_type] = rule_types.get(rule_type, 0) + 1

    print(f"  规则类型分布:")
    for rule_type, count in rule_types.items():
        print(f"    • {rule_type}: {count}")

    # 数据处理统计
    total_data_ops = 0
    has_db_ops = 0
    has_validation = 0

    for qa in qa_pairs:
        data_processing = qa['data_processing']
        total_data_ops += data_processing['summary']['total_operations']
        if data_processing['summary']['has_database_operations']:
            has_db_ops += 1
        if data_processing['summary']['has_validation_rules']:
            has_validation += 1

    print(f"\n  数据处理统计:")
    print(f"    • 平均每个问答对数据处理操作数: {total_data_ops/len(qa_pairs):.1f}")
    print(f"    • 包含数据库操作的问答对: {has_db_ops}")
    print(f"    • 包含验证规则的问答对: {has_validation}")


def print_examples(qa_pairs: List[Dict], count: int = 2):
    """打印示例（展示数据处理信息）"""
    print(f"\n示例问答对（展示数据处理）:")

    for i, qa in enumerate(qa_pairs[:count]):
        print(f"\n示例 {i+1}: {qa['rule_type']}")
        print(f"  文件: {qa['file']}")
        print(f"  问题: {qa['question']}")

        # 显示代码片段
        if qa['original_code']['code_snippet']:
            code_preview = qa['original_code']['code_snippet'][:80]
            print(f"  代码片段: {code_preview}...")

        # 显示数据处理信息
        data_processing = qa['data_processing']
        if data_processing['detected_patterns']:
            print(f"  数据处理操作: {', '.join(data_processing['detected_patterns'][:3])}")

        # 显示答案前150个字符
        answer_preview = qa['answer']
        if len(answer_preview) > 150:
            answer_preview = answer_preview[:150] + "..."
        print(f"  答案预览: {answer_preview}")


def main():
    """主函数"""
    print("=" * 60)
    print("场景1：问答对生成（包含代码及数据处理）")
    print("=" * 60)

    api_key = os.getenv("ZHIPUAI_API_KEY")
    if api_key:
        print("✓ 使用智谱AI生成答案（包含数据处理分析）")
    else:
        print("⚠ 未找到API密钥，使用规则生成答案")
        print("  提示: 在.env文件中添加 ZHIPUAI_API_KEY=your_key")

    qa_pairs, use_llm = generate_qa_pairs()

    if qa_pairs:
        save_results(qa_pairs, use_llm)
        print_statistics(qa_pairs)
        print_examples(qa_pairs)
        print(f"\n✓ 完成！每个问答对都包含:")
        print(f"  1. 原文代码 (original_code)")
        print(f"  2. 数据处理信息 (data_processing)")
    else:
        print("未生成任何问答对")


if __name__ == "__main__":
    main()
