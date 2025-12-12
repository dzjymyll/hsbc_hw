import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

load_dotenv()


class ArchitectureAnalyzer:
    """分析代码仓库的架构"""

    def __init__(self, parsed_code_path: str):
        with open(parsed_code_path, 'r', encoding='utf-8') as f:
            self.parsed_data = json.load(f)

        self.architecture_info = self._analyze_architecture()

    def _analyze_architecture(self) -> Dict[str, Any]:
        """分析代码仓库架构"""
        files = self.parsed_data.get('files', [])

        analysis = {
            'tech_stack': {
                'frameworks': set(),
                'databases': set(),
                'libraries': set(),
                'patterns': set()
            },
            'module_structure': {},
            'api_endpoints': [],
            'data_models': [],
            'entry_points': [],
            'dependencies': {}
        }

        # 分析每个文件
        for file_info in files:
            file_path = file_info['file_path']

            # 技术栈分析
            self._analyze_tech_stack(file_info, analysis['tech_stack'])

            # 模块结构分析
            self._analyze_module_structure(file_path, file_info, analysis['module_structure'])

            # API端点分析
            self._analyze_api_endpoints(file_info, analysis['api_endpoints'])

            # 数据模型分析
            self._analyze_data_models(file_info, analysis['data_models'])

            # 入口点分析
            self._analyze_entry_points(file_path, file_info, analysis['entry_points'])

        # 转换为可序列化的格式
        analysis['tech_stack'] = {
            'frameworks': list(analysis['tech_stack']['frameworks']),
            'databases': list(analysis['tech_stack']['databases']),
            'libraries': list(analysis['tech_stack']['libraries']),
            'patterns': list(analysis['tech_stack']['patterns'])
        }

        return analysis

    def _analyze_tech_stack(self, file_info: Dict, tech_stack: Dict):
        """分析技术栈"""
        imports = file_info.get('imports', [])

        for imp in imports:
            module = imp.get('module', '')

            # 框架检测
            if any(framework in module for framework in ['fastapi', 'flask', 'django', 'starlette']):
                tech_stack['frameworks'].add('FastAPI' if 'fastapi' in module else
                                           'Flask' if 'flask' in module else
                                           'Django' if 'django' in module else module)

            # 数据库检测
            if any(db in module for db in ['sqlalchemy', 'sqlmodel', 'pymongo', 'redis', 'aioredis']):
                tech_stack['databases'].add('SQLAlchemy' if 'sqlalchemy' in module else
                                          'SQLModel' if 'sqlmodel' in module else
                                          'MongoDB' if 'pymongo' in module else
                                          'Redis' if 'redis' in module else module)

            # 常用库检测
            common_libs = ['pydantic', 'jinja2', 'httpx', 'requests', 'aiohttp', 'celery']
            for lib in common_libs:
                if lib in module:
                    tech_stack['libraries'].add(lib)

        # 从类基类检测模式
        for cls in file_info.get('classes', []):
            bases = cls.get('bases', [])
            if any('BaseModel' in base for base in bases):
                tech_stack['patterns'].add('Pydantic Models')
            if any('DeclarativeBase' in base or 'Base' in base for base in bases):
                tech_stack['patterns'].add('SQLAlchemy ORM')

    def _analyze_module_structure(self, file_path: str, file_info: Dict, module_structure: Dict):
        """分析模块结构"""
        dir_path = os.path.dirname(file_path)

        if dir_path not in module_structure:
            module_structure[dir_path] = {
                'files': [],
                'functions': 0,
                'classes': 0,
                'api_endpoints': 0
            }

        module_structure[dir_path]['files'].append(os.path.basename(file_path))
        module_structure[dir_path]['functions'] += len(file_info.get('functions', []))
        module_structure[dir_path]['classes'] += len(file_info.get('classes', []))

        # 统计API端点
        for func in file_info.get('functions', []):
            if func.get('is_api_endpoint'):
                module_structure[dir_path]['api_endpoints'] += 1

    def _analyze_api_endpoints(self, file_info: Dict, api_endpoints: List):
        """分析API端点"""
        for func in file_info.get('functions', []):
            if func.get('is_api_endpoint'):
                endpoint_info = {
                    'name': func['name'],
                    'file': file_info['file_path'],
                    'decorators': func.get('decorators', []),
                    'is_async': func.get('is_async', False),
                    'line_number': func.get('line_start')
                }

                # 从装饰器提取HTTP方法和路径
                for decorator in func.get('decorators', []):
                    if '.get' in decorator:
                        endpoint_info['method'] = 'GET'
                    elif '.post' in decorator:
                        endpoint_info['method'] = 'POST'
                    elif '.put' in decorator:
                        endpoint_info['method'] = 'PUT'
                    elif '.delete' in decorator:
                        endpoint_info['method'] = 'DELETE'

                    # 提取路径（简化版）
                    if '"' in decorator or "'" in decorator:
                        import re
                        path_match = re.search(r'["\'](/[^"\']*)["\']', decorator)
                        if path_match:
                            endpoint_info['path'] = path_match.group(1)

                api_endpoints.append(endpoint_info)

    def _analyze_data_models(self, file_info: Dict, data_models: List):
        """分析数据模型"""
        for cls in file_info.get('classes', []):
            class_name = cls['name']
            bases = cls.get('bases', [])

            # 检测是否是数据模型
            is_data_model = any('BaseModel' in base or 'Base' in base or 'Model' in base
                              for base in bases)

            if is_data_model:
                model_info = {
                    'name': class_name,
                    'file': file_info['file_path'],
                    'line_number': cls.get('line_start'),
                    'methods': len(cls.get('methods', [])),
                    'has_validation': any('validator' in method['name'].lower()
                                        for method in cls.get('methods', []))
                }
                data_models.append(model_info)

    def _analyze_entry_points(self, file_path: str, file_info: Dict, entry_points: List):
        """分析入口点"""
        filename = os.path.basename(file_path)

        # 常见的入口点文件
        if filename in ['main.py', 'app.py', '__main__.py', 'run.py']:
            entry_points.append({
                'file': file_path,
                'functions': [f['name'] for f in file_info.get('functions', []) if f.get('is_api_endpoint')],
                'total_functions': len(file_info.get('functions', []))
            })

    def get_architecture_summary(self) -> Dict[str, Any]:
        """获取架构摘要"""
        tech_stack = self.architecture_info['tech_stack']

        summary = {
            'framework': tech_stack['frameworks'][0] if tech_stack['frameworks'] else 'Unknown',
            'database': tech_stack['databases'][0] if tech_stack['databases'] else 'None detected',
            'main_libraries': tech_stack['libraries'][:5],
            'design_patterns': tech_stack['patterns'],
            'total_modules': len(self.architecture_info['module_structure']),
            'total_api_endpoints': len(self.architecture_info['api_endpoints']),
            'total_data_models': len(self.architecture_info['data_models']),
            'main_entry_points': [ep['file'] for ep in self.architecture_info['entry_points'][:3]]
        }

        return summary

    def get_detailed_analysis(self) -> Dict[str, Any]:
        """获取详细分析"""
        return self.architecture_info


class DesignRequirementGenerator:
    """生成设计需求"""

    def __init__(self, architecture_summary: Dict):
        self.architecture_summary = architecture_summary

        # 常见的设计需求模板
        self.requirement_templates = [
            "如何为该系统添加{feature}功能？",
            "如何扩展系统以支持{feature}？",
            "如何优化现有的{aspect}？",
            "如何实现{feature}模块？",
            "如何重构{component}以改善{quality}？"
        ]

        # 常见功能特性
        self.common_features = [
            "用户认证", "权限管理", "日志记录", "缓存", "消息队列",
            "文件上传", "API文档", "监控告警", "数据导出", "搜索功能",
            "实时通知", "任务调度", "第三方集成", "支付功能", "数据分析"
        ]

    def generate_requirements(self, count: int = 5) -> List[Dict[str, str]]:
        """生成设计需求列表"""
        requirements = []
        framework = self.architecture_summary['framework']

        for i in range(min(count, len(self.common_features))):
            feature = self.common_features[i]

            requirement = {
                'id': f'req_{i+1:03d}',
                'text': f"如何基于{framework}架构添加{feature}功能？",
                'feature': feature,
                'context': {
                    'framework': framework,
                    'has_database': 'None detected' not in self.architecture_summary['database'],
                    'has_api': self.architecture_summary['total_api_endpoints'] > 0
                }
            }
            requirements.append(requirement)

        return requirements


class DesignSolutionGenerator:
    """生成设计方案"""

    def __init__(self, api_key: str, architecture_analyzer: ArchitectureAnalyzer):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.model = "glm-4"
        self.architecture_analyzer = architecture_analyzer
        self.architecture_info = architecture_analyzer.get_detailed_analysis()
        self.architecture_summary = architecture_analyzer.get_architecture_summary()

    def generate_design_solution(self, requirement: Dict) -> Dict[str, Any]:
        """为需求生成设计方案"""

        # 构建详细的系统提示词
        system_prompt = self._build_system_prompt()

        # 构建用户提示词
        user_prompt = self._build_user_prompt(requirement)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,  # 稍高的温度以获得更多创意
                max_tokens=1200,  # 更长的响应
                timeout=45
            )

            full_response = response.choices[0].message.content.strip()

            # 解析响应，提取推理trace
            solution, reasoning_trace = self._parse_response(full_response)

            # 构建完整的解决方案
            design_solution = {
                'requirement_id': requirement['id'],
                'requirement_text': requirement['text'],
                'feature': requirement['feature'],
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'solution': solution,
                'reasoning_trace': reasoning_trace,
                'architecture_context': {
                    'framework': self.architecture_summary['framework'],
                    'database': self.architecture_summary['database'],
                    'relevant_modules': self._find_relevant_modules(requirement['feature'])
                }
            }

            return design_solution

        except Exception as e:
            print(f"生成设计方案失败: {e}")
            return self._generate_fallback_solution(requirement)

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个经验丰富的软件架构师，擅长基于现有系统架构设计扩展方案。

你的任务是根据现有代码仓库的架构信息，为给定的需求设计一个可行的技术方案。

请按照以下结构组织你的回答：

## 设计方案
[详细的技术设计方案，包括：架构决策、模块设计、接口设计、数据模型变更等]

## 推理过程
[分步骤解释你的设计决策，包括：
1. 分析现有架构的哪些部分可以复用
2. 为什么选择特定的技术方案
3. 如何最小化对现有代码的影响
4. 考虑的性能、安全、可维护性因素
]

## 实施步骤
[具体的实施步骤，包括代码示例、文件结构建议等]

请确保方案：
1. 基于现有的技术栈和架构
2. 保持代码的一致性和可维护性
3. 提供具体的技术细节
4. 考虑实际实施的可行性"""

    def _build_user_prompt(self, requirement: Dict) -> str:
        """构建用户提示词"""
        architecture_summary = self.architecture_summary
        architecture_info = self.architecture_info

        prompt = f"""# 设计需求
{requirement['text']}

# 现有系统架构分析

## 技术栈
- 主要框架: {architecture_summary['framework']}
- 数据库: {architecture_summary['database']}
- 主要库: {', '.join(architecture_summary['main_libraries'])}
- 设计模式: {', '.join(architecture_summary['design_patterns'])}

## 模块结构
"""

        # 添加模块信息
        for module_path, module_info in list(architecture_info['module_structure'].items())[:5]:
            prompt += f"- {module_path}: {module_info['files']}个文件, {module_info['functions']}个函数, {module_info['api_endpoints']}个API端点\n"

        prompt += f"\n## API端点 (共{len(architecture_info['api_endpoints'])}个)"
        for endpoint in architecture_info['api_endpoints'][:3]:
            prompt += f"\n- {endpoint.get('method', 'UNKNOWN')} {endpoint.get('path', 'unknown')} ({endpoint['name']})"

        prompt += f"\n\n## 数据模型 (共{len(architecture_info['data_models'])}个)"
        for model in architecture_info['data_models'][:3]:
            prompt += f"\n- {model['name']} ({model['file']})"

        prompt += f"""

## 入口点
"""
        for entry_point in architecture_info['entry_points'][:2]:
            prompt += f"- {entry_point['file']}\n"

        prompt += """

请基于以上架构信息，为设计需求提供详细的技术方案。"""

        return prompt

    def _parse_response(self, response: str) -> tuple:
        """解析LLM响应，分离设计方案和推理trace"""
        # 简单解析：根据标题分割
        sections = {
            '设计方案': '',
            '推理过程': '',
            '实施步骤': ''
        }

        current_section = None
        lines = response.split('\n')

        for line in lines:
            line_stripped = line.strip()

            # 检测章节标题
            if line_stripped.startswith('##'):
                for section in sections.keys():
                    if section in line_stripped:
                        current_section = section
                        break
            elif current_section:
                sections[current_section] += line + '\n'

        # 如果解析失败，使用简单分割
        if not any(sections.values()):
            # 尝试找到"推理"相关的段落
            if '推理' in response or '原因' in response or '考虑' in response:
                parts = response.split('\n\n')
                if len(parts) >= 2:
                    sections['设计方案'] = parts[0]
                    sections['推理过程'] = '\n\n'.join(parts[1:])
                else:
                    sections['设计方案'] = response
                    sections['推理过程'] = '自动生成的推理过程...'
            else:
                sections['设计方案'] = response
                sections['推理过程'] = '基于现有架构分析生成的设计决策。'

        solution = sections['设计方案'].strip()
        reasoning = sections['推理过程'].strip()

        return solution, reasoning

    def _find_relevant_modules(self, feature: str) -> List[str]:
        """查找与功能相关的现有模块"""
        relevant_modules = []
        module_structure = self.architecture_info['module_structure']

        # 简单的关键词匹配
        feature_lower = feature.lower()

        for module_path, module_info in module_structure.items():
            module_name = os.path.basename(module_path) if module_path else ''

            # 检查模块是否可能相关
            if ('auth' in feature_lower and 'auth' in module_name) or \
               ('user' in feature_lower and 'user' in module_name) or \
               ('api' in feature_lower and 'api' in module_name):
                relevant_modules.append(module_path)

        return relevant_modules[:3]  # 返回最多3个相关模块

    def _generate_fallback_solution(self, requirement: Dict) -> Dict[str, Any]:
        """生成备用解决方案"""
        framework = self.architecture_summary['framework']
        feature = requirement['feature']

        return {
            'requirement_id': requirement['id'],
            'requirement_text': requirement['text'],
            'feature': feature,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'solution': f"基于{framework}架构，建议采用以下方案实现{feature}功能：\n1. 创建新的模块处理{feature}相关逻辑\n2. 扩展现有数据模型以支持{feature}需求\n3. 添加相应的API端点\n4. 确保与现有系统的集成",
            'reasoning_trace': [
                "1. 分析现有架构，确定扩展点",
                f"2. 考虑到系统使用{framework}，遵循其最佳实践",
                "3. 最小化对现有代码的修改",
                "4. 确保新功能的可维护性和可测试性"
            ],
            'architecture_context': {
                'framework': framework,
                'database': self.architecture_summary['database'],
                'relevant_modules': []
            }
        }


class Scene2Pipeline:
    """场景2完整管道"""

    def __init__(self, parsed_code_path: str, api_key: Optional[str] = None):
        self.parsed_code_path = parsed_code_path
        self.api_key = api_key

        print("初始化场景2管道...")

        # 1. 架构分析
        print("步骤1: 分析代码架构")
        self.architecture_analyzer = ArchitectureAnalyzer(parsed_code_path)
        self.architecture_summary = self.architecture_analyzer.get_architecture_summary()

        # 2. 需求生成
        print("步骤2: 生成设计需求")
        self.requirement_generator = DesignRequirementGenerator(self.architecture_summary)

        # 3. 方案生成
        if api_key:
            print("步骤3: 初始化设计生成器")
            self.solution_generator = DesignSolutionGenerator(api_key, self.architecture_analyzer)
        else:
            print("⚠ 未提供API密钥，只能生成需求，无法生成设计方案")
            self.solution_generator = None

    def run(self, num_requirements: int = 3) -> Dict[str, Any]:
        """运行完整管道"""
        results = {
            'architecture_analysis': self.architecture_summary,
            'requirements': [],
            'design_solutions': [],
            'metadata': {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_requirements': 0,
                'total_solutions': 0,
                'has_llm': self.solution_generator is not None
            }
        }

        # 生成需求
        requirements = self.requirement_generator.generate_requirements(num_requirements)
        results['requirements'] = requirements
        results['metadata']['total_requirements'] = len(requirements)

        print(f"生成 {len(requirements)} 个设计需求")

        # 为每个需求生成设计方案
        if self.solution_generator:
            print("开始生成设计方案...")

            for i, requirement in enumerate(requirements):
                print(f"  处理需求 {i+1}/{len(requirements)}: {requirement['feature']}")

                solution = self.solution_generator.generate_design_solution(requirement)
                results['design_solutions'].append(solution)

                time.sleep(1)  # 避免请求过快

            results['metadata']['total_solutions'] = len(results['design_solutions'])
            print(f"生成 {len(results['design_solutions'])} 个设计方案")

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str = "../data"):
        """保存结果"""
        output_path = Path(output_dir) / "scene2_design_solutions.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"场景2结果保存到: {output_path}")

        # 同时保存为更易读的格式
        readable_path = Path(output_dir) / "scene2_design_solutions_readable.md"
        self._save_readable_format(results, readable_path)

        return output_path

    def _save_readable_format(self, results: Dict[str, Any], output_path: Path):
        """保存为易读的Markdown格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 场景2: 架构设计方案生成\n\n")

            f.write("## 架构分析摘要\n")
            arch = results['architecture_analysis']
            f.write(f"- **框架**: {arch['framework']}\n")
            f.write(f"- **数据库**: {arch['database']}\n")
            f.write(f"- **主要库**: {', '.join(arch['main_libraries'])}\n")
            f.write(f"- **API端点总数**: {arch['total_api_endpoints']}\n")
            f.write(f"- **数据模型总数**: {arch['total_data_models']}\n\n")

            f.write("## 设计需求\n")
            for req in results['requirements']:
                f.write(f"### {req['id']}: {req['feature']}\n")
                f.write(f"{req['text']}\n\n")

            if results['design_solutions']:
                f.write("## 设计方案\n")
                for solution in results['design_solutions']:
                    f.write(f"### {solution['requirement_id']}: {solution['feature']}\n")
                    f.write(f"**需求**: {solution['requirement_text']}\n\n")

                    f.write("#### 设计方案\n")
                    f.write(f"{solution['solution']}\n\n")

                    f.write("#### 推理过程\n")
                    if isinstance(solution['reasoning_trace'], list):
                        for step in solution['reasoning_trace']:
                            f.write(f"- {step}\n")
                    else:
                        f.write(f"{solution['reasoning_trace']}\n")

                    f.write("\n---\n\n")

        print(f"可读格式保存到: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("场景2: 架构设计方案生成")
    print("=" * 60)

    # 配置
    PARSED_CODE_PATH = "../data/parsed_code.json"
    API_KEY = os.getenv("ZHIPUAI_API_KEY")

    if not os.path.exists(PARSED_CODE_PATH):
        print(f"错误: 找不到解析的代码文件 {PARSED_CODE_PATH}")
        print("请先运行代码解析器 (code_parser.py)")
        return

    # 创建管道
    pipeline = Scene2Pipeline(PARSED_CODE_PATH, API_KEY)

    # 运行管道
    print("\n开始运行场景2管道...")
    results = pipeline.run(num_requirements=3)

    # 保存结果
    output_path = pipeline.save_results(results)

    # 打印摘要
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

    arch = results['architecture_analysis']
    print(f"架构分析: {arch['framework']} + {arch['database']}")
    print(f"生成 {len(results['requirements'])} 个设计需求")
    print(f"生成 {len(results['design_solutions'])} 个设计方案")

    if results['design_solutions']:
        print("\n示例设计方案:")
        solution = results['design_solutions'][0]
        print(f"需求: {solution['feature']}")
        print(f"方案预览: {solution['solution'][:150]}...")

    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
