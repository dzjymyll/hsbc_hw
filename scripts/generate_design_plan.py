import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

load_dotenv()


class EnhancedArchitectureAnalyzer:
    """增强版架构分析器"""

    def __init__(self, parsed_code_path: str):
        with open(parsed_code_path, 'r', encoding='utf-8') as f:
            self.parsed_data = json.load(f)

        self.architecture = self._analyze()

    def _analyze(self) -> Dict[str, Any]:
        """分析架构信息"""
        files = self.parsed_data.get('files', [])

        tech_stack = self._analyze_tech_stack(files)
        components = self._analyze_components(files)
        patterns = self._analyze_design_patterns(files)

        return {
            "framework": self._detect_framework(tech_stack),
            "database": self._detect_database(tech_stack),
            "main_libraries": list(tech_stack.get("libraries", set())),
            "design_patterns": list(patterns),
            "existing_components": components
        }

    def _analyze_tech_stack(self, files: List[Dict]) -> Dict[str, set]:
        """分析技术栈"""
        tech_stack = {
            "frameworks": set(),
            "databases": set(),
            "libraries": set()
        }

        # 常用库的检测映射
        library_patterns = {
            "pydantic": ["pydantic", "basemodel"],
            "jinja2": ["jinja2", "templateresponse"],
            "sqlalchemy": ["sqlalchemy", "session", "declarativebase"],
            "sqlmodel": ["sqlmodel"],
            "fastapi": ["fastapi", "depends", "apirouter"],
            "httpx": ["httpx"],
            "requests": ["requests"],
            "aiohttp": ["aiohttp"],
            "celery": ["celery"],
            "redis": ["redis"],
            "pymongo": ["pymongo"]
        }

        for file_info in files:
            # 从导入语句检测
            for imp in file_info.get('imports', []):
                module = imp.get('module', '').lower()

                for lib_name, patterns in library_patterns.items():
                    for pattern in patterns:
                        if pattern in module:
                            tech_stack["libraries"].add(lib_name)
                            break

            # 从代码内容检测
            code_context = file_info.get('context_snippet', '').lower()
            for lib_name, patterns in library_patterns.items():
                for pattern in patterns:
                    if pattern in code_context:
                        tech_stack["libraries"].add(lib_name)
                        break

        return tech_stack

    def _analyze_components(self, files: List[Dict]) -> Dict[str, List]:
        """分析现有组件"""
        components = {
            "models": [],
            "api_endpoints": [],
            "main_files": []
        }

        for file_info in files:
            file_path = file_info.get('file_path', '')

            # 记录主要文件
            if file_path.endswith('.py') and 'test' not in file_path.lower():
                components["main_files"].append(file_path)

            # 提取数据模型
            for cls in file_info.get('classes', []):
                class_name = cls.get('name', '')
                bases = cls.get('bases', [])

                # 检测是否是数据模型
                is_model = False
                for base in bases:
                    base_lower = base.lower()
                    if 'model' in base_lower or 'base' in base_lower or 'sql' in base_lower:
                        is_model = True
                        break

                if is_model and class_name and class_name not in components["models"]:
                    components["models"].append(class_name)

            # 提取API端点
            for func in file_info.get('functions', []):
                if func.get('is_api_endpoint'):
                    endpoint_name = func.get('name', '')
                    if endpoint_name and endpoint_name not in components["api_endpoints"]:
                        components["api_endpoints"].append(endpoint_name)

        return components

    def _analyze_design_patterns(self, files: List[Dict]) -> set:
        """分析设计模式"""
        patterns = set()

        for file_info in files:
            file_path = file_info.get('file_path', '').lower()

            # 检测依赖注入模式
            for imp in file_info.get('imports', []):
                module = imp.get('module', '').lower()
                if 'depends' in module:
                    patterns.add("Dependency Injection")
                    break

            # 检测Repository模式
            for cls in file_info.get('classes', []):
                for method in cls.get('methods', []):
                    method_name = method.get('name', '').lower()
                    if 'repository' in method_name or 'repo' in method_name:
                        patterns.add("Repository Pattern")
                        break

            # 检测MVC/MVT模式
            if 'template' in file_path or 'view' in file_path:
                patterns.add("MVC Pattern")

        return patterns

    def _detect_framework(self, tech_stack: Dict) -> str:
        """检测主要框架"""
        libraries = tech_stack.get('libraries', set())
        if 'fastapi' in libraries:
            return 'FastAPI'
        elif 'flask' in libraries:
            return 'Flask'
        elif 'django' in libraries:
            return 'Django'
        return 'Unknown'

    def _detect_database(self, tech_stack: Dict) -> str:
        """检测数据库"""
        libraries = tech_stack.get('libraries', set())
        if 'sqlmodel' in libraries:
            return 'SQLModel'
        elif 'sqlalchemy' in libraries:
            return 'SQLAlchemy'
        elif 'pymongo' in libraries:
            return 'MongoDB'
        elif 'redis' in libraries:
            return 'Redis'
        return 'Unknown'

class DesignSampleGenerator:
    """设计样本生成器"""

    def __init__(self, api_key: str, architecture: Dict):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.model = "glm-4"
        self.architecture = architecture

        # 设计需求模板
        self.requirement_templates = [
            {
                "feature": "用户认证",
                "template": "如何为现有的{framework}应用添加用户认证功能？需要支持用户注册、登录和JWT令牌验证。",
                "priority": "high"
            },
            {
                "feature": "权限管理",
                "template": "如何在现有系统上添加基于角色的权限控制（RBAC）功能？需要区分不同用户角色的访问权限。",
                "priority": "high"
            },
            {
                "feature": "日志记录",
                "template": "如何为系统添加结构化的日志记录功能？需要记录API请求、业务操作和错误信息。",
                "priority": "medium"
            },
            {
                "feature": "缓存机制",
                "template": "如何为现有的{framework}应用添加缓存层以提高性能？需要考虑数据库查询缓存和API响应缓存。",
                "priority": "medium"
            },
            {
                "feature": "文件上传",
                "template": "如何扩展系统以支持文件上传功能？需要处理图片、文档等文件类型，并考虑存储和安全。",
                "priority": "medium"
            }
        ]

    def generate_requirements(self, count: int = 3) -> List[Dict]:
        """生成设计需求"""
        requirements = []
        framework = self.architecture["framework"]

        for i in range(min(count, len(self.requirement_templates))):
            template = self.requirement_templates[i]
            requirement = {
                "id": f"req_{i+1:03d}",
                "text": template["template"].format(framework=framework),
                "feature": template["feature"],
                "priority": template["priority"]
            }
            requirements.append(requirement)

        return requirements

    def generate_design_sample(self, requirement: Dict) -> Dict[str, Any]:
        """生成完整的设计样本"""

        # 构建系统提示词
        system_prompt = """你是一个经验丰富的软件架构师。根据给定的需求和技术架构，设计一个详细的技术方案。

请按以下结构提供设计方案：
1. 设计概述：简要说明解决方案的核心思想
2. 技术方案：详细的技术实现方案，包括架构设计、组件设计
3. 具体实施：列出需要新增/修改的文件、数据库变更、API端点变更
4. 注意事项：实施过程中需要注意的关键点

对于技术方案，请尽可能具体，可以包含：
- 需要新增的模块和文件
- 需要修改的现有文件
- 数据库表结构变更
- 新的API端点设计
- 关键的技术选型和理由"""

        # 构建用户提示词
        user_prompt = self._build_design_prompt(requirement)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1200,
                timeout=45
            )

            full_response = response.choices[0].message.content.strip()

            # 解析响应，提取设计解决方案和推理过程
            design_solution, reasoning_trace = self._parse_design_response(full_response)

            # 构建完整样本
            sample = {
                "id": f"design_{requirement['id']}",
                "input": {
                    "requirement": {
                        "text": requirement["text"],
                        "feature": requirement["feature"],
                        "priority": requirement["priority"]
                    },
                    "current_architecture": self.architecture
                },
                "output": {
                    "design_solution": design_solution,
                    "reasoning_trace": reasoning_trace
                },
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": self.model
                }
            }

            return sample

        except Exception as e:
            print(f"生成设计方案失败: {e}")
            return self._generate_fallback_sample(requirement)

    def _build_design_prompt(self, requirement: Dict) -> str:
        """构建设计提示词"""
        arch = self.architecture

        prompt = f"""# 设计需求
{requirement['text']}

# 当前系统架构

## 技术栈
- 主要框架: {arch['framework']}
- 数据库: {arch['database']}
- 主要库: {', '.join(arch['main_libraries'][:5])}
- 检测到的设计模式: {', '.join(arch['design_patterns'])}

## 现有组件
- 数据模型: {', '.join(arch['existing_components']['models'][:5])}
- API端点: {', '.join(arch['existing_components']['api_endpoints'][:5])}
- 主要文件: {', '.join(arch['existing_components']['main_files'][:3])}

# 设计要求
请基于以上架构信息，设计一个可行的技术方案。方案需要：
1. 尽可能复用现有的技术栈和组件
2. 最小化对现有代码的影响
3. 提供具体的实施步骤
4. 考虑性能、安全和可维护性

请提供详细的设计方案："""

        return prompt

    def _parse_design_response(self, response: str) -> tuple:
        """解析设计响应"""
        # 尝试提取推理过程（通常在最后部分）
        reasoning_keywords = ['考虑', '因为', '原因', '选择', '决策', '权衡', '理由']

        # 简单分割：前80%作为设计方案，后20%作为推理
        lines = response.split('\n')
        split_point = int(len(lines) * 0.8)

        design_solution = '\n'.join(lines[:split_point]).strip()
        reasoning_trace = '\n'.join(lines[split_point:]).strip()

        # 如果推理部分太短，重新提取
        if len(reasoning_trace) < 100:
            # 查找包含推理关键词的段落
            reasoning_lines = []
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in reasoning_keywords):
                    # 取当前行及后续2行
                    reasoning_lines.extend(lines[i:i+3])

            if reasoning_lines:
                reasoning_trace = '\n'.join(reasoning_lines).strip()
            else:
                reasoning_trace = "基于现有架构分析和技术栈选择的最优方案。"

        return design_solution, reasoning_trace

    def _generate_fallback_sample(self, requirement: Dict) -> Dict[str, Any]:
        """生成备用样本"""
        arch = self.architecture

        design_solution = f"""## 设计方案：{requirement['feature']}

基于{arch['framework']}和{arch['database']}架构，建议采用以下方案：

### 1. 设计概述
在现有架构基础上添加{requirement['feature']}功能模块。

### 2. 技术方案
- 创建新的模块处理{requirement['feature']}相关逻辑
- 扩展现有数据模型以支持新功能
- 添加相应的API端点
- 确保与现有系统的无缝集成

### 3. 具体实施
需要新增以下文件：
- `src/fastapi_app/{requirement['feature'].lower()}/__init__.py`
- `src/fastapi_app/{requirement['feature'].lower()}/models.py`
- `src/fastapi_app/{requirement['feature'].lower()}/router.py`

需要修改现有文件：
- `src/fastapi_app/models.py` (添加相关模型)
- `src/fastapi_app/app.py` (集成新路由)

### 4. 注意事项
- 保持向后兼容性
- 添加适当的错误处理
- 编写单元测试"""

        reasoning_trace = f"""## 推理过程

1. 分析现有架构：系统使用{arch['framework']}框架和{arch['database']}数据库，现有模型包括{', '.join(arch['existing_components']['models'][:3])}

2. 设计决策：
   - 选择在现有架构基础上扩展，而不是重写
   - 遵循现有代码的组织结构和命名约定
   - 复用现有的数据库连接和配置管理

3. 技术选型理由：
   - 使用{arch['framework']}原生支持的功能
   - 确保新功能与现有组件的兼容性
   - 最小化系统复杂度和维护成本

4. 实施策略：
   - 分阶段实施，先完成核心功能
   - 充分测试确保不影响现有功能
   - 提供清晰的文档和示例"""

        return {
            "id": f"design_fallback_{requirement['id']}",
            "input": {
                "requirement": {
                    "text": requirement["text"],
                    "feature": requirement["feature"],
                    "priority": requirement["priority"]
                },
                "current_architecture": arch
            },
            "output": {
                "design_solution": design_solution,
                "reasoning_trace": reasoning_trace
            },
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "fallback",
                "note": "LLM生成失败，使用备用方案"
            }
        }


class Scene2Pipeline:
    """场景2处理管道"""

    def __init__(self, parsed_code_path: str, api_key: Optional[str] = None):
        self.parsed_code_path = parsed_code_path
        self.api_key = api_key

        print("初始化场景2管道...")

        # 架构分析
        print("步骤1: 分析代码架构")
        self.analyzer = EnhancedArchitectureAnalyzer(parsed_code_path)
        self.architecture = self.analyzer.architecture

        print(f"  框架: {self.architecture['framework']}")
        print(f"  数据库: {self.architecture['database']}")
        print(f"  检测到库: {', '.join(self.architecture['main_libraries'][:5])}")

        # 初始化生成器
        if api_key:
            print("步骤2: 初始化设计生成器")
            self.generator = DesignSampleGenerator(api_key, self.architecture)
        else:
            print("⚠ 未提供API密钥，将无法生成设计方案")
            self.generator = None

    def run(self, num_samples: int = 3) -> List[Dict[str, Any]]:
        """运行管道生成样本"""
        if not self.generator:
            print("错误: 未初始化设计生成器")
            return []

        print(f"\n开始生成 {num_samples} 个设计样本...")

        # 生成需求
        requirements = self.generator.generate_requirements(num_samples)

        # 为每个需求生成设计样本
        samples = []
        for i, requirement in enumerate(requirements):
            print(f"  生成样本 {i+1}/{len(requirements)}: {requirement['feature']}")

            sample = self.generator.generate_design_sample(requirement)
            samples.append(sample)

            # 避免请求过快
            if i < len(requirements) - 1:
                time.sleep(1.5)

        return samples

    def save_samples(self, samples: List[Dict], output_dir: str = "../data"):
        """保存生成的样本"""
        output_path = Path(output_dir) / "scene2_design_samples.json"

        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_samples": len(samples),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "architecture": self.architecture['framework'],
                    "has_llm": self.api_key is not None
                },
                "samples": samples
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 设计样本已保存到: {output_path}")

        # 同时保存为JSONL格式
        jsonl_path = output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"📄 JSONL格式: {jsonl_path}")

        return output_path

    def print_sample_summary(self, samples: List[Dict], num_to_show: int = 2):
        """打印样本摘要"""
        print(f"\n{'='*60}")
        print(f"{'设计样本摘要':^60}")
        print(f"{'='*60}")

        for i, sample in enumerate(samples[:num_to_show]):
            print(f"\n🔹 样本 {i+1}: {sample['input']['requirement']['feature']}")
            print(f"   需求: {sample['input']['requirement']['text'][:80]}...")

            # 显示设计方案预览
            solution = sample['output']['design_solution']
            lines = solution.split('\n')
            if len(lines) > 3:
                preview = '\n'.join(lines[:3])
                print(f"   设计方案预览:\n   {preview[:100]}...")

            # 显示推理过程预览
            reasoning = sample['output']['reasoning_trace']
            if isinstance(reasoning, str) and len(reasoning) > 0:
                reasoning_preview = reasoning[:120] + "..." if len(reasoning) > 120 else reasoning
                print(f"   推理过程: {reasoning_preview}")


def main():
    """主函数"""
    print("=" * 60)
    print("场景2: 架构设计样本生成")
    print("=" * 60)

    # 配置
    PARSED_CODE_PATH = "../data/parsed_code.json"
    API_KEY = os.getenv("ZHIPUAI_API_KEY")

    # 检查文件是否存在
    if not os.path.exists(PARSED_CODE_PATH):
        print(f"错误: 找不到文件 {PARSED_CODE_PATH}")
        print("请先运行代码解析器")
        return

    # 创建管道
    pipeline = Scene2Pipeline(PARSED_CODE_PATH, API_KEY)

    # 运行管道
    samples = pipeline.run(num_samples=3)

    if samples:
        # 保存结果
        output_path = pipeline.save_samples(samples)

        # 打印摘要
        pipeline.print_sample_summary(samples)

        # 统计信息
        print(f"\n📊 生成统计:")
        print(f"  • 总样本数: {len(samples)}")

        feature_count = {}
        for sample in samples:
            feature = sample['input']['requirement']['feature']
            feature_count[feature] = feature_count.get(feature, 0) + 1

        for feature, count in feature_count.items():
            print(f"  • {feature}: {count} 个样本")

        print(f"\n✅ 完成! 每个样本包含:")
        print(f"  • input: requirement + current_architecture")
        print(f"  • output: design_solution + reasoning_trace")

    else:
        print("未生成任何设计样本")


if __name__ == "__main__":
    main()
