import os
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class CodeParser:
    """
    解析Python代码文件，提取函数、类、方法等结构化信息
    特别针对FastAPI项目结构优化
    """

    def __init__(self, repo_path: str):
        """
        初始化解析器

        Args:
            repo_path: 代码仓库的根目录路径
        """
        self.repo_path = Path(repo_path)
        self.results = {
            'repo_name': self.repo_path.name,
            'files': [],
            'statistics': {
                'total_files': 0,
                'total_functions': 0,
                'total_classes': 0,
                'total_lines': 0
            }
        }

    def parse_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        解析单个Python文件

        Returns:
            包含文件结构化信息的字典，如果解析失败则返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 使用AST解析Python代码
            tree = ast.parse(content)

            file_info = {
                'file_path': str(file_path.relative_to(self.repo_path)),
                'file_name': file_path.name,
                'absolute_path': str(file_path),
                'lines': content.count('\n') + 1,
                'imports': [],
                'functions': [],
                'classes': [],
                'errors': []
            }

            # 提取导入语句
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info['imports'].append({
                            'module': alias.name,
                            'alias': alias.asname
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        file_info['imports'].append({
                            'module': f"{module}.{alias.name}" if module else alias.name,
                            'alias': alias.asname,
                            'is_from_import': True
                        })

            # 提取函数和类定义
            for node in ast.walk(tree):
                # 提取函数
                if isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function_info(node, content)
                    file_info['functions'].append(func_info)

                # 提取类
                elif isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node, content)
                    file_info['classes'].append(class_info)

            return file_info

        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"警告: 无法解析文件 {file_path}: {e}")
            return None

    def _extract_function_info(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """提取函数的详细信息"""
        # 获取函数签名
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        # 获取装饰器
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.unparse(decorator))
            elif isinstance(decorator, ast.Call):
                decorators.append(ast.unparse(decorator.func))

        # 获取函数体前几行作为上下文
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = min(start_line + 10, node.end_lineno) if hasattr(node, 'end_lineno') else start_line + 10
        function_context = '\n'.join(lines[start_line:end_line])

        # 检查是否是API端点（基于装饰器）
        is_api_endpoint = any(decorator in ['app.get', 'app.post', 'app.put', 'app.delete',
                                            'router.get', 'router.post']
                              for decorator in decorators)

        return {
            'name': node.name,
            'args': args,
            'decorators': decorators,
            'docstring': ast.get_docstring(node),
            'line_start': node.lineno,
            'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            'is_api_endpoint': is_api_endpoint,
            'context_snippet': function_context[:500]  # 限制长度
        }

    def _extract_class_info(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """提取类的详细信息"""
        # 提取类的方法
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, content)
                methods.append(method_info)

        # 获取基类
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            else:
                bases.append(ast.unparse(base))

        return {
            'name': node.name,
            'bases': bases,
            'methods': methods,
            'docstring': ast.get_docstring(node),
            'line_start': node.lineno,
            'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
        }

    def parse_repository(self, exclude_dirs: List[str] = None) -> Dict[str, Any]:
        """
        解析整个代码仓库

        Args:
            exclude_dirs: 要排除的目录列表

        Returns:
            包含所有解析结果的字典
        """
        if exclude_dirs is None:
            exclude_dirs = ['__pycache__', '.git', 'venv', 'env', 'tests', 'migrations']

        # 遍历仓库中的所有Python文件
        for root, dirs, files in os.walk(self.repo_path):
            # 跳过排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    print(f"解析: {file_path.relative_to(self.repo_path)}")

                    file_info = self.parse_file(file_path)
                    if file_info:
                        self.results['files'].append(file_info)

                        # 更新统计信息
                        self.results['statistics']['total_files'] += 1
                        self.results['statistics']['total_functions'] += len(file_info['functions'])
                        self.results['statistics']['total_classes'] += len(file_info['classes'])
                        self.results['statistics']['total_lines'] += file_info['lines']

        return self.results

    def save_results(self, output_path: str = None):
        """
        将解析结果保存为JSON文件

        Args:
            output_path: 输出文件路径，默认为data/parsed_code.json
        """
        if output_path is None:
            output_dir = Path(self.repo_path.parent) / 'data'
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / 'parsed_code.json'
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"解析完成！结果已保存到: {output_path}")
        print(f"统计信息: {self.results['statistics']}")

        return str(output_path)


# 使用示例
if __name__ == "__main__":
    # 1. 设置你的仓库路径
    REPO_PATH = "../test_repo"  # 根据你的实际路径调整

    # 2. 创建解析器并解析仓库
    parser = CodeParser(REPO_PATH)

    # 3. 解析并保存结果
    results = parser.parse_repository()
    output_file = parser.save_results()

    # 4. 打印一些示例信息
    print("\n=== 解析结果示例 ===")
    for file_info in results['files'][:3]:  # 只显示前3个文件
        print(f"\n文件: {file_info['file_path']}")
        print(f"函数数: {len(file_info['functions'])}")
        print(f"类数: {len(file_info['classes'])}")

        # 显示前2个函数
        for func in file_info['functions'][:2]:
            print(f"  - 函数: {func['name']}({', '.join(func['args'])})")
            if func['docstring']:
                print(f"    文档: {func['docstring'][:100]}...")
