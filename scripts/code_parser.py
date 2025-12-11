import os
import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class CodeParser:
    """
    解析Python代码文件，提取函数、类、方法等结构化信息
    不使用 ast.unparse，兼容所有Python版本
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.results = {
            'repo_name': self.repo_path.name,
            'files': [],
            'statistics': {
                'total_files': 0,
                'total_functions': 0,
                'total_async_functions': 0,
                'total_classes': 0,
                'total_methods': 0,
                'total_async_methods': 0,
                'total_lines': 0
            }
        }

    def parse_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            file_info = {
                'file_path': str(file_path.relative_to(self.repo_path)),
                'file_name': file_path.name,
                'absolute_path': str(file_path),
                'lines': content.count('\n') + 1,
                'imports': [],
                'functions': [],
                'async_functions': [],
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
                if isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function_info(node, content, is_async=False)
                    file_info['functions'].append(func_info)

                elif isinstance(node, ast.AsyncFunctionDef):
                    func_info = self._extract_function_info(node, content, is_async=True)
                    file_info['functions'].append(func_info)
                    file_info['async_functions'].append(func_info)

                elif isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node, content)
                    file_info['classes'].append(class_info)

            return file_info

        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"警告: 无法解析文件 {file_path}: {e}")
            return None

    def _extract_function_info(self, node, content: str, is_async: bool = False) -> Dict[str, Any]:
        """提取函数的详细信息（不使用 ast.unparse）"""
        # 获取函数签名
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        # 简化装饰器提取：只提取装饰器名称
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                # 处理形如 app.get 的装饰器
                parts = []
                current = decorator
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                decorators.append('.'.join(reversed(parts)))
            elif isinstance(decorator, ast.Call):
                # 处理形如 @decorator(arg) 的情况
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    # 处理形如 app.get("/path") 的情况
                    parts = []
                    current = decorator.func
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    decorators.append('.'.join(reversed(parts)))

        # 获取函数体上下文
        lines = content.split('\n')
        start_line = node.lineno - 1

        if hasattr(node, 'end_lineno'):
            end_line = node.end_lineno
        else:
            end_line = min(start_line + 30, len(lines))

        end_line = min(end_line, len(lines))
        function_context = '\n'.join(lines[start_line:end_line])

        # 如果函数太长，截取前50行
        if end_line - start_line > 50:
            function_context = '\n'.join(lines[start_line:start_line + 50])
            function_context += '\n# ... (函数内容过长，已截断)'

        # 检查是否是API端点（基于装饰器）
        api_decorators = {'app.get', 'app.post', 'app.put', 'app.delete',
                         'router.get', 'router.post', 'router.put', 'router.delete',
                         'app.websocket', 'router.websocket'}
        is_api_endpoint = any(decorator in api_decorators for decorator in decorators)

        return {
            'name': node.name,
            'is_async': is_async,
            'args': args,
            'decorators': decorators,
            'docstring': ast.get_docstring(node),
            'line_start': node.lineno,
            'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            'is_api_endpoint': is_api_endpoint,
            'context_snippet': function_context[:800],
            # 简化返回类型提取
            'return_type': self._extract_return_type(node) if node.returns else None
        }

    def _extract_return_type(self, node) -> str:
        """简化返回类型提取"""
        if node.returns is None:
            return None

        # 处理常见返回类型
        if isinstance(node.returns, ast.Name):
            return node.returns.id
        elif isinstance(node.returns, ast.Subscript):
            # 处理 List[str] 等类型
            if isinstance(node.returns.value, ast.Name):
                base = node.returns.value.id
                if node.returns.slice:
                    if isinstance(node.returns.slice, ast.Name):
                        return f"{base}[{node.returns.slice.id}]"
                    elif hasattr(node.returns.slice, 'value'):
                        return f"{base}[{node.returns.slice.value.id}]"
        elif isinstance(node.returns, ast.Attribute):
            # 处理 typing.Optional 等
            return self._extract_attribute_name(node.returns)

        return str(type(node.returns).__name__)

    def _extract_attribute_name(self, node) -> str:
        """提取属性名称，如 typing.Optional"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return '.'.join(reversed(parts))

    def _extract_class_info(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """提取类的详细信息"""
        # 提取类的方法
        methods = []
        async_methods = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, content, is_async=False)
                methods.append(method_info)
            elif isinstance(item, ast.AsyncFunctionDef):
                method_info = self._extract_function_info(item, content, is_async=True)
                methods.append(method_info)
                async_methods.append(method_info)

        # 简化基类提取
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            else:
                bases.append(self._extract_attribute_name(base) if isinstance(base, ast.Attribute) else str(base))

        # 简化类装饰器提取
        class_decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                class_decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                class_decorators.append(self._extract_attribute_name(decorator))
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    class_decorators.append(decorator.func.id)

        return {
            'name': node.name,
            'bases': bases,
            'class_decorators': class_decorators,
            'methods': methods,
            'async_methods': async_methods,
            'total_methods': len(methods),
            'total_async_methods': len(async_methods),
            'docstring': ast.get_docstring(node),
            'line_start': node.lineno,
            'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
        }

    def parse_repository(self, exclude_dirs: List[str] = None) -> Dict[str, Any]:
        if exclude_dirs is None:
            exclude_dirs = ['__pycache__', '.git', 'venv', 'env', 'tests', 'migrations']

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    print(f"解析: {file_path.relative_to(self.repo_path)}")

                    file_info = self.parse_file(file_path)
                    if file_info:
                        self.results['files'].append(file_info)

                        stats = self.results['statistics']
                        stats['total_files'] += 1
                        stats['total_functions'] += len(file_info['functions'])
                        stats['total_async_functions'] += len(file_info.get('async_functions', []))
                        stats['total_classes'] += len(file_info['classes'])
                        stats['total_lines'] += file_info['lines']

                        for cls in file_info['classes']:
                            stats['total_methods'] += len(cls.get('methods', []))
                            stats['total_async_methods'] += len(cls.get('async_methods', []))

        return self.results

    def save_results(self, output_path: str = None):
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
    REPO_PATH = "../test_repo"
    parser = CodeParser(REPO_PATH)
    results = parser.parse_repository()
    output_file = parser.save_results()

    # 显示统计信息
    print("\n=== 解析统计 ===")
    stats = results['statistics']
    print(f"文件总数: {stats['total_files']}")
    print(f"函数总数: {stats['total_functions']}")
    print(f"异步函数数: {stats['total_async_functions']}")
    print(f"类总数: {stats['total_classes']}")

    # 显示前几个函数的示例
    print("\n=== 函数示例 ===")
    func_count = 0
    for file_info in results['files'][:2]:  # 只显示前2个文件
        for func in file_info['functions'][:3]:  # 每个文件显示前3个函数
            print(f"\n文件: {file_info['file_path']}")
            print(f"函数: {func['name']}")
            print(f"是否异步: {func['is_async']}")
            print(f"装饰器: {func['decorators']}")
            print(f"是否为API端点: {func['is_api_endpoint']}")
            func_count += 1
