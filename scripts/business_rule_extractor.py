import re
import ast
from typing import List, Dict, Optional, Set

class BusinessRuleExtractor:
    """从解析的代码中提取业务规则（增强版）"""

    def __init__(self, parsed_data: Dict):
        self.parsed_data = parsed_data
        self.business_rules = []

        # 扩展的业务关键词和模式
        self.validation_keywords = {
            'validate', 'check', 'verify', 'ensure', 'require', 'must', 'should',
            'authorize', 'authenticate', 'permit', 'forbid', 'restrict', 'limit',
            'calculate', 'compute', 'transform', 'process', 'handle', 'filter'
        }

        self.business_entities = {
            'user', 'customer', 'order', 'product', 'item', 'restaurant', 'review',
            'payment', 'invoice', 'account', 'transaction', 'reservation', 'booking'
        }

        self.api_methods = {'get', 'post', 'put', 'delete', 'patch'}

    def extract_rules(self) -> List[Dict]:
        """从所有文件中提取业务规则（增强版）"""
        for file_info in self.parsed_data['files']:
            # 1. 从函数中提取
            for func in file_info['functions']:
                rules = self._extract_from_function_enhanced(func, file_info)
                self.business_rules.extend(rules)

            # 2. 从类中提取
            for cls in file_info['classes']:
                rules = self._extract_from_class_enhanced(cls, file_info)
                self.business_rules.extend(rules)

            # 3. 从文件整体提取（如模型字段验证）
            file_rules = self._extract_from_file(file_info)
            self.business_rules.extend(file_rules)

        return self.business_rules

    def _extract_from_function_enhanced(self, func: Dict, file_info: Dict) -> List[Dict]:
        """增强版函数规则提取"""
        rules = []

        # 检查1：函数名是否包含业务关键词
        func_name_lower = func['name'].lower()
        for keyword in self.validation_keywords:
            if keyword in func_name_lower:
                rules.append({
                    'type': 'validation_function',
                    'file': file_info['file_path'],
                    'function_name': func['name'],
                    'line_number': func['line_start'],
                    'context': func['context_snippet'],
                    'docstring': func['docstring'],
                    'confidence': 'high',
                    'matched_keyword': keyword
                })
                break

        # 检查2：是否是业务操作（create, update, delete等）
        operation_keywords = ['create_', 'add_', 'update_', 'delete_', 'remove_',
                             'get_', 'find_', 'search_', 'list_']
        for op in operation_keywords:
            if func_name_lower.startswith(op):
                # 检查操作的对象是否是业务实体
                for entity in self.business_entities:
                    if entity in func_name_lower:
                        rules.append({
                            'type': 'business_operation',
                            'file': file_info['file_path'],
                            'function_name': func['name'],
                            'line_number': func['line_start'],
                            'context': func['context_snippet'],
                            'docstring': func['docstring'],
                            'confidence': 'high',
                            'operation': op.rstrip('_'),
                            'entity': entity
                        })
                        break
                break

        # 检查3：文档字符串中的业务规则
        if func.get('docstring'):
            doc_rules = self._extract_rules_from_docstring(
                func['docstring'], func, file_info, 'function'
            )
            rules.extend(doc_rules)

        # 检查4：函数体中的业务规则
        if func.get('context_snippet'):
            code_rules = self._extract_rules_from_code(
                func['context_snippet'], func, file_info, 'function'
            )
            rules.extend(code_rules)

        # 检查5：API端点（但需要提取其中的业务逻辑）
        if func.get('is_api_endpoint', False):
            # 深入分析API端点的内容
            endpoint_rules = self._analyze_api_endpoint(func, file_info)
            rules.extend(endpoint_rules)

        return rules

    def _extract_from_class_enhanced(self, cls: Dict, file_info: Dict) -> List[Dict]:
        """增强版类规则提取"""
        rules = []

        # 检查1：类名是否表示业务实体
        class_name_lower = cls['name'].lower()
        for entity in self.business_entities:
            if entity in class_name_lower:
                # 提取类级别的业务规则
                class_rule = {
                    'type': 'business_entity',
                    'file': file_info['file_path'],
                    'class_name': cls['name'],
                    'line_number': cls['line_start'],
                    'class_docstring': cls['docstring'],
                    'confidence': 'high',
                    'entity_type': entity
                }

                # 从文档字符串中提取规则
                if cls.get('docstring'):
                    doc_rules = self._extract_rules_from_docstring(
                        cls['docstring'], cls, file_info, 'class'
                    )
                    class_rule['documented_rules'] = doc_rules

                # 从字段/属性中提取验证规则
                field_rules = self._extract_field_rules(cls, file_info)
                if field_rules:
                    class_rule['field_rules'] = field_rules

                rules.append(class_rule)
                break

        # 检查2：是否是数据模型（Pydantic/SQLAlchemy）
        model_indicators = ['model', 'schema', 'table']
        if any(indicator in class_name_lower for indicator in model_indicators):
            # 提取模型验证规则
            model_rules = self._extract_model_rules(cls, file_info)
            rules.extend(model_rules)

        # 检查3：类中的方法
        for method in cls.get('methods', []):
            method_rules = self._extract_from_function_enhanced(method, file_info)
            for rule in method_rules:
                rule['parent_class'] = cls['name']
            rules.extend(method_rules)

        return rules

    def _extract_from_file(self, file_info: Dict) -> List[Dict]:
        """从文件整体内容提取规则"""
        rules = []
        file_path = file_info['file_path'].lower()

        # 检查是否是特定的业务文件
        if any(entity in file_path for entity in self.business_entities):
            rules.append({
                'type': 'business_file',
                'file': file_info['file_path'],
                'description': f"业务相关的文件，包含{len(file_info['functions'])}个函数和{len(file_info['classes'])}个类",
                'confidence': 'medium'
            })

        return rules

    def _extract_rules_from_docstring(self, docstring: str, source: Dict,
                                    file_info: Dict, source_type: str) -> List[Dict]:
        """从文档字符串中提取业务规则"""
        rules = []

        # 查找常见的业务规则模式
        patterns = [
            (r'(必须|should|must|required to|needs to)\s+(.+?)(?:。|\.)', 'requirement'),
            (r'(验证|validate|check|verify)\s+(.+?)(?:。|\.)', 'validation'),
            (r'(规则|rule|policy|constraint)[：:]\s*(.+?)(?:。|\.)', 'policy'),
            (r'(如果|if)\s+(.+?)\s*(?:则|then)\s*(.+?)(?:。|\.)', 'condition'),
            (r'(最小|最少|at least|min(?:imum)?)\s*[:：]?\s*(\d+)', 'min_constraint'),
            (r'(最大|最多|at most|max(?:imum)?)\s*[:：]?\s*(\d+)', 'max_constraint'),
            (r'(只能|only)\s+(.+?)(?:。|\.)', 'restriction'),
        ]

        for pattern, rule_type in patterns:
            matches = re.findall(pattern, docstring, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    rule_text = ' '.join([m for m in match if m])
                else:
                    rule_text = match

                if len(rule_text) > 10:  # 避免太短的匹配
                    rules.append({
                        'type': f'documented_{rule_type}',
                        'file': file_info['file_path'],
                        'source': source['name'] if source_type == 'function' else source['name'],
                        'source_type': source_type,
                        'rule_text': rule_text.strip(),
                        'confidence': 'medium'
                    })

        return rules

    def _extract_rules_from_code(self, code_snippet: str, source: Dict,
                               file_info: Dict, source_type: str) -> List[Dict]:
        """从代码片段中提取业务规则"""
        rules = []

        # 查找常见的验证代码模式
        validation_patterns = [
            (r'if\s+not\s+(.+?):', 'null_check'),
            (r'if\s+len\((.+?)\)\s*[<>]=?\s*(\d+):', 'length_check'),
            (r'if\s+(.+?)\s*[<>]=?\s*(\d+):', 'value_check'),
            (r'assert\s+(.+?)', 'assertion'),
            (r'raise\s+(.+?)Exception', 'exception'),
            (r'\.validate\(', 'validation_call'),
            (r'Field\([^)]*(?:min|max|gt|lt|ge|le)[^)]*\)', 'field_validation'),
        ]

        lines = code_snippet.split('\n')
        for i, line in enumerate(lines):
            for pattern, rule_type in validation_patterns:
                match = re.search(pattern, line)
                if match:
                    rules.append({
                        'type': f'code_{rule_type}',
                        'file': file_info['file_path'],
                        'source': source['name'],
                        'source_type': source_type,
                        'line_content': line.strip(),
                        'line_in_snippet': i + 1,
                        'confidence': 'high',
                        'pattern_matched': pattern
                    })

        return rules

    def _analyze_api_endpoint(self, func: Dict, file_info: Dict) -> List[Dict]:
        """深入分析API端点，提取其中的业务逻辑"""
        rules = []

        # 从端点名称推断业务操作
        endpoint_name = func['name'].lower()

        # 映射常见端点模式到业务操作
        endpoint_patterns = [
            (r'create_(\w+)', 'create_operation'),
            (r'add_(\w+)', 'add_operation'),
            (r'update_(\w+)', 'update_operation'),
            (r'delete_(\w+)', 'delete_operation'),
            (r'get_(\w+)', 'read_operation'),
            (r'list_(\w+)', 'list_operation'),
        ]

        for pattern, operation_type in endpoint_patterns:
            match = re.match(pattern, endpoint_name)
            if match:
                entity = match.group(1)
                rules.append({
                    'type': 'api_business_operation',
                    'file': file_info['file_path'],
                    'endpoint': func['name'],
                    'operation': operation_type.replace('_operation', ''),
                    'entity': entity,
                    'line_number': func['line_start'],
                    'confidence': 'high',
                    'description': f"API端点执行{operation_type.replace('_', ' ')}操作"
                })
                break

        # 从装饰器提取HTTP方法和路径
        for decorator in func.get('decorators', []):
            for method in self.api_methods:
                if f'.{method}(' in decorator or f'.{method}]' in decorator:
                    # 提取路径
                    path_match = re.search(r'["\'](/[^"\']+)["\']', decorator)
                    path = path_match.group(1) if path_match else 'unknown'

                    rules.append({
                        'type': 'api_endpoint_details',
                        'file': file_info['file_path'],
                        'endpoint': func['name'],
                        'http_method': method.upper(),
                        'path': path,
                        'line_number': func['line_start'],
                        'confidence': 'high'
                    })
                    break

        return rules

    def _extract_field_rules(self, cls: Dict, file_info: Dict) -> List[Dict]:
        """从类定义中提取字段验证规则"""
        rules = []
        context = cls.get('context_snippet', '')

        # 查找Pydantic Field验证
        field_patterns = [
            (r'(\w+)\s*:\s*\w+\s*=\s*Field\([^)]*min_length\s*=\s*(\d+)', 'min_length'),
            (r'(\w+)\s*:\s*\w+\s*=\s*Field\([^)]*max_length\s*=\s*(\d+)', 'max_length'),
            (r'(\w+)\s*:\s*\w+\s*=\s*Field\([^)]*gt\s*=\s*(\d+)', 'greater_than'),
            (r'(\w+)\s*:\s*\w+\s*=\s*Field\([^)]*lt\s*=\s*(\d+)', 'less_than'),
            (r'(\w+)\s*:\s*\w+\s*=\s*Field\([^)]*ge\s*=\s*(\d+)', 'min_value'),
            (r'(\w+)\s*:\s*\w+\s*=\s*Field\([^)]*le\s*=\s*(\d+)', 'max_value'),
            (r'(\w+)\s*:\s*\w+\s*=\s*Field\([^)]*regex\s*=\s*[\'"]([^\'"]+)[\'"]', 'regex'),
        ]

        for pattern, rule_type in field_patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    field_name, constraint_value = match[0], match[1]
                    rules.append({
                        'type': f'field_{rule_type}',
                        'file': file_info['file_path'],
                        'class_name': cls['name'],
                        'field': field_name,
                        'constraint': f"{rule_type}: {constraint_value}",
                        'confidence': 'high'
                    })

        return rules

    def _extract_model_rules(self, cls: Dict, file_info: Dict) -> List[Dict]:
        """提取数据模型规则"""
        rules = []

        # 检查是否是SQLAlchemy模型
        if any('Base' in base for base in cls.get('bases', [])):
            rules.append({
                'type': 'database_model',
                'file': file_info['file_path'],
                'class_name': cls['name'],
                'description': '数据库模型类',
                'total_fields': self._count_model_fields(cls),
                'confidence': 'medium'
            })

        # 检查是否是Pydantic模型
        if any('BaseModel' in base for base in cls.get('bases', [])):
            rules.append({
                'type': 'pydantic_model',
                'file': file_info['file_path'],
                'class_name': cls['name'],
                'description': '数据验证模型类',
                'confidence': 'medium'
            })

        return rules

    def _count_model_fields(self, cls: Dict) -> int:
        """估算模型字段数量（通过分析类上下文）"""
        context = cls.get('context_snippet', '')
        # 简单的字段计数：查找冒号后的类型注解
        field_pattern = r'\w+\s*:\s*\w+(\s*=\s*\w+)?'
        return len(re.findall(field_pattern, context))

    def get_summary(self) -> Dict:
        """获取提取结果的详细摘要"""
        rule_types = {}
        confidence_levels = {'high': 0, 'medium': 0, 'low': 0}
        files_analyzed = set()

        for rule in self.business_rules:
            rule_type = rule['type']
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1

            confidence = rule.get('confidence', 'medium')
            confidence_levels[confidence] = confidence_levels.get(confidence, 0) + 1

            files_analyzed.add(rule['file'])

        return {
            'total_rules': len(self.business_rules),
            'rule_types': rule_types,
            'confidence_distribution': confidence_levels,
            'files_analyzed': len(files_analyzed),
            'unique_sources': len(set(r.get('source', '') for r in self.business_rules if r.get('source')))
        }

    def print_detailed_report(self):
        """打印详细的提取报告"""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print(f"{'业务规则提取详细报告':^60}")
        print(f"{'='*60}")

        print(f"\n📊 统计摘要:")
        print(f"  • 总共提取到 {summary['total_rules']} 条业务规则")
        print(f"  • 分析了 {summary['files_analyzed']} 个文件")
        print(f"  • 置信度分布: {summary['confidence_distribution']}")

        print(f"\n📁 规则类型分布:")
        for rule_type, count in summary['rule_types'].items():
            print(f"  • {rule_type}: {count} 条")

        print(f"\n🔍 详细规则示例 (前10条):")
        for i, rule in enumerate(self.business_rules[:10]):
            print(f"\n  [{i+1}] {rule['type']} (置信度: {rule.get('confidence', 'N/A')})")
            print(f"      文件: {rule['file']}")

            if 'function_name' in rule:
                print(f"      函数: {rule['function_name']}")
            elif 'class_name' in rule:
                print(f"      类: {rule['class_name']}")

            if 'description' in rule:
                print(f"      描述: {rule['description']}")
            elif 'rule_text' in rule:
                print(f"      规则: {rule['rule_text'][:80]}...")

            if 'line_number' in rule:
                print(f"      行号: {rule['line_number']}")


# 使用示例
if __name__ == "__main__":
    # 1. 首先运行代码解析器
    from code_parser import CodeParser

    REPO_PATH = "../test_repo"
    parser = CodeParser(REPO_PATH)
    parsed_data = parser.parse_repository()

    # 2. 运行业务规则提取器（增强版）
    extractor = BusinessRuleExtractor(parsed_data)
    business_rules = extractor.extract_rules()

    # 3. 打印详细报告
    extractor.print_detailed_report()

    # 4. 保存提取结果
    import json
    output_path = "../data/business_rules_enhanced.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': extractor.get_summary(),
            'rules': business_rules
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 详细结果已保存到: {output_path}")
