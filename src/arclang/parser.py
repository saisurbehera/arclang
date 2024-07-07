import re
from typing import List, Tuple

from arclang.nodes import Node, Symbol, SymbolTable

class Parser:
    def __init__(self):
        self.symbol_table = SymbolTable()

    def parse(self, script: str, root_name="ROOT") -> Node:
        lines = script.split('\n')
        root = Node(root_name)
        self.parse_block(lines, 0, root)
        return root

    def parse_block(self, lines: List[str], start: int, parent_node: Node) -> int:
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            if line.upper() in ['END', 'ELSE']:
                return i

            if line.upper().startswith('DEF'):
                i, func_def_node = self.parse_function_definition(lines, i)
                parent_node.children.append(func_def_node)
                self.symbol_table.add_symbol(func_def_node.value['name'], 'function')
            elif line.upper().startswith('IF'):
                i, if_node = self.parse_if_block(lines, i)
                parent_node.children.append(if_node)
            elif line.upper().startswith('WHILE'):
                i, while_node = self.parse_while_loop(lines, i)
                parent_node.children.append(while_node)
            elif line.upper().startswith('FOR'):
                i, for_node = self.parse_for_loop(lines, i)
                parent_node.children.append(for_node)
            elif '=' in line:
                assign_node = self.parse_assignment(line)
                parent_node.children.append(assign_node)
                self.symbol_table.add_symbol(assign_node.value['variable'], 'variable', assign_node.value['expression'])
            else:
                func_call_node = self.parse_function_call(line)
                parent_node.children.append(func_call_node)
                if not self.symbol_table.lookup(func_call_node.value['name']):
                    print(f"Warning: Function '{func_call_node.value['name']}' called but not defined")
            i += 1
        
        return i

    def parse_assignment(self, line: str) -> Node:
        var_name, expression = map(str.strip, line.split('='))
        return Node('ASSIGN', {'variable': var_name, 'expression': expression})

    def parse_function_call(self, line: str) -> Node:
        match = re.match(r'(\w+)\s*(.*)', line)
        if not match:
            raise ValueError(f"Invalid function call: {line}")
        
        func_name, args_str = match.groups()
        args_str = args_str.replace(";","")
        args = [arg.strip() for arg in args_str.split()] if args_str else []
        
        return Node('FUNCTION_CALL', {'name': func_name.upper(), 'args': args})

    def parse_function_definition(self, lines: List[str], start: int) -> Tuple[int, Node]:
        def_line = lines[start].strip()
        match = re.match(r'DEF\s+(\w+)\s*\((.*?)\)', def_line, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid function definition: {def_line}")
        
        func_name, params_str = match.groups()
        params = [param.strip() for param in params_str.split(',') if param.strip()]
        
        self.symbol_table.enter_scope(func_name)
        for param in params:
            self.symbol_table.add_symbol(param, 'parameter')
        
        body_node = Node('BLOCK')
        i = self.parse_block(lines, start + 1, body_node)
        
        self.symbol_table.exit_scope()
        
        return i, Node('FUNCTION_DEF', {'name': func_name, 'params': params}, [body_node])

    def parse_if_block(self, lines: List[str], start: int) -> Tuple[int, Node]:
        if_line = lines[start].strip()
        match = re.match(r'IF\s+(.+)', if_line, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid IF statement: {if_line}")
        
        condition = match.group(1)
        true_block = Node('TRUE_BLOCK')
        false_block = Node('FALSE_BLOCK')
        
        i = self.parse_block(lines, start + 1, true_block)
        
        if i < len(lines) and lines[i].strip().upper() == 'ELSE':
            i = self.parse_block(lines, i + 1, false_block)
        
        return i, Node('IF', {'condition': condition}, [true_block, false_block])

    def parse_while_loop(self, lines: List[str], start: int) -> Tuple[int, Node]:
        while_line = lines[start].strip()
        match = re.match(r'WHILE\s+(.+)', while_line, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid WHILE statement: {while_line}")
        
        condition = match.group(1)
        body_node = Node('LOOP_WHILE')
        
        i = self.parse_block(lines, start + 1, body_node)
        return i, Node('WHILE', {'condition': condition}, [body_node])

    def parse_for_loop(self, lines: List[str], start: int) -> Tuple[int, Node]:
        for_line = lines[start].strip()
        match = re.match(r'FOR\s+(\w+)\s+IN\s+(.+)', for_line, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid FOR statement: {for_line}")
        
        iterator, iterable = match.groups()
        body_node = Node('LOOP_FOR')
        
        i = self.parse_block(lines, start + 1, body_node)
        
        return i, Node('FOR', {'iterator': iterator, 'iterable': iterable}, [body_node])