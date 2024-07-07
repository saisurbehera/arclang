from typing import List, Any, Dict, Optional

class Node:
    def __init__(self, node_type: str, value: Any = None, children: List['Node'] = None):
        self.node_type = node_type
        self.value = value
        self.children = children or []

    def __repr__(self):
        return f"Node(type={self.node_type}, value={self.value}, children={self.children})"

    def format(self, indent=0):
        """
        Format the Node and its children into a readable string representation.
        
        :param indent: The current indentation level (default: 0)
        :return: A formatted string representation of the Node
        """
        indent_str = '    ' * indent
        result = f"{indent_str}Node(type={self.node_type}"
        
        if self.value is not None:
            result += f", value={self.value}"
        
        if self.children:
            result += ", children=[\n"
            for child in self.children:
                result += child.format(indent + 1) + ",\n"
            result += f"{indent_str}]"
        else:
            result += ", children=[]"
        
        result += ")"
        return result

    def show(self):
        """
        Print the formatted representation of the Node and its children.
        """
        print(self.format())

class Symbol:
    def __init__(self, name: str, type: str, scope: str, value: Any = None):
        self.name = name
        self.type = type  # 'variable', 'function', etc.
        self.scope = scope
        self.value = value


class SymbolTable:
    def __init__(self):
        self.scopes: Dict[str, Dict[str, Symbol]] = {"global": {}}
        self.current_scope = "global"

    def enter_scope(self, scope_name: str):
        if scope_name not in self.scopes:
            self.scopes[scope_name] = {}
        self.current_scope = scope_name

    def exit_scope(self):
        self.current_scope = "global"

    def add_symbol(self, name: str, type: str, value: Any = None):
        symbol = Symbol(name, type, self.current_scope, value)
        self.scopes[self.current_scope][name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        if name in self.scopes[self.current_scope]:
            return self.scopes[self.current_scope][name]
        elif name in self.scopes["global"]:
            return self.scopes["global"][name]
        return None



