import re
from typing import List, Dict, Any

from arclang.nodes import Node
from arclang.image import Image, Point
from arclang.command import CommandMapper

class ExecutionEngine:
    def __init__(self, input_img):
        self.command_mapper = CommandMapper(input_img)
        self.initial_image = input_img
        self.current_image = input_img
        self.prev_state_img = []
        self.variables = {}
    
    def execute(self, root_node: Node) -> Image:
        for child in root_node.children:
            self.execute_node(child)
        return self.current_image

    def execute_node(self, node: Node) -> None:
        if node.node_type == 'ASSIGN':
            self.execute_assignment(node)
        elif node.node_type == 'FUNCTION':
            self.execute_function(node)
        elif node.node_type == 'IF':
            self.execute_if(node)
        elif node.node_type == 'LOOP':
            self.execute_loop(node)
        elif node.node_type == 'EMPTY':
            pass
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")

    def execute_assignment(self, node: Node) -> None:
        var_name = node.value['variable']
        expression = node.value['expression']
        
        if node.children and node.children[0].node_type == 'FUNCTION':
            # If the assignment is a function call, execute it and store the result
            self.execute_function(node.children[0])
            self.variables[var_name] = self.current_image
        else:
            # Otherwise, evaluate the expression and store the result
            self.variables[var_name] = self.evaluate_expression(expression)

    def execute_function(self, node: Node) -> None:
        func_name = node.value['name']
        args = node.value['args']
        
        # Replace variable references with their values
        args = [self.evaluate_expression(arg) if isinstance(arg, str) and arg in self.variables else arg for arg in args]
        
        if func_name in self.command_mapper.function_map:
            # Execute the function and update the current image
            self.current_image = self.command_mapper.function_map[func_name](args)
        else:
            raise ValueError(f"Unknown function: {func_name}")

    def execute_if(self, node: Node) -> None:
        condition = self.evaluate_condition(node.value['condition'])
        print(condition)
        if condition:
            self.execute(node.children[0])
        elif len(node.children) > 1:
            self.execute(node.children[1])

    def execute_loop(self, node: Node) -> None:
        loop_type = node.value['type']
        if loop_type == 'FOR':
            self.execute_for_loop(node)
        elif loop_type == 'WHILE':
            self.execute_while_loop(node)
        else:
            raise ValueError(f"Unknown loop type: {loop_type}")

    def execute_for_loop(self, node: Node) -> None:
        var_name = node.value['variable']
        start, end, step = node.value['range']
        for i in range(start, end, step):
            self.variables[var_name] = i
            self.execute(node.children[0])

    def execute_while_loop(self, node: Node) -> None:
        while self.evaluate_condition(node.value['condition']):
            self.execute(node.children[0])

    def evaluate_expression(self, expression: str) -> Any:
        if isinstance(expression, str):
            try:
                return eval(expression, {}, self.variables)
            except:
                return expression  # If it's not a valid Python expression, return as is
        return expression

    def evaluate_condition(self, condition: str) -> bool:
        return bool(self.evaluate_expression(condition))