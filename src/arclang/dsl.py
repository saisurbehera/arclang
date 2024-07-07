from arclang.function import *
from arclang.image import Image, Point 

import numpy as np
import numpy as np
from collections import namedtuple
from typing import List, Tuple, Callable, Any

class CommandMapper:
    def __init__(self,image):
        self.image  = image
        self.function_map = {
            'SELECT': self.select,
            'DESELECT': self.deselect,
            'FILL': self.fill,
            'SPLIT': self.split,
            'EXPAND': self.expand,
            'SHRINK': self.shrink,
            'CLEAR': self.clear,
            'COPY': self.copy,
            'MOVE': self.move,
            'ROTATE': self.rotate,
            'FLIP': self.flip,
            'REPLACECOL': self.replacecol,
            'OVERLAY': self.overlay,
            'SUBTRACT': self.subtract,
            'INTERSECT': self.intersect,
            'REPEAT': self.repeat,
            'APPLY_PATTERN': self.apply_pattern,
            'COUNTCOL': self.count,
            'GROUP': self.group,
            'FOREACH': self.foreach,
            'WHILE': self.while_loop,
            'CONVOLVE': self.convolve,
            'APPLY_FILTER': self.apply_filter,
            'SCALE': self.scale,
            'EXTRUDE': self.extrude,
            'RANDOMIZE': self.randomize,
            'SORT_COLORS': self.sort_colors,
            'INVERT_COLORS': self.invert_colors,
            'CREATE_BORDER': self.create_border,
            'CROP': self.crop,
            'RESIZE': self.resize,
            'APPLY_MASK': self.apply_mask,
            'BLEND': self.blend,
            'RECURSIVE': self.recursive,
            'DISPLAY': self.display,
        }
        self.selected_area = None

    def select(self, args: List[str]) -> Callable[[Image], Image]:
        x1, y1, x2, y2 = map(int, args)
        self.selected_area = (x1, y1, x2, y2)
        return lambda img: img
    
    def deselect(self, args: List[str]) -> Callable[[Image], Image]:
        self.selected_area = None
        return lambda img: img

    def split(self,  args: List[str]) -> Callable[[Image], Image]:
        raise NotImplementedError()

    def fill(self, args: List[str]) -> Callable[[Image], Image]:
        color = int(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: Image.full(Point(area.x, area.y), Point(area.w, area.h), color))

    def expand(self, args: List[str]) -> Callable[[Image], Image]:
        size = int(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: extend2(area, Image.full(Point(area.x-size, area.y-size), Point(area.w+2*size, area.h+2*size), 0)))

    def shrink(self, args: List[str]) -> Callable[[Image], Image]:
        size = int(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: area.sub_image(Point(size, size), Point(area.w-2*size, area.h-2*size)))

    def clear(self, args: List[str]) -> Callable[[Image], Image]:
        return lambda img: self.apply_to_selected_area(img, lambda area: Image.empty(area.x, area.y, area.w, area.h))

    def copy(self, args: List[str]) -> Callable[[Image], Image]:
        x, y = map(int, args)
        return lambda img: self.apply_to_selected_area(img, lambda area: compose_id(area, move(area, (x, y)), 0))

    def move(self, args: List[str]) -> Callable[[Image], Image]:
        x, y = map(int, args)
        return lambda img: self.apply_to_selected_area_move(img, lambda area: move(area, (x, y)),(x,y))

    def rotate(self, args: List[str]) -> Callable[[Image], Image]:
        angle = int(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: rigid(area, angle // 90))

    def flip(self, args: List[str]) -> Callable[[Image], Image]:
        axis = args[0].upper()
        return lambda img: self.apply_to_selected_area(img, lambda area: rigid(area, 4 if axis == 'HORIZONTAL' else 5))

    def replacecol(self, args: List[str]) -> Callable[[Image], Image]:
        old_color, new_color = map(int, args[:2])
        if len(args) > 2 and args[2] == 'IN':
            x1, y1, x2, y2 = args[3:]  # Keep these as strings for now
            return lambda img: self.replace_in_area(img, old_color, new_color, x1, y1, x2, y2)
        return lambda img: self.replace_color(img, old_color, new_color)

    def replace_in_area(self, img: Image, old_color: int, new_color: int, x1: str, y1: str, x2: str, y2: str) -> Image:
        # The actual replacement will be done in the SQLExecutionEngine
        return lambda col: self._replace_in_area(img, old_color, new_color, x1, y1, x2, y2, col)

    def _replace_in_area(self, img: Image, old_color: int, new_color: int, x1: str, y1: str, x2: str, y2: str, col: int) -> Image:
        new_img = img.copy()
        x1 = int(x1.replace('[COL]', str(col)))
        x2 = int(x2.replace('[COL]', str(col)))
        y1, y2 = int(y1), int(y2)
        mask = (new_img.mask[y1:y2, x1:x2] == old_color)
        new_img.mask[y1:y2, x1:x2][mask] = new_color
        return new_img

    def replace_color(self, img: Image, old_color: int, new_color: int) -> Image:
        new_img = img.copy()
        new_img.mask[new_img.mask == old_color] = new_color
        return new_img

    def overlay(self, args: List[str]) -> Callable[[Image], Image]:
        return lambda img: self.apply_to_selected_area(img, lambda area: compose_id(area, area, 3))

    def subtract(self, args: List[str]) -> Callable[[Image], Image]:
        return lambda img: self.apply_to_selected_area(img, lambda area: compose_id(area, invert(area), 2))

    def intersect(self, args: List[str]) -> Callable[[Image], Image]:
        return lambda img: self.apply_to_selected_area(img, lambda area: compose_id(area, area, 2))

    def repeat(self, args: List[str]) -> Callable[[Image], Image]:
        operation, times = args[0], int(args[1])
        return lambda img: self.apply_n_times(img, operation, times)

    def apply_pattern(self, args: List[str]) -> Callable[[Image], Image]:
        pattern_type = args[0].upper()
        if pattern_type == 'CHECKERBOARD':
            return lambda img: self.apply_to_selected_area(img, lambda area: compose(area, Image.full(Point(0, 0), Point(area.w, area.h), 1), lambda x, y: y if (x + y) % 2 else x, 0))
        elif pattern_type == 'STRIPES':
            return lambda img: self.apply_to_selected_area(img, lambda area: compose(area, Image.full(Point(0, 0), Point(area.w, area.h), 1), lambda x, y: y if x % 2 else x, 0))
        # Add more patterns as needed

    def count(self, args: List[str]) -> Callable[[Image], int]:
        color = int(args[0])
        return lambda img: (print(f"The count of color {color} is: {np.sum(img.mask == color) }"), img)[-1]

    def group(self, args: List[str]) -> Callable[[Image], Image]:
        operations = args
        return lambda img: self.apply_operations(img, operations)

    def foreach(self, args: List[str]) -> Callable[[Image], Image]:
        color, operations = args[0], args[1]
        return lambda img: (color, operations)

    def if_condition(self, args: List[str]) -> Callable[[Image], Image]:
        condition, true_op, false_op = args[0], args[1], args[2]
        return lambda img: self.apply_conditional(img, condition, true_op, false_op)

    def while_loop(self, args: List[str]) -> Callable[[Image], Image]:
        condition, operation = args[0], args[1]
        return lambda img: self.apply_while(img, condition, operation)

    def convolve(self, args: List[str]) -> Callable[[Image], Image]:
        kernel_name, mode = args[0], args[1]
        return lambda img: self.apply_convolution(img, kernel_name, mode)

    def apply_filter(self, args: List[str]) -> Callable[[Image], Image]:
        filter_type, kernel_size = args[0], int(args[1])
        return lambda img: self.apply_image_filter(img, filter_type, kernel_size)

    def scale(self, args: List[str]) -> Callable[[Image], Image]:
        factor = float(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.scale_image(area, factor))


    def extrude(self, args: List[str]) -> Callable[[Image], Image]:
        direction, amount = args[0], int(args[1])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.extrude_image(area, direction, amount))

    def erode(self, args: List[str]) -> Callable[[Image], Image]:
        amount = int(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.erode_image(area, amount))

    def dilate(self, args: List[str]) -> Callable[[Image], Image]:
        amount = int(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.dilate_image(area, amount))

    def randomize(self, args: List[str]) -> Callable[[Image], Image]:
        colors = list(map(int, args))
        return lambda img: self.apply_to_selected_area(img, lambda area: self.randomize_colors(area, colors))

    def sort_colors(self, args: List[str]) -> Callable[[Image], Image]:
        direction, criteria = args[0], args[1]
        return lambda img: self.apply_to_selected_area(img, lambda area: self.sort_image_colors(area, direction, criteria))

    def blur(self, args: List[str]) -> Callable[[Image], Image]:
        radius = int(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.blur_image(area, radius))

    def sharpen(self, args: List[str]) -> Callable[[Image], Image]:
        amount = float(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.sharpen_image(area, amount))

    def invert_colors(self, args: List[str]) -> Callable[[Image], Image]:
        return lambda img: self.apply_to_selected_area(img, lambda area: invert(area))

    def adjust_brightness(self, args: List[str]) -> Callable[[Image], Image]:
        amount = float(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.adjust_image_brightness(area, amount))

    def adjust_contrast(self, args: List[str]) -> Callable[[Image], Image]:
        amount = float(args[0])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.adjust_image_contrast(area, amount))

    def apply_gradient(self, args: List[str]) -> Callable[[Image], Image]:
        start_color, end_color, direction = int(args[0]), int(args[1]), args[2]
        return lambda img: self.apply_to_selected_area(img, lambda area: self.create_gradient(area, start_color, end_color, direction))

    def create_border(self, args: List[str]) -> Callable[[Image], Image]:
        width, color = int(args[0]), int(args[1])
        return lambda img: self.apply_to_selected_area(img, lambda area: self.add_border(area, width, color))

    def crop(self, args: List[str]) -> Callable[[Image], Image]:
        x1, y1, x2, y2 = map(int, args)
        return lambda img: img.sub_image(Point(x1, y1), Point(x2-x1, y2-y1))

    def resize(self, args: List[str]) -> Callable[[Image], Image]:
        new_width, new_height, method = int(args[0]), int(args[1]), args[2]
        return lambda img: self.apply_to_selected_area(img, lambda area: self.resize_image(area, new_width, new_height, method))

    def apply_mask(self, args: List[str]) -> Callable[[Image], Image]:
        mask_name, operation = args[0], args[1]
        return lambda img: self.apply_mask_operation(img, mask_name, operation)

    def blend(self, args: List[str]) -> Callable[[Image], Image]:
        layer2_name, mode = args[0], args[1]
        return lambda img: self.blend_images(img, layer2_name, mode)

    def recursive(self, args: List[str]) -> Callable[[Image], Image]:
        operation, depth, condition = args[0], int(args[1]), args[2]
        return lambda img: self.apply_recursive(img, operation, depth, condition)

    def display(self, args: List[str]) -> Callable[[Image], None]:
        return lambda img: display_matrix(img)

    # Helper methods for applying operations
    def apply_to_selected_area(self, img: Image, func: Callable[[Image], Image]) -> Image:
        if self.selected_area:
            x1, y1, x2, y2 = self.selected_area
            selected = img.sub_image(Point(x1, y1), Point(x2-x1, y2-y1))
            modified = func(selected)
            result = img.copy()
            result.mask[y1:y2, x1:x2] = modified.mask
            return result
        return func(img)
    
    def apply_area(self, img: Image, func: Callable[[Image], Image]) -> Image:
        if self.selected_area:
            x1, y1, x2, y2 = self.selected_area
            selected = img.sub_image(Point(x1, y1), Point(x2-x1, y2-y1))
            modified = func(selected)
            result = img.copy()
            result.mask[y1:y2, x1:x2] = modified.mask
            return result
        return func(img)
    

    def apply_to_selected_area_move(self, img: Image, func: Callable[[Image], Image], shift: tuple[int, int]) -> Image:
        if self.selected_area:
            shift_x, shift_y = shift
            x1, y1, x2, y2 = self.selected_area
            result_matrix = img.copy()
            matrix = img.mask
            
            # Get the submatrix of the selected area
            submatrix = matrix[y1:y2, x1:x2].copy()
            
            # Calculate the new coordinates after the shift
            x1_new, y1_new = x1 + shift_x, y1 + shift_y
            x2_new, y2_new = x2 + shift_x, y2 + shift_y
            
            # Clear the original submatrix area
            result_matrix.mask[y1:y2, x1:x2] = 0
            
            # Calculate the valid source and destination regions
            src_x1 = max(0, -x1_new)
            src_y1 = max(0, -y1_new)
            src_x2 = min(x2 - x1, img.w - x1_new)
            src_y2 = min(y2 - y1, img.h - y1_new)
            
            dst_x1 = max(0, x1_new)
            dst_y1 = max(0, y1_new)
            dst_x2 = min(img.w, x2_new)
            dst_y2 = min(img.h, y2_new)
            
            # Calculate the dimensions of the area to be filled
            width = min(src_x2 - src_x1, dst_x2 - dst_x1)
            height = min(src_y2 - src_y1, dst_y2 - dst_y1)
            
            if width > 0 and height > 0:
                # Place the submatrix in the new position
                result_matrix.mask[dst_y1:dst_y1+height, dst_x1:dst_x1+width] = submatrix[src_y1:src_y1+height, src_x1:src_x1+width]
            
            return result_matrix
        return func(img)

    def apply_n_times(self, img: Image, operation: str, n: int) -> Image:
        result = img
        for _ in range(n):
            result = self.function_map[operation]([])(result)
        return result

    def apply_operations(self, img: Image, operations: List[str]) -> Image:
        result = img
        for op in operations:
            result = self.function_map[op]([])(result)
        return result

    def apply_to_color(self, img: Image, color: str, operation: str) -> Image:
        result = img.copy()
        mask = img.mask == int(color)
        selected = img.copy()
        selected.mask = np.where(mask, img.mask, 0)
        modified = self.function_map[operation]([])(selected)
        result.mask = np.where(mask, modified.mask, result.mask)
        return result

    def apply_conditional(self, img: Image, condition: str, true_op: str, false_op: str) -> Image:
        if self.evaluate_condition(img, condition):
            return self.function_map[true_op]([])(img)
        else:
            return self.function_map[false_op]([])(img)

    def apply_while(self, img: Image, condition: str, operation: str) -> Image:
        result = img
        while self.evaluate_condition(result, condition):
            result = self.function_map[operation]([])(result)
        return result

    def apply_convolution(self, img: Image, kernel_name: str, mode: str) -> Image:
        kernel = self.get_kernel(kernel_name)
        return convolve(img, kernel, mode)

    def apply_image_filter(self, img: Image, filter_type: str, kernel_size: int) -> Image:
        kernel = self.get_filter_kernel(filter_type, kernel_size)
        return convolve(img, kernel, 'SAME')

    def scale_image(self, img: Image, factor: float) -> Image:
        new_w, new_h = int(img.w * factor), int(img.h * factor)
        return resize(img, new_w, new_h, 'NEAREST')


    def extrude_image(self, img: Image, direction: str, amount: int) -> Image:
        # Implementation depends on how you want to handle extrusion
        pass

    def erode_image(self, img: Image, amount: int) -> Image:
        return erode(img, amount)

    def dilate_image(self, img: Image, amount: int) -> Image:
        return dilate(img, amount)

    def randomize_colors(self, img: Image, colors: List[int]) -> Image:
        return randomize(img, colors)

    def sort_image_colors(self, img: Image, direction: str, criteria: str) -> Image:
        # Implementation depends on how you want to handle color sorting
        pass

    def blur_image(self, img: Image, radius: int) -> Image:
        return blur(img, radius)

    def sharpen_image(self, img: Image, amount: float) -> Image:
        return sharpen(img, amount)

    def adjust_image_brightness(self, img: Image, amount: float) -> Image:
        # Implementation depends on how you want to handle brightness adjustment
        pass

    def adjust_image_contrast(self, img: Image, amount: float) -> Image:
        # Implementation depends on how you want to handle contrast adjustment
        pass

    def create_gradient(self, img: Image, start_color: int, end_color: int, direction: str) -> Image:
        return apply_gradient(img, start_color, end_color, direction)

    def add_border(self, img: Image, width: int, color: int) -> Image:
        return create_border(img, width, color)

    def resize_image(self, img: Image, new_width: int, new_height: int, method: str) -> Image:
        return resize(img, new_width, new_height, method)

    def apply_mask_operation(self, img: Image, mask_name: str, operation: str) -> Image:
        mask = self.get_mask(mask_name)
        return apply_mask(img, mask, self.function_map[operation]([]))

    def blend_images(self, img1: Image, img2_name: str, mode: str) -> Image:
        img2 = self.get_image(img2_name)
        return blend(img1, img2, mode)

    def apply_recursive(self, img: Image, operation: str, depth: int, condition: str) -> Image:
        if depth == 0 or not self.evaluate_condition(img, condition):
            return img
        result = self.function_map[operation]([])(img)
        return self.apply_recursive(result, operation, depth - 1, condition)

    def evaluate_condition(self, img: Image, condition: str) -> bool:
        # Implement condition evaluation logic here
        pass

    def get_kernel(self, kernel_name: str) -> np.ndarray:
        # Implement kernel retrieval logic here
        pass

    def get_filter_kernel(self, filter_type: str, kernel_size: int) -> np.ndarray:
        # Implement filter kernel creation logic here
        pass

    def get_mask(self, mask_name: str) -> Image:
        # Implement mask retrieval logic here
        pass

    def get_image(self, image_name: str) -> Image:
        # Implement image retrieval logic here
        pass