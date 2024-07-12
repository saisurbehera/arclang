from arclang.function import *
from arclang.image import Image, Point 

import numpy as np
from collections import namedtuple
from typing import List, Tuple, Any

class CommandMapper:
    def __init__(self, image):
        self.image = image
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

    def select(self, args: List[str]) -> Image:
        x1, y1, x2, y2 = map(int, args)
        self.selected_area = (x1, y1, x2, y2)
        return self.image
    
    def deselect(self, args: List[str]) -> Image:
        self.selected_area = None
        return self.image

    def split(self, args: List[str]) -> Image:
        raise NotImplementedError()

    def fill(self, args: List[str]) -> Image:
        color = int(args[0])
        return self.apply_to_selected_area(lambda area: Image.full(Point(area.x, area.y), Point(area.w, area.h), color))

    def expand(self, args: List[str]) -> Image:
        size = int(args[0])
        return self.apply_to_selected_area(lambda area: extend2(area, Image.full(Point(area.x-size, area.y-size), Point(area.w+2*size, area.h+2*size), 0)))

    def shrink(self, args: List[str]) -> Image:
        size = int(args[0])
        return self.apply_to_selected_area(lambda area: area.sub_image(Point(size, size), Point(area.w-2*size, area.h-2*size)))

    def clear(self, args: List[str]) -> Image:
        return self.apply_to_selected_area(lambda area: Image.empty(area.x, area.y, area.w, area.h))

    def copy(self, args: List[str]) -> Image:
        x, y = map(int, args)
        return self.apply_to_selected_area(lambda area: compose_id(area, move(area, (x, y)), 0))

    def move(self, args: List[str]) -> Image:
        x, y = map(int, args)
        return self.apply_to_selected_area_move(lambda area: move(area, (x, y)), (x, y))

    def rotate(self, args: List[str]) -> Image:
        angle = int(args[0])
        return self.apply_to_selected_area(lambda area: rigid(area, angle // 90))

    def flip(self, args: List[str]) -> Image:
        axis = args[0].upper()
        return self.apply_to_selected_area(lambda area: rigid(area, 4 if axis == 'HORIZONTAL' else 5))

    def replacecol(self, args: List[str]) -> Image:
        old_color, new_color = map(int, args[:2])
        if len(args) > 2 and args[2] == 'IN':
            x1, y1, x2, y2 = args[3:]  # Keep these as strings for now
            return self.replace_in_area(old_color, new_color, x1, y1, x2, y2)
        return self.replace_color(old_color, new_color)

    def replace_in_area(self, old_color: int, new_color: int, x1: str, y1: str, x2: str, y2: str) -> Image:
        new_img = self.image.copy()
        x1 = int(x1.replace('[COL]', str(old_color)))
        x2 = int(x2.replace('[COL]', str(old_color)))
        y1, y2 = int(y1), int(y2)
        mask = (new_img.mask[y1:y2, x1:x2] == old_color)
        new_img.mask[y1:y2, x1:x2][mask] = new_color
        return new_img

    def replace_color(self, old_color: int, new_color: int) -> Image:
        new_img = self.image.copy()
        new_img.mask[new_img.mask == old_color] = new_color
        return new_img

    def overlay(self, args: List[str]) -> Image:
        return self.apply_to_selected_area(lambda area: compose_id(area, area, 3))

    def subtract(self, args: List[str]) -> Image:
        return self.apply_to_selected_area(lambda area: compose_id(area, invert(area), 2))

    def intersect(self, args: List[str]) -> Image:
        return self.apply_to_selected_area(lambda area: compose_id(area, area, 2))

    def repeat(self, args: List[str]) -> Image:
        operation, times = args[0], int(args[1])
        return self.apply_n_times(operation, times)

    def apply_pattern(self, args: List[str]) -> Image:
        pattern_type = args[0].upper()
        if pattern_type == 'CHECKERBOARD':
            return self.apply_to_selected_area(lambda area: compose(area, Image.full(Point(0, 0), Point(area.w, area.h), 1), lambda x, y: y if (x + y) % 2 else x, 0))
        elif pattern_type == 'STRIPES':
            return self.apply_to_selected_area(lambda area: compose(area, Image.full(Point(0, 0), Point(area.w, area.h), 1), lambda x, y: y if x % 2 else x, 0))
        # Add more patterns as needed

    def count(self, args: List[str]) -> int:
        color = int(args[0])
        count = np.sum(self.image.mask == color)
        return count

    def group(self, args: List[str]) -> Image:
        operations = args
        return self.apply_operations(operations)

    def foreach(self, args: List[str]) -> Tuple[str, List[str]]:
        color, operations = args[0], args[1]
        return color, operations

    def while_loop(self, args: List[str]) -> Image:
        condition, operation = args[0], args[1]
        return self.apply_while(condition, operation)

    def convolve(self, args: List[str]) -> Image:
        kernel_name, mode = args[0], args[1]
        return self.apply_convolution(kernel_name, mode)

    def apply_filter(self, args: List[str]) -> Image:
        filter_type, kernel_size = args[0], int(args[1])
        return self.apply_image_filter(filter_type, kernel_size)

    def scale(self, args: List[str]) -> Image:
        factor = float(args[0])
        return self.apply_to_selected_area(lambda area: self.scale_image(area, factor))

    def extrude(self, args: List[str]) -> Image:
        direction, amount = args[0], int(args[1])
        return self.apply_to_selected_area(lambda area: self.extrude_image(area, direction, amount))

    def randomize(self, args: List[str]) -> Image:
        colors = list(map(int, args))
        return self.apply_to_selected_area(lambda area: self.randomize_colors(area, colors))

    def sort_colors(self, args: List[str]) -> Image:
        direction, criteria = args[0], args[1]
        return self.apply_to_selected_area(lambda area: self.sort_image_colors(area, direction, criteria))

    def invert_colors(self, args: List[str]) -> Image:
        return self.apply_to_selected_area(lambda area: invert(area))

    def create_border(self, args: List[str]) -> Image:
        width, color = int(args[0]), int(args[1])
        return self.apply_to_selected_area(lambda area: self.add_border(area, width, color))

    def crop(self, args: List[str]) -> Image:
        x1, y1, x2, y2 = map(int, args)
        return self.image.sub_image(Point(x1, y1), Point(x2-x1, y2-y1))

    def resize(self, args: List[str]) -> Image:
        new_width, new_height, method = int(args[0]), int(args[1]), args[2]
        return self.apply_to_selected_area(lambda area: self.resize_image(area, new_width, new_height, method))

    def apply_mask(self, args: List[str]) -> Image:
        mask_name, operation = args[0], args[1]
        return self.apply_mask_operation(mask_name, operation)

    def blend(self, args: List[str]) -> Image:
        layer2_name, mode = args[0], args[1]
        return self.blend_images(layer2_name, mode)

    def recursive(self, args: List[str]) -> Image:
        operation, depth, condition = args[0], int(args[1]), args[2]
        return self.apply_recursive(operation, depth, condition)

    def display(self, args: List[str]) -> None:
        display_matrix(self.image)
        return self.image

    # Helper methods for applying operations
    def apply_to_selected_area(self, func):
        if self.selected_area:
            x1, y1, x2, y2 = self.selected_area
            selected = self.image.sub_image(Point(x1, y1), Point(x2-x1, y2-y1))
            modified = func(selected)
            result = self.image.copy()
            result.mask[y1:y2, x1:x2] = modified.mask
            self.image = result
            return result
        print("applying glip")
        self.image = func(self.image) 
        return self.image

    def apply_to_selected_area_move(self, func, shift):
        if self.selected_area:
            shift_x, shift_y = shift
            x1, y1, x2, y2 = self.selected_area
            result_matrix = self.image.copy()
            matrix = self.image.mask
            
            submatrix = matrix[y1:y2, x1:x2].copy()
            
            x1_new, y1_new = x1 + shift_x, y1 + shift_y
            x2_new, y2_new = x2 + shift_x, y2 + shift_y
            
            result_matrix.mask[y1:y2, x1:x2] = 0
            
            src_x1 = max(0, -x1_new)
            src_y1 = max(0, -y1_new)
            src_x2 = min(x2 - x1, self.image.w - x1_new)
            src_y2 = min(y2 - y1, self.image.h - y1_new)
            
            dst_x1 = max(0, x1_new)
            dst_y1 = max(0, y1_new)
            dst_x2 = min(self.image.w, x2_new)
            dst_y2 = min(self.image.h, y2_new)
            
            width = min(src_x2 - src_x1, dst_x2 - dst_x1)
            height = min(src_y2 - src_y1, dst_y2 - dst_y1)
            
            if width > 0 and height > 0:
                result_matrix.mask[dst_y1:dst_y1+height, dst_x1:dst_x1+width] = submatrix[src_y1:src_y1+height, src_x1:src_x1+width]
            
            return result_matrix
        return func(self.image)

    def apply_n_times(self, operation: str, n: int) -> Image:
        result = self.image
        for _ in range(n):
            result = self.function_map[operation]([])(result)
        return result

    def apply_operations(self, operations: List[str]) -> Image:
        result = self.image
        for op in operations:
            result = self.function_map[op]([])
        return result

    def apply_to_color(self, color: str, operation: str) -> Image:
        result = self.image.copy()
        mask = self.image.mask == int(color)
        selected = self.image.copy()
        selected.mask = np.where(mask, self.image.mask, 0)
        modified = self.function_map[operation]([])
        result.mask = np.where(mask, modified.mask, result.mask)
        return result

    def apply_while(self, condition: str, operation: str) -> Image:
        result = self.image
        while self.evaluate_condition(result, condition):
            result = self.function_map[operation]([])
        return result

    def apply_convolution(self, kernel_name: str, mode: str) -> Image:
        kernel = self.get_kernel(kernel_name)
        return convolve(self.image, kernel, mode)

    def apply_image_filter(self, filter_type: str, kernel_size: int) -> Image:
        kernel = self.get_filter_kernel(filter_type, kernel_size)
        return convolve(self.image, kernel, 'SAME')

    def scale_image(self, img: Image, factor: float) -> Image:
        new_w, new_h = int(img.w * factor), int(img.h * factor)
        return resize(img, new_w, new_h, 'NEAREST')

    def extrude_image(self, img: Image, direction: str, amount: int) -> Image:
        # Implementation depends on how you want to handle extrusion
        # This is a placeholder implementation
        result = img.copy()
        if direction.upper() == 'UP':
            result.mask = np.pad(img.mask, ((amount, 0), (0, 0)), mode='edge')[:img.h, :]
        elif direction.upper() == 'DOWN':
            result.mask = np.pad(img.mask, ((0, amount), (0, 0)), mode='edge')[-img.h:, :]
        elif direction.upper() == 'LEFT':
            result.mask = np.pad(img.mask, ((0, 0), (amount, 0)), mode='edge')[:, :img.w]
        elif direction.upper() == 'RIGHT':
            result.mask = np.pad(img.mask, ((0, 0), (0, amount)), mode='edge')[:, -img.w:]
        return result

    def randomize_colors(self, img: Image, colors: List[int]) -> Image:
        return randomize(img, colors)

    def sort_image_colors(self, img: Image, direction: str, criteria: str) -> Image:
        # This is a placeholder implementation
        sorted_colors = np.sort(np.unique(img.mask))
        if direction.upper() == 'DESCENDING':
            sorted_colors = sorted_colors[::-1]
        color_map = {old: new for old, new in zip(np.unique(img.mask), sorted_colors)}
        result = img.copy()
        for old, new in color_map.items():
            result.mask[img.mask == old] = new
        return result

    def add_border(self, img: Image, width: int, color: int) -> Image:
        return create_border(img, width, color)

    def resize_image(self, img: Image, new_width: int, new_height: int, method: str) -> Image:
        return resize(img, new_width, new_height, method)

    def apply_mask_operation(self, mask_name: str, operation: str) -> Image:
        mask = self.get_mask(mask_name)
        return apply_mask(self.image, mask, self.function_map[operation]([]))

    def blend_images(self, img2_name: str, mode: str) -> Image:
        img2 = self.get_image(img2_name)
        return blend(self.image, img2, mode)

    def apply_recursive(self, operation: str, depth: int, condition: str) -> Image:
        if depth == 0 or not self.evaluate_condition(self.image, condition):
            return self.image
        result = self.function_map[operation]([])
        self.image = result  # Update the image for the next recursion
        return self.apply_recursive(operation, depth - 1, condition)

    def evaluate_condition(self, img: Image, condition: str) -> bool:
        # This is a placeholder implementation
        # You should implement a proper condition evaluator based on your needs
        if condition.upper() == 'ALWAYS':
            return True
        elif condition.upper() == 'NEVER':
            return False
        else:
            # Implement more complex condition evaluation here
            return False

    def get_kernel(self, kernel_name: str) -> np.ndarray:
        # Implement kernel retrieval logic here
        # This is a placeholder implementation
        if kernel_name.upper() == 'BLUR':
            return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        elif kernel_name.upper() == 'SHARPEN':
            return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

    def get_filter_kernel(self, filter_type: str, kernel_size: int) -> np.ndarray:
        # Implement filter kernel creation logic here
        # This is a placeholder implementation
        if filter_type.upper() == 'GAUSSIAN':
            x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
            g = np.exp(-((x**2 + y**2)/(2.0*kernel_size**2)))
            return g / g.sum()
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def get_mask(self, mask_name: str) -> Image:
        # Implement mask retrieval logic here
        # This is a placeholder implementation
        raise NotImplementedError("Mask retrieval not implemented")

    def get_image(self, image_name: str) -> Image:
        # Implement image retrieval logic here
        # This is a placeholder implementation
        raise NotImplementedError("Image retrieval not implemented")

    def apply_gradient(self, start_color: int, end_color: int, direction: str) -> Image:
        return self.apply_to_selected_area(lambda area: apply_gradient(area, start_color, end_color, direction))

    def blur_image(self, radius: int) -> Image:
        return self.apply_to_selected_area(lambda area: blur(area, radius))

    def sharpen_image(self, amount: float) -> Image:
        return self.apply_to_selected_area(lambda area: sharpen(area, amount))

    def adjust_image_brightness(self, amount: float) -> Image:
        # This is a placeholder implementation
        result = self.image.copy()
        result.mask = np.clip(result.mask.astype(float) + amount, 0, 255).astype(int)
        return result

    def adjust_image_contrast(self, amount: float) -> Image:
        # This is a placeholder implementation
        result = self.image.copy()
        factor = (259 * (amount + 255)) / (255 * (259 - amount))
        result.mask = np.clip(factor * (result.mask.astype(float) - 128) + 128, 0, 255).astype(int)
        return result

    def erode_image(self, amount: int) -> Image:
        return self.apply_to_selected_area(lambda area: erode(area, amount))

    def dilate_image(self, amount: int) -> Image:
        return self.apply_to_selected_area(lambda area: dilate(area, amount))