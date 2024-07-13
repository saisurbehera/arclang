from arclang.function import *
from arclang.image import Image, Point 

import numpy as np
import numpy as np
from collections import namedtuple
from typing import List, Tuple, Callable, Any


class CommandMapper:
    """ This class only acts as an wrapper between the arclang functions and the DSL code. The functions are pretty hard to apply directly.
    This class enables us to abstract away most of the functions and we can focus on adding to our database. 
    
    This class is not responsible for state management. It is only responsible for taking a command 
    and outputing a number, modifying and image object or creating a list.
    
    It also handles the selection and apply all operation  
    """
    
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
    
    def select(self, args: List[str]) -> Callable[[Image], Image]:
        x1, y1, x2, y2 = map(int, args)
        self.selected_area = (x1, y1, x2, y2)
        return lambda img: img