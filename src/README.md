## ArcLang command line 


The goal of this is to be able to create a SQL type language for solving Arc type languages. 

It will follow a syntax like python for blocks. Example:
```
INI
SELECT 0 5 3 6
    FILL 3
SELECT 0 5 3 6
    FILL 4
X = SPLIT DFS
FOREACH Y X
    HALF Y
    REPLACECLR 1 3
    STACK Y INI

```

List of primitives:
- Colors
- Rules( conditionals)
- Loops
- Function 
- Pattern
- Templates

List of functions you can use:
- SELECT
- FILL
- SPLIT
- EXPAND
- CENTER
- ORDER
- BOUNDARY
- INSIDE
- OUTSIDE
- SHRINK
- CLEAR
- COPY
- MOVE 
- ROTATE
- MAX
- MIN
- ODD_ONE_OUT
- FLIP
- REPLACE
- REPLACECLR
- OVERLAY
- SUBTRACT
- APPLY_PATERN
- IF
- COUNT
- GROUP
- FOREACH
- WHILE
- CONNECT
- CONVOLVE
- SCALE
- SHIFT
- EXTRUDE
- RANDOMIZE
- SORT_COLORS
- CROP
- RESIZE
- APPLY_MASK
- BLEND
- RECURSIVE
- DISPLAY

# ArcLang Command Line

ArcLang is a SQL-type language designed for solving Arc-type problems. It uses a Python-like syntax for blocks and provides a set of functions to manipulate images.

## Syntax Example

```ini
SELECT 0 5 3 6
    FILL 3
SELECT 0 5 3 6
    FILL 4
X = SPLIT DFS
FOREACH Y X
    HALF Y
    REPLACECLR 1 3
    STACK Y
```

## Available Functions

### Implemented Functions
These functions can be directly implemented using the provided Image class and its methods:

1. SELECT: Can be implemented using Image.sub_image()
    - Selected Done
2. FILL: Can be implemented using Image.full()
    - Fill selected done 
3. SPLIT: Can be implemented using list_components() or split_cols()
4. CENTER: Can be implemented using center()
5. BOUNDARY: Can be implemented using border()
6. INSIDE: Can be implemented using interior()
7. OUTSIDE: Can be implemented using invert()
8. SHRINK: Can be implemented using compress()
    - Shrink implemented
9. CLEAR: Can be implemented using Image.empty()
    - Clear implemented
10. COPY: Can be implemented using Image.copy()
    - Testing required
11. MOVE: Can be implemented using align() or by modifying x and y attributes
    - Move implemented
12. ROTATE: Can be implemented using rigid()
    - Rotate implmented
13. FLIP: Can be implemented using rigid()
    - Flip implemented
14. REPLACE: Can be implemented using replace_cols()
15. REPLACECOL: Can be implemented using filter_col() and compose()
    - Replace Colors implemented
16. OVERLAY: Can be implemented using compose()
17. SUBTRACT: Can be implemented using compose() with a custom function
18. COUNT: Can be implemented using count()
19. CONNECT: Can be implemented using connect()

### Functions Requiring Modification or Combination
These functions can be implemented by modifying or combining existing functions:

1. EXPAND: Can be implemented by modifying extend() or extend2()
2. ORDER: Can be implemented using sort and existing functions
3. MAX/MIN: Can be implemented using pick_max() with custom criteria
4. ODD_ONE_OUT: Can be implemented using pick_unique()
5. APPLY_PATTERN: Can be implemented using replace_template()
6. GROUP: Can be implemented using list_components() and compose()
7. FOREACH: Can be implemented as a loop over components or colors
8. WHILE: Can be implemented as a Python while loop
9. CONVOLVE: Can be implemented using numpy operations on the mask
10. SCALE: Can be implemented by modifying transform()
11. SHIFT: Can be implemented using move()
12. EXTRUDE: Can be implemented by extending existing 2D operations
13. SORT_COLORS: Can be implemented using split_cols() and compose()
14. CROP: Can be implemented using sub_image()
15. RESIZE: Can be implemented by modifying transform()
16. APPLY_MASK: Can be implemented using compose() with a custom function
17. BLEND: Can be implemented using compose() with a custom blending function

### Functions Requiring Additional Implementation
These functions would require additional implementation or are not directly supported by the current codebase:

1. RANDOMIZE: Would require implementing a random number generator
2. RECURSIVE: Would require implementing a recursive function caller

### Control Structures
The following control structures can be implemented using Python-like syntax:

1. IF: Can be implemented using Python's if statements
2. FOREACH: Can be implemented using Python's for loops
3. WHILE: Can be implemented using Python's while loops

## Usage

To use ArcLang, write your commands in a file with a .arc extension. Then run the ArcLang interpreter with your file as an argument:

```
python arclang_interpreter.py your_file.arc
```

## Limitations

- The current implementation is based on 2D image operations. 3D operations would require significant extensions.
- Some functions may have performance limitations for very large images.
- The language is designed for Arc-type problems and may not be suitable for general-purpose programming.

## Contributing

Contributions to extend the functionality of ArcLang are welcome. Please submit pull requests with clear descriptions of new functions or improvements.