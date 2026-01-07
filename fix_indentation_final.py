"""
Fix indentation by removing excess leading whitespace and reindenting properly
This looks for lines with 12+ spaces of indentation and normalizes them
"""

def fix_indentation(input_file, output_file):
    """
    Remove excess indentation - fixes the issue where lines have 12 spaces instead of 0-8
    
    Strategy: Lines that start with 12 spaces should start with 0
              Lines that start with 16 spaces should start with 4
              Lines that start with 20 spaces should start with 8
              Pattern: Remove 12 spaces from each line that has them
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if not line.strip():  # Empty line
            fixed_lines.append('')
            continue
        
        # Count leading spaces
        leading_spaces = len(line) - len(line.lstrip(' '))
        
        # If line has exactly 12 spaces or more, reduce by 12
        # But preserve some minimum indentation for nested blocks
        if leading_spaces == 12:
            # Remove all 12 spaces
            fixed_line = line[12:]
        elif leading_spaces > 12:
            # Remove 12 spaces but keep the rest
            fixed_line = line[12:]
        else:
            # Keep as-is
            fixed_line = line
        
        fixed_lines.append(fixed_line)
    
    # Write the fixed content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"Processed {len(fixed_lines)} lines")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    fix_indentation('app.py', 'app_fixed.py')
    print("\nValidating the fixed file...")
    
    # Try to compile it
    try:
        import ast
        with open('app_fixed.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'app_fixed.py', 'exec')
        print("✓ Fixed file compiles successfully!")
    except SyntaxError as e:
        print(f"✗ Syntax error at line {e.lineno}: {e.msg}")
        print(f"   {e.text}")
        # Show context
        with open('app_fixed.py', 'r') as f:
            lines = f.readlines()
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            print("\nContext:")
            for i in range(start, end):
                marker = ">>>" if i == e.lineno - 1 else "   "
                print(f"{marker} {i+1}: {lines[i].rstrip()}")
