"""
Smart indentation fixer - analyzes Python code structure and fixes indentation
"""
import re

def smart_fix_indent(input_file, output_file):
    """
    Fix indentation by understanding Python structure
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    indent_level = 0
    prev_line_content = ""
    
    for i, line in enumerate(lines):
        # Handle blank lines
        if line.strip() == '':
            fixed_lines.append('')
            continue
        
        content = line.lstrip()
        orig_indent = len(line) - len(content)
        
        # Dedent keywords
        if re.match(r'^(elif |else:|except |except:|finally:)', content):
            if indent_level > 0:
                indent_level -= 4
        
        # Special case: detect if we're starting a new top-level section
        # Comments at odd indentation (like 12 spaces) should be at the current level
        if content.startswith('#') and orig_indent == 12:
            # This is likely a misindented comment - use current indent level
            pass
        elif orig_indent >= 12 and not (prev_line_content.rstrip().endswith(':') or 
                                        prev_line_content.rstrip().endswith(',') or
                                        prev_line_content.rstrip().endswith('(') or
                                       prev_line_content.strip().startswith('"""')):
            # Line has 12+ spaces and previous line doesn't suggest it should be indented
            # Reduce by 12 to get correct indentation
            indent_level = max(0, orig_indent - 12)
        
        # Apply current indentation
        fixed_line = ' ' * indent_level + content
        fixed_lines.append(fixed_line.rstrip() + '\n')
        
        # Increase indent after colon
        if content.rstrip().endswith(':') and not content.strip().startswith('#'):
            indent_level += 4
        
        # Track previous line
        prev_line_content = content
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed {len(fixed_lines)} lines -> {output_file}")

if __name__ == "__main__":
    smart_fix_indent('app.py', 'app_smart_fixed.py')
    
    # Test compilation
    try:
        with open('app_smart_fixed.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'app_smart_fixed.py', 'exec')
        print("✓ File compiles successfully!")
    except SyntaxError as e:
        print(f"✗ Syntax error at line {e.lineno}: {e.msg}")
