"""
Proper indentation fixer that understands Python block structure
"""

def fix_python_indentation(filename):
    """Fix indentation by tracking block depth properly"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    indent_stack = [0]  # Stack of indentation levels
    
    for i, line in enumerate(lines):
        # Empty or whitespace-only lines
        if not line.strip():
            fixed_lines.append('\n')
            continue
        
        content = line.lstrip()
        original_indent = len(line) - len(content)
        
        # Determine correct indentation based on context
        # Dedent keywords
        if content.startswith(('elif ', 'else:', 'except:', 'except ', 'finally:')):
            # Pop back to parent level
            if len(indent_stack) > 1:
                indent_stack.pop()
            current_indent = indent_stack[-1]
        else:
            current_indent = indent_stack[-1]
        
        # Write line with correct indentation
        fixed_line = ' ' * current_indent + content
        fixed_lines.append(fixed_line)
        
        # Update indent stack for next line
        if content.rstrip().endswith(':') and not content.strip().startswith('#'):
            # Starting a new block
            indent_stack.append(current_indent + 4)
        elif content.startswith(('elif ', 'else:', 'except:', 'except ', 'finally:')):
            # These start a new block after dedenting
            indent_stack.append(current_indent + 4)
        
        # Special case: unindent after return, pass, continue, break if not in a block
        if content.strip() in ('pass', 'continue', 'break') or content.startswith('return '):
            # Might close a block, but we need more context to be sure
            pass
    
    return fixed_lines

if __name__ == "__main__":
    print("Fixing indentation in app.py...")
    fixed = fix_python_indentation('app.py')
    
    with open('app_structure_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed)
    
    print(f"Wrote {len(fixed)} lines to app_structure_fixed.py")
    
    # Test compilation
    try:
        with open('app_structure_fixed.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'app_structure_fixed.py', 'exec')
        print("✓ File compiles successfully!")
    except SyntaxError as e:
        print(f"✗ Syntax error at line {e.lineno}: {e.msg}")
