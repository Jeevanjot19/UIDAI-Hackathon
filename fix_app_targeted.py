"""
Fix app.py indentation - targeted approach
Strategy: Lines 383 onwards that have 12+ spaces should be reduced by 8
(except for legitimately nested blocks like try/with combinations)
"""

def fix_app_indentation():
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Empty lines - keep as is
        if not line.strip():
            fixed_lines.append(line)
            continue
        
        # Count indentation
        indent = len(line) - len(line.lstrip(' '))
        
        # Lines 1-382: keep as is (they're correct)
        if line_num <= 382:
            fixed_lines.append(line)
        # Lines 383+: check if they have problematic indentation
        else:
            # If line has 12+ spaces, reduce by 8
            # UNLESS it's legitimately nested (we'll be conservative)
            # Check context: if previous non-blank line ended with : or is multi-line continuation
            
            # Find previous non-blank line
            prev_line = ""
            for j in range(i-1, -1, -1):
                if lines[j].strip():
                    prev_line = lines[j]
                    break
            
            # If indent is 12+ and previous line doesn't suggest deep nesting
            if indent >= 12:
                # Check if this looks like legitimate nesting
                # Legitimate 12-space: inside try/with, or after multiple colons
                # Let's check the previous line
                prev_stripped = prev_line.rstrip()
                
                # If prev line ends with : or , or ( - might be continuation
                if prev_stripped.endswith((':',  ',', '(')):
                    # Might be legitimate - but let's check indent difference
                    prev_indent = len(prev_line) - len(prev_line.lstrip(' '))
                    expected_indent = prev_indent + 4
                    
                    if indent > expected_indent + 4:  # More than one level extra
                        # Remove 8 spaces
                        new_line = line[8:]
                        fixed_lines.append(new_line)
                    else:
                        fixed_lines.append(line)
                else:
                    # Previous line doesn't end with : or continuation
                    # This is likely wrong indentation - remove 8 spaces
                    if indent >= 20:
                        new_line = line[8:]
                    elif indent >= 12:
                        new_line = line[8:]
                    else:
                        new_line = line
                    fixed_lines.append(new_line)
            else:
                # Indent < 12, keep as is
                fixed_lines.append(line)
    
    # Write output
    with open('app_fixed.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Processed {len(lines)} lines")
    print(f"Fixed file saved as app_fixed.py")
    
    # Try to compile
    try:
        with open('app_fixed.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'app_fixed.py', 'exec')
        print("✓ Fixed file compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error at line {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False

if __name__ == "__main__":
    success = fix_app_indentation()
    if success:
        print("\nYou can now replace app.py with app_fixed.py")
        print("Command: copy app_fixed.py app.py")
