"""
FINAL indentation fix for app.py
Simple rule: All lines from line 383 onwards that have 12+ spaces get 12 spaces removed
EXCEPT lines 1-382 which are correct as-is (they're before the problem starts)
"""

def final_fix():
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Lines 1-382: keep as-is (they're correct)
        if line_num <= 382:
            fixed_lines.append(line)
            continue
        
        # Line 383+: if indent >= 12, remove 12 spaces
        if line.strip():  # Non-empty line
            indent = len(line) - len(line.lstrip(' '))
            if indent >= 12:
                # Remove exactly 12 spaces
                fixed_line = line[12:]
                fixed_lines.append(fixed_line)
            else:
                # Keep as-is
                fixed_lines.append(line)
        else:
            # Empty line
            fixed_lines.append(line)
    
    # Save
    with open('app_FIXED.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed {len(lines)} lines")
    print("Saved to: app_FIXED.py")
    
    # Test
    try:
        with open('app_FIXED.py', 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, 'app_FIXED.py', 'exec')
        print("✓✓✓ SUCCESS! File compiles perfectly!")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error at line {e.lineno}: {e.msg}")
        return False

if __name__ == "__main__":
    if final_fix():
        print("\nREADY TO USE!")
        print("To replace your file, run:")
        print("   copy app_FIXED.py app.py")
