import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix garbled emojis in navigation
content = content.replace('"� Model Trust Center"', '"🔬 Model Trust Center"')
content = content.replace('"�📋 About"', '"📋 About"')

# Remove any double commas
content = re.sub(r',,\s*', ', ', content)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fixed emoji encoding issues')
print('✅ Navigation should now display correctly')
