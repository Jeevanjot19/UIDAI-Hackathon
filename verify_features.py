"""
Quick feature verification script
Tests that all major sections exist in app.py
"""

import re

def verify_features():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    features_to_check = {
        '2 New Pages': [
            'Model Trust Center',
            'National Intelligence',
        ],
        'Enhanced Features': [
            'Decision Quality Metrics',
            'System Resilience & Failure Recovery',
            'System Evolution Roadmap',
            'Ethical & Constitutional Alignment',
        ],
        'Specific Implementations': [
            'District Confidence Scoring',
            'Model Failure Modes',
            'Migration Intelligence',
            'Urban Stress Signals',
            'Regret Risk',
            'Intervention Effectiveness Tracking',
            'Stress Propagation Analysis',
            'Human-in-the-Loop Decision Markers',
            'Privacy-by-Design Principles',
        ]
    }
    
    print("="*60)
    print("FEATURE VERIFICATION REPORT")
    print("="*60)
    
    all_found = True
    
    for category, items in features_to_check.items():
        print(f"\n{category}:")
        for item in items:
            found = item in content
            status = "✅ FOUND" if found else "❌ MISSING"
            print(f"  {status}: {item}")
            if not found:
                all_found = False
    
    # Count pages in navigation
    nav_match = re.search(r'page = st\.sidebar\.radio\((.*?)\)', content, re.DOTALL)
    if nav_match:
        nav_text = nav_match.group(1)
        page_count = nav_text.count('"')
        print(f"\nTotal Pages in Navigation: {page_count}")
    
    # Count total lines
    line_count = content.count('\n')
    print(f"Total Lines of Code: {line_count}")
    
    print("\n" + "="*60)
    if all_found:
        print("✅ ALL FEATURES VERIFIED - READY FOR SUBMISSION")
    else:
        print("❌ SOME FEATURES MISSING - REVIEW NEEDED")
    print("="*60)

if __name__ == "__main__":
    verify_features()
