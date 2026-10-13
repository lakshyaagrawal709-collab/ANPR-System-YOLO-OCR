content = open('app.py', encoding='utf-8').read()
fixed = content.replace('use_container_width=True', 'width="stretch"').replace('use_container_width=False', 'width="content"')
open('app.py', 'w', encoding='utf-8').write(fixed)
print(f"Fixed {content.count('use_container_width')} occurrences of use_container_width")
