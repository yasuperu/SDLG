import re

seq_length = 20

filename = "data/dataset_SVG_path.txt"

with open(filename, encoding='utf-8-sig') as f:
    text = f.read()

start_story = '| ' * seq_length
    
text = start_story + text
text = text.replace('d="', start_story)
text = text.replace('\n', ' ')
text = text.replace('\t', ' ')
text = text.replace('<?', ' ')
text = text.replace('<path', ' ')
text = re.sub(r'<?xml.*?>', '', text)
text = re.sub(r'<svg.*?>', '', text)
text = re.sub(r'</svg>', '', text)
text = re.sub(r'<g.*?>', '', text)
text = re.sub(r'</g>', '', text)
text = re.sub(r'<path.*?>', '', text)
text = re.sub(r'</path>', '', text)
text = re.sub(r'<text.*?>', '', text)
text = re.sub(r'</text>', '', text)
text = re.sub(r'fill="*.*?>', '', text)
text = re.sub(r'Z"', '', text)
text = re.sub(r'stroke-opacity="*.*?>', '', text)
text = re.sub(r'fill-opacity="*.*?>', '', text)
text = re.sub(r'stroke="*.*?>', '', text)
text = re.sub(r'stroke-width="*.*?>', '', text)