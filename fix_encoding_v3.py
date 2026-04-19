import re
path = r'C:/DACS/Pharmalink_GNN/web_app/templates/history.html'

with open(path, 'rb') as f:
    raw = f.read()

print('--- SCANNING BYTES ---')
# Check for remnants of "iểm" and "ã biết"
for m in re.finditer(rb'.{0,10}i\xe1\xbb\x83m', raw):
    print('DIEM:', m.group().hex(), '|', repr(m.group()))
for m in re.finditer(rb'.{0,10}\xc3\xa3 bi', raw):
    print('DA BIET:', m.group().hex(), '|', repr(m.group()))

patterns = [
    (rb'Th[\x80-\xff\xef\xbf\xbd\x3f]+i gian', 'Thời gian'),
    (rb'[\x80-\xff\xef\xbf\xbd\x3f]+i\xe1\xbb\x83m', 'Điểm'),
    (rb'[\x80-\xff\xef\xbf\xbd\x3f]+\xc3\xa3 bi\xe1\xba\xbft', 'Đã biết'),
    (rb'.\?i\xe1\xbb\x83m', 'Điểm'), # Catch single question mark if that's what's there
    (rb'.\?\xc3\xa3 bi\xe1\xba\xbft', 'Đã biết') # Catch single question mark
]

for p, replacement in patterns:
    raw = re.sub(p, replacement.encode('utf-8'), raw)

with open(path, 'wb') as f:
    f.write(raw)

print('--- FINAL CHECK ---')
with open(path, 'rb') as f:
    content = f.read()
for s in ['Thời gian', 'Điểm', 'Đã biết']:
    print(f'"{s}": {"OK" if s.encode("utf-8") in content else "FAIL"}')
