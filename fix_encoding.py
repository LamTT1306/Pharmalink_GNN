path = r'C:/DACS/Pharmalink_GNN/web_app/templates/history.html'
with open(path, 'rb') as f:
    raw = f.read()

import re
print('--- SCANNING BYTES ---')
for m in re.finditer(rb'Th.{0,5}i gian', raw):
    print('HEADER:', m.group().hex(), '|', repr(m.group()))

for m in re.finditer(rb'.{0,4}i\xe1\xbb\x83m', raw):
    print('DIEM:', m.group().hex(), '|', repr(m.group()))

for m in re.finditer(rb'.{0,4}\xc3\xa3 bi', raw):
    print('DA BIET:', m.group().hex(), '|', repr(m.group()))

print('--- APPLYING FIX ---')
raw = re.sub(rb'Th[\xef\xbf\xbd\x3f]+i gian', 'Thời gian'.encode('utf-8'), raw)
raw = re.sub(rb'[\xef\xbf\xbd\x3f]+i\xe1\xbb\x83m', 'Điểm'.encode('utf-8'), raw)
raw = re.sub(rb'[\xef\xbf\xbd\x3f]+\xc3\xa3 bi\xe1\xba\xbft', 'Đã biết'.encode('utf-8'), raw)

with open(path, 'wb') as f:
    f.write(raw)

print('--- VERIFICATION ---')
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

checks = ['Thời gian', 'Điểm', 'Đã biết']
for c in checks:
    found = c in content
    print(f'{"OK" if found else "FAIL"}: "{c}" {"found" if found else "NOT FOUND"}')
