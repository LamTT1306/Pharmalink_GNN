import re
path = r'C:/DACS/Pharmalink_GNN/web_app/templates/history.html'

def fix():
    with open(path, 'rb') as f:
        raw = f.read()

    print('--- SCANNING BYTES ---')
    # Find anything with high bytes around the targets
    for m in re.finditer(rb'Th[\x80-\xff]+i gian', raw):
        print('HEADER:', m.group().hex(), '|', repr(m.group()))
    for m in re.finditer(rb'[\x80-\xff]+i\xe1\xbb\x83m', raw):
        print('DIEM:', m.group().hex(), '|', repr(m.group()))
    for m in re.finditer(rb'[\x80-\xff]+\xc3\xa3 bi\xe1\xba\xbft', raw):
        print('DA BIET:', m.group().hex(), '|', repr(m.group()))

    print('--- APPLYING FIX ---')
    # Replace any sequence of high-bit/unknown bytes in the specific spots
    raw = re.sub(rb'Th[\x80-\xff\xef\xbf\xbd\x3f]+i gian', 'Thời gian'.encode('utf-8'), raw)
    raw = re.sub(rb'[\x80-\xff\xef\xbf\xbd\x3f]+i\xe1\xbb\x83m', 'Điểm'.encode('utf-8'), raw)
    raw = re.sub(rb'[\x80-\xff\xef\xbf\xbd\x3f]+\xc3\xa3 bi\xe1\xba\xbft', 'Đã biết'.encode('utf-8'), raw)

    with open(path, 'wb') as f:
        f.write(raw)

    print('--- VERIFICATION ---')
    with open(path, 'rb') as f:
        content = f.read()
    
    checks = {
        'Thời gian': 'Thời gian'.encode('utf-8'),
        'Điểm': 'Điểm'.encode('utf-8'),
        'Đã biết': 'Đã biết'.encode('utf-8')
    }
    for name, b_val in checks.items():
        found = b_val in content
        print(f'{"OK" if found else "FAIL"}: "{name}" {"found" if found else "NOT FOUND"}')

fix()
