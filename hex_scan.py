path = r'C:\DACS\Pharmalink_GNN\web_app\templates\history.html'
with open(path, 'rb') as f:
    raw = f.read()

import re

# Find all occurrences of "iểm" and surrounding bytes (15 bytes before and after)
print("=== Searching for 'iem' suffix patterns ===")
# iểm in various encodings - try just searching for "m" preceded by something strange
for m in re.finditer(rb'.{0,20}[^\x00-\x7f\xe1\xbb\x83].{0,3}m.{0,5}', raw):
    region = m.group()
    if b'th' in region.lower() or b'Tr' in region or b'm<' in region or b'</th>' in region:
        print(f'POS {m.start()}: {region.hex()} | {repr(region)}')

# Search for the JavaScript header line with iểm
print("\n=== Searching for JS header line ===")
for m in re.finditer(rb'M.{0,10}OMIM.{0,30}m.{0,10}Tr', raw):
    print(f'POS {m.start()}: {m.group().hex()} | {repr(m.group())}')

# Search for Dự đoán and Đã biết 
print("\n=== Searching around badge area ===")
for m in re.finditer(rb'D.{0,5} \xc4\x91o.n|.{0,3}bi.{0,5}t', raw):
    print(f'POS {m.start()}: {m.group().hex()} | {repr(m.group())}')

# Show bytes at position of any "?" chars (0x3f) in the file
print("\n=== All ? chars (0x3f) in JS section ===")
# Find JS section
js_start = raw.find(b'const header = isDrug')
js_end = raw.find(b'document.getElementById', js_start) 
if js_start > 0:
    js_section = raw[js_start:js_start+800]
    print(f'JS section bytes: {js_section.hex()}')
    print(f'JS section text: {repr(js_section)}')
