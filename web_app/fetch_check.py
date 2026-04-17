import urllib.request

try:
    req = urllib.request.Request('http://127.0.0.1:5000/predict', headers={'Cookie': ''})
    with urllib.request.urlopen(req, timeout=5) as r:
        html = r.read().decode('utf-8')
    
    # Check tab-matrix
    marker = 'tab-matrix'
    idx = html.find(marker)
    if idx >= 0:
        print('tab-matrix found at char', idx)
        snippet = html[idx:idx+400]
        print('Snippet:', snippet[:400])
    else:
        print('tab-matrix NOT in rendered HTML')
    
    # Check {% block content %} area
    block_idx = html.find('tab-single')
    print('tab-single found:', block_idx >= 0)
    
    # Check for Jinja2 errors
    if 'Internal Server Error' in html:
        print('SERVER ERROR in response')
    elif 'Traceback' in html:
        print('TRACEBACK in response')
    else:
        print('Page rendered OK, length:', len(html))
        
except Exception as e:
    print('Error:', type(e).__name__, str(e))
