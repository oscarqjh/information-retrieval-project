from flask import Flask, render_template, request, jsonify
import requests
import time

app = Flask(__name__)

SOLR_URL = "http://localhost:8983/solr/opinions"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '*')
    sentiment = request.args.getlist('sentiment')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    sort = request.args.get('sort', 'score desc')
    rows = int(request.args.get('rows', 10))
    start = int(request.args.get('start', 0))

    # Build filter queries
    fq = []

    if sentiment:
        sentiment_fq = ' OR '.join([f'sentiment_label:"{s}"' for s in sentiment])
        fq.append(f'({sentiment_fq})')

    if date_from or date_to:
        date_from_val = date_from if date_from else '*'
        date_to_val = date_to if date_to else '*'
        fq.append(f'created_at:["{date_from_val}" TO "{date_to_val}"]')

    # Build sort
    sort_map = {
        'relevance': 'score desc',
        'likes': 'likes desc',
        'reposts': 'reposts desc',
        'date_desc': 'created_at desc',
        'date_asc': 'created_at asc',
    }
    sort_param = sort_map.get(sort, 'score desc')

    # Build Solr params
    params = {
        'q': f'cleaned_text:"{query}" OR text:"{query}"' if query != '*' else '*:*',
        'rows': rows,
        'start': start,
        'wt': 'json',
        'hl': 'true',
        'hl.fl': 'text',
        'hl.simple.pre': '<mark>',
        'hl.simple.post': '</mark>',
        'sort': sort_param,
        'facet': 'true',
        'facet.field': 'sentiment_label',
    }

    if fq:
        params['fq'] = fq

    start_time = time.time()
    response = requests.get(f"{SOLR_URL}/select", params=params)
    elapsed = round((time.time() - start_time) * 1000, 2)

    data = response.json()
    docs = data['response']['docs']
    num_found = data['response']['numFound']
    highlighting = data.get('highlighting', {})

    # Merge highlights into docs
    for doc in docs:
        doc_id = doc.get('id', '')
        if doc_id in highlighting and 'text' in highlighting[doc_id]:
            doc['highlighted_text'] = highlighting[doc_id]['text'][0]
        else:
            doc['highlighted_text'] = doc.get('text', '')[:200] + '...'

    # Facets for pie chart
    facet_counts = data.get('facet_counts', {}).get('facet_fields', {}).get('sentiment_label', [])
    facets = {}
    for i in range(0, len(facet_counts), 2):
        if facet_counts[i+1] > 0:
            facets[facet_counts[i]] = facet_counts[i+1]

    return jsonify({
        'docs': docs,
        'numFound': num_found,
        'elapsed': elapsed,
        'facets': facets,
        'start': start,
        'rows': rows,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
