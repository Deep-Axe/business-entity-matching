import json, sqlite3
con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
print(con.execute('select * from records limit 2').fetchall())
rows = [json.loads(l) for l in open('artifacts/improvement_v2/research_next/dev100k_error_analysis_v1/error_pairs.jsonl')]
fp = [r for r in rows if r['kind'] == 'false_positive']
unowned = 0
for r in fp:
    o = con.execute('select reference_id from labels where target_id=?', (r['target_id'],)).fetchone()
    rec = con.execute('select entity_id,name,address from records where entity_id=?', (o[0],)).fetchone() if o else None
    unowned += o is None
    a, b = r['reference'], r['target']
    print(f"{r['probability']:.2f} {a['business_name']} | {a['business_address']}\n    T: {b['business_name']} | {b['business_address']}\n    OWNER: {o and o[0]} {rec}")
print('unowned', unowned, 'of', len(fp))
