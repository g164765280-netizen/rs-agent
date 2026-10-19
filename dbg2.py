import json, re, glob
import litellm
from eval_full_v4 import extract_table
API_KEY='<YOUR_API_KEY>'; API_BASE='https://tokenrhythm.studio/v1'; MODEL='openai/qwen3.8-flash'
# 建表
title2id={}; table={}
for ft in sorted(glob.glob('/media/sdb1/gzj/data/rscd/corpus/fulltext/*.json')):
    d=json.load(open(ft)); title2id[d.get('title','').strip()]=d['arxiv_id']
    table[d['arxiv_id']]=extract_table('\n'.join(d.get('pages_text',[])))
qs=[json.loads(l) for l in open('/media/sdb1/gzj/data/rscd/corpus/eval_set_v3.jsonl') if l.strip()]
for q in qs[:5]:
    g=q['gold']; pid=title2id.get(q['title'])
    hits=[e for e in table.get(pid,[]) if e['metric']==g['metric'] and e['dataset']==g['dataset']]
    ev='\n'.join(f'[{i+1}] {e["sentence"]}' for i,e in enumerate(hits[:3]))
    r=litellm.completion(model=MODEL,api_key=API_KEY,api_base=API_BASE,messages=[{'role':'system','content':'严格基于证据，数字照抄。'},{'role':'user','content':f'问题：{q["question"]}\n证据：\n{ev}\n请回答：'}],temperature=0.0,max_tokens=150,extra_body={'enable_thinking':False})
    ans=r.choices[0].message.content
    print(f'问题: {q["question"]}')
    print(f'  gold值: {g["value"]}  命中候选数: {len(hits)}')
    print(f'  候选值: {[e["value"] for e in hits[:5]]}')
    print(f'  模型答: {ans[:150]}')
    print()
