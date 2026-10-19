import json, litellm
from eval_set_build_v8 import PROMPT, API_KEY, API_BASE, MODEL
d=json.load(open('/media/sdb1/gzj/data/rscd/corpus/fulltext/1810.08462v1.json'))
text='\n'.join(d.get('pages_text',[]))
r=litellm.completion(model=MODEL,api_key=API_KEY,api_base=API_BASE,messages=[{'role':'system','content':'你是抽取器，只会照抄原文。'},{'role':'user','content':PROMPT+'\n\n论文文本：\n'+text[:10000]}],temperature=0.0,max_tokens=2000,extra_body={'enable_thinking':False})
print(repr(r.choices[0].message.content)[:1500])
