import json, litellm
API_KEY='<YOUR_API_KEY>'; API_BASE='https://tokenrhythm.studio/v1'
d=json.load(open('/media/sdb1/gzj/data/rscd/corpus/fulltext/2103.00208v3.json'))
text='\n'.join(d.get('pages_text',[]))
# 喂全文（结果在 28000-40000 字符处）
prompt='''下面是一篇论文的全文。请从中找出论文提出的方法（BIT）在各数据集上的 F1 和 IoU 数值。
每个数值给出：方法名、数据集、指标(F1/IoU)、数值、包含该数值的原文原句。
只输出 JSON：{results:[{method:..,dataset:..,metric:..,value:..,quote:..}]}
value 必须原样出现在 quote 里。'''
r=litellm.completion(model='openai/kimi-k2.6',api_key=API_KEY,api_base=API_BASE,messages=[{'role':'user','content':prompt+'\n\n论文全文：\n'+text}],temperature=0.0,max_tokens=2500,extra_body={'enable_thinking':False})
print(r.choices[0].message.content[:2000])
