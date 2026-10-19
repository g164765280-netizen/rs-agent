import json, litellm
API_KEY='<YOUR_API_KEY>'; API_BASE='https://tokenrhythm.studio/v1'
d=json.load(open('/media/sdb1/gzj/data/rscd/corpus/fulltext/2103.00208v3.json'))
text='\n'.join(d.get('pages_text',[]))
prompt='''下面是一篇论文的文本。请从中找出论文提出的方法（如 BIT）在各数据集上的 F1 和 IoU 数值。
对每个数值，给出：方法名、数据集、指标(F1/IoU)、数值、以及包含该数值的原文原句。
只输出 JSON：{results:[{method:..,dataset:..,metric:..,value:..,quote:..}]}
注意：value 必须原样出现在 quote 里。'''
for m in ['openai/kimi-k2.6','openai/glm-5.3']:
    try:
        r=litellm.completion(model=m,api_key=API_KEY,api_base=API_BASE,messages=[{'role':'user','content':prompt+'\n\n论文文本：\n'+text[:8000]}],temperature=0.0,max_tokens=2000,extra_body={'enable_thinking':False})
        print('=== 模型:',m,'===')
        print(repr(r.choices[0].message.content)[:1500])
    except Exception as e:
        print(m,'错误:',str(e)[:120])
