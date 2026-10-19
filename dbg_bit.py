import json, litellm
API_KEY='<YOUR_API_KEY>'; API_BASE='https://tokenrhythm.studio/v1'
d=json.load(open('/media/sdb1/gzj/data/rscd/corpus/fulltext/2103.00208v3.json'))
text='\n'.join(d.get('pages_text',[]))
prompt='''下面是一篇论文(BIT变化检测)的全文。论文里有实验结果表，格式是竖排：方法名单独一行，下一行是一串用 / 分隔的数值。
请你理解表头与列，抽出每个方法在各数据集上的 F1 和 IoU 数值。
逐条输出 JSON：{results:[{method:方法名,dataset:数据集,metric:f1或iou,value:数值,quote:包含该数值的原文那一行}]}
要求：value 必须原样出现在 quote 里；方法名从原文里找（如 Base、BIT、SNUNet 等）；数据集从表头或上下文判断（LEVIR-CD/WHU-CD/DSIFN-CD）。'''
r=litellm.completion(model='openai/kimi-k2.6',api_key=API_KEY,api_base=API_BASE,messages=[{'role':'user','content':prompt+'\n\n论文全文：\n'+text}],temperature=0.0,max_tokens=3000,extra_body={'enable_thinking':False})
print(r.choices[0].message.content[:2500])
