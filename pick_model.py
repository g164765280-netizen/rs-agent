import litellm
models = ['openai/qwen3.8-flash', 'openai/glm-5', 'openai/kimi-k2.5', 'openai/minimax-m2.5']
for m in models:
    try:
        r = litellm.completion(model=m, api_key='<YOUR_API_KEY>', api_base='https://tokenrhythm.studio/v1', messages=[{'role':'user','content':'一句话回答：F1 和 IoU 的区别？'}], max_tokens=200, temperature=0.0)
        msg = r.choices[0].message
        c = getattr(msg, 'content', None); rc = getattr(msg, 'reasoning_content', None)
        print(f'{m}: content={repr(c)[:80]}  reasoning_len={len(rc) if rc else 0}')
    except Exception as e:
        print(f'{m}: ERROR {str(e)[:80]}')
