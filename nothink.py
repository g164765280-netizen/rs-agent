import litellm
opts = [{'enable_thinking': False}, {'chat_template_kwargs': {'enable_thinking': False}}]
for eb in opts:
    try:
        r = litellm.completion(model='openai/qwen3.8-flash', api_key='<YOUR_API_KEY>', api_base='https://tokenrhythm.studio/v1', messages=[{'role':'user','content':'一句话：F1和IoU区别'}], max_tokens=200, temperature=0.0, extra_body=eb)
        m = r.choices[0].message
        c = getattr(m, 'content', None)
        rc = getattr(m, 'reasoning_content', '') or ''
        print('opt=', eb)
        print('  content_len=', len(c) if c else 0, 'reasoning_len=', len(rc))
        print('  content=', repr(c)[:100])
    except Exception as e:
        print('opt=', eb, 'ERROR', str(e)[:120])
