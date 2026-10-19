import litellm, json
r = litellm.completion(
    model='openai/deepseek-v4-pro-0813',
    api_key='<YOUR_API_KEY>',
    api_base='https://tokenrhythm.studio/v1',
    messages=[{'role':'user','content':'变化检测和图像分割的区别？一句话'}],
    max_tokens=300, temperature=0.0,
)
m = r.choices[0].message
print('content:', repr(m.content)[:200])
print('reasoning_content:', repr(getattr(m, 'reasoning_content', None))[:200])
print('finish_reason:', r.choices[0].finish_reason)
