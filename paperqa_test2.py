import asyncio, os, json, glob
os.environ['OPENAI_API_KEY'] = '<YOUR_API_KEY>'
os.environ['OPENAI_API_BASE'] = 'https://tokenrhythm.studio/v1'
from paperqa import Docs, Settings
SETTINGS = Settings(llm='openai/qwen3.8-flash', embedding='st-/media/sdb1/gzj/weights/llm/rscd/bge-m3')

async def main():
    docs = Docs()
    # 从 fulltext JSON 拿标题，作为 citation 传入跳过 S2
    ft = sorted(glob.glob('/media/sdb1/gzj/data/rscd/corpus/fulltext/*.json'))[:3]
    for f in ft:
        d = json.load(open(f))
        title = d.get('title', 'unknown')
        pdf = f.replace('/fulltext/', '/pdf_fulltext/').replace('.json', '.pdf')
        print('adding', title[:50], flush=True)
        try:
            await docs.aadd(pdf, settings=SETTINGS, citation=title, title=title)
        except Exception as e:
            print('  aadd ERROR:', str(e)[:150], flush=True)
    q = '变化检测常用评价指标有哪些？'
    print('=== 提问 ===', flush=True)
    try:
        s = await docs.aquery(q, settings=SETTINGS)
        a = getattr(s, 'answer', None)
        print(str(a)[:500] if a else 'answer 为空', flush=True)
    except Exception as e:
        print('aquery ERROR:', str(e)[:200], flush=True)

asyncio.run(main())
