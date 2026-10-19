import asyncio, os, glob, json
os.environ['OPENAI_API_KEY'] = '<YOUR_API_KEY>'
os.environ['OPENAI_API_BASE'] = 'https://tokenrhythm.studio/v1'
from paperqa import Docs, Settings
SETTINGS = Settings(llm='openai/qwen3.8-flash', embedding='st-/media/sdb1/gzj/weights/llm/rscd/bge-m3')
SETTINGS.parsing.use_doc_details = False

async def main():
    docs = Docs()
    fts = sorted(glob.glob('/media/sdb1/gzj/data/rscd/corpus/fulltext/*.json'))[:3]
    for f in fts:
        d = json.load(open(f))
        title = d.get('title', 'unknown')
        pdf = f.replace('/fulltext/', '/pdf_fulltext/').replace('.json', '.pdf')
        print('adding', title[:40], flush=True)
        try:
            await docs.aadd(pdf, settings=SETTINGS, citation=title, title=title)
            print('  ok', flush=True)
        except Exception as e:
            print('  ERROR:', str(e)[:120], flush=True)
    q = '变化检测常用评价指标有哪些？'
    print('=== 提问 ===', flush=True)
    s = await docs.aquery(q, settings=SETTINGS)
    a = getattr(s, 'answer', None)
    print(str(a)[:500] if a else 'answer 为空', flush=True)

asyncio.run(main())
