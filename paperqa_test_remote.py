import asyncio, os
from pathlib import Path
os.environ['OPENAI_API_KEY'] = '<YOUR_API_KEY>'
os.environ['OPENAI_API_BASE'] = 'https://tokenrhythm.studio/v1'
from paperqa import Docs, Settings

SETTINGS = Settings(
    llm='openai/qwen3.8-flash',
    embedding='st-/media/sdb1/gzj/weights/llm/rscd/bge-m3',
)
PAPERS_DIR = Path('/media/sdb1/gzj/data/rscd/corpus/pdf_fulltext')

async def main():
    docs = Docs()
    papers = sorted(PAPERS_DIR.glob('*.pdf'))
    print(f'共 {len(papers)} 篇，先加前 5 篇测试', flush=True)
    for p in papers[:5]:
        print('  adding', p.name, flush=True)
        await docs.aadd(str(p), settings=SETTINGS)
    q = '变化检测中常用的评价指标有哪些？'
    print(f'=== 提问：{q} ===', flush=True)
    session = await docs.aquery(q, settings=SETTINGS)
    ans = getattr(session, 'answer', None)
    print('=== 回答 ===', flush=True)
    print(str(ans)[:600] if ans else 'answer 为空', flush=True)

asyncio.run(main())
