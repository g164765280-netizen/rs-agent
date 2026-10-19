#!/bin/bash
cd /media/sdb1/gzj/support/llm/rscd
export OPENAI_API_KEY=<YOUR_API_KEY>
export OPENAI_API_BASE=https://tokenrhythm.studio/v1
exec /media/sdb1/gzj/support/llm/envs/paperqa/bin/python build_paper_metrics_full.py > pm_full.log 2>&1
