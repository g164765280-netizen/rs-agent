#!/bin/bash
cd /media/sdb1/gzj/support/llm/rscd
export OPENAI_API_KEY=<YOUR_API_KEY>
export OPENAI_API_BASE=https://tokenrhythm.studio/v1
export ANSWERS_FILE=/media/sdb1/gzj/support/llm/rscd/concept_answers_v3.jsonl
export OUT_FILE=/media/sdb1/gzj/support/llm/rscd/concept_eval_report_v3.jsonl
exec /media/sdb1/gzj/support/llm/envs/paperqa/bin/python eval_concept_judge.py > judge_v3.log 2>&1
