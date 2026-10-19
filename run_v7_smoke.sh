#!/bin/bash
cd /media/sdb1/gzj/support/llm/rscd
exec /media/sdb1/gzj/support/llm/envs/paperqa/bin/python paperqa_concept_v7_30b.py 3 > v7_smoke.log 2>&1
