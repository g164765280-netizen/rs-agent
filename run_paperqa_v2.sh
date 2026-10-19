#!/bin/bash
cd /media/sdb1/gzj/support/llm/rscd
exec /media/sdb1/gzj/support/llm/envs/paperqa/bin/python paperqa_concept_v2.py > paperqa_v2_full.log 2>&1
