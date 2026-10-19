#!/bin/bash
cd /media/sdb1/gzj/support/llm/rscd
exec /media/sdb1/gzj/support/llm/envs/paperqa/bin/python paperqa_concept_v5_multi.py 3 > paperqa_v5_multi.log 2>&1
