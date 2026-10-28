"""Assemble the official index and extract Wikipedia without deleting source assets."""
import gzip,shutil
from pathlib import Path
root=Path('/home/jk/work/datasets/nq_search')
index=root/'e5_Flat.index'
parts=[root/'part_aa',root/'part_ab']
expected=42949672960+21609402413
if not index.exists():
 assert all(p.exists() for p in parts), 'Download index parts first'
 assert sum(p.stat().st_size for p in parts)==expected
 tmp=root/'e5_Flat.index.assembling'
 with tmp.open('wb') as out:
  for p in parts:
   with p.open('rb') as source:shutil.copyfileobj(source,out,16*1024*1024)
 assert tmp.stat().st_size==expected
 tmp.replace(index)
 print('Index assembled',flush=True)
corpus=root/'wiki-18.jsonl'
if not corpus.exists():
 tmp=root/'wiki-18.jsonl.extracting'
 with gzip.open(root/'wiki-18.jsonl.gz','rb') as source,tmp.open('wb') as out:
  shutil.copyfileobj(source,out,16*1024*1024)
 tmp.replace(corpus)
 print('Corpus extracted',flush=True)
