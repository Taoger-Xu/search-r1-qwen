"""Download pinned assets needed for the default Qwen2.5-3B/NQ experiment."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

ASSETS = [
 ('Qwen/Qwen2.5-3B','model','3aab1f1954e9cc14eb9509a215f9e5ca08227a9b','/home/jk/work/models/Qwen2.5-3B', ['*.json','*.txt','*.safetensors','LICENSE','README.md']),
 ('intfloat/e5-base-v2','model','f52bf8ec8c7124536f0efb74aca902b2995e5bcd','/home/jk/work/models/e5-base-v2', ['config.json','model.safetensors','tokenizer*','special_tokens_map.json','vocab.txt','README.md']),
 ('RUC-NLPIR/FlashRAG_datasets','dataset','bcafb8dd07d453be3cbeeeb3f78be1841bddf92c','/home/jk/work/datasets/FlashRAG_datasets', ['nq/*','README.md']),
 ('PeterJinGo/wiki-18-e5-index','dataset','a4d31160a035f30764604f4827cd8f1d0315eb86','/home/jk/work/datasets/nq_search', ['part_aa','part_ab']),
 ('PeterJinGo/wiki-18-corpus','dataset','69c1c00ffe7c5554c68d8548355cb22e46aabc51','/home/jk/work/datasets/nq_search', ['wiki-18.jsonl.gz']),
]
def download(asset):
 repo, kind, revision, directory, patterns = asset
 print('START',repo,flush=True)
 snapshot_download(repo, repo_type=kind, revision=revision, local_dir=directory, allow_patterns=patterns,max_workers=4)
 print('DONE',repo,flush=True)
with ThreadPoolExecutor(max_workers=5) as pool:
 list(pool.map(download,ASSETS))
Path('/home/jk/work/datasets/assets-manifest.json').write_text(json.dumps(ASSETS,indent=2))
