"""Verify local downloaded weights, index parts and corpus against upstream SHA256."""
import hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
manifest=json.loads(Path('/home/jk/work/datasets/assets-manifest.json').read_text())
dirs={'Qwen/Qwen2.5-3B':Path('/home/jk/work/models/Qwen2.5-3B'),'intfloat/e5-base-v2':Path('/home/jk/work/models/e5-base-v2'),'PeterJinGo/wiki-18-e5-index':Path('/home/jk/work/datasets/nq_search'),'PeterJinGo/wiki-18-corpus':Path('/home/jk/work/datasets/nq_search')}
def verify(job):
 repo,f=job;path=dirs[repo]/f['rfilename'];assert path.exists(),path
 assert path.stat().st_size==f['size'],path
 h=hashlib.sha256()
 with path.open('rb') as stream:
  while block:=stream.read(16*1024*1024):h.update(block)
 assert h.hexdigest()==f['lfs']['sha256'],f'Hash mismatch: {path}'
 print('VERIFIED',path,h.hexdigest(),flush=True)
jobs=[(repo,f) for repo,info in manifest.items() for f in info['files'] if 'lfs' in f and (dirs[repo]/f['rfilename']).exists()]
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(verify,jobs))
