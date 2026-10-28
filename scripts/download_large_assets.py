"""Resumable range downloader for large pinned experiment assets."""
import os,time,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
ROOT=Path('/home/jk/work')
FILES=[
('Qwen/Qwen2.5-3B','3aab1f1954e9cc14eb9509a215f9e5ca08227a9b','model-00001-of-00002.safetensors',3968658944,ROOT/'models/Qwen2.5-3B',False),
('Qwen/Qwen2.5-3B','3aab1f1954e9cc14eb9509a215f9e5ca08227a9b','model-00002-of-00002.safetensors',2203268048,ROOT/'models/Qwen2.5-3B',False),
('PeterJinGo/wiki-18-e5-index','a4d31160a035f30764604f4827cd8f1d0315eb86','part_aa',42949672960,ROOT/'datasets/nq_search',True),
('PeterJinGo/wiki-18-e5-index','a4d31160a035f30764604f4827cd8f1d0315eb86','part_ab',21609402413,ROOT/'datasets/nq_search',True),
('PeterJinGo/wiki-18-corpus','69c1c00ffe7c5554c68d8548355cb22e46aabc51','wiki-18.jsonl.gz',5123307260,ROOT/'datasets/nq_search',True)]
CHUNK=64*1024*1024

def session():
 s=requests.Session();s.trust_env=False;return s

def download_file(spec):
 repo,rev,name,size,directory,is_data=spec
 directory.mkdir(parents=True,exist_ok=True)
 target=directory/name
 if target.exists() and target.stat().st_size==size:
  print('EXISTS',name,flush=True);return
 url=f'https://hf-mirror.com/{"datasets/" if is_data else ""}{repo}/resolve/{rev}/{name}'
 s=session()
 for attempt in range(10):
  try:
   r=s.head(url,allow_redirects=True,timeout=60);r.raise_for_status();resolved=r.url;break
  except requests.RequestException:
   if attempt==9:raise
   time.sleep(2)
 chunkdir=directory/'.download-chunks'/name;chunkdir.mkdir(parents=True,exist_ok=True)
 def chunk(i):
  start=i*CHUNK;end=min(size,start+CHUNK)-1;dest=chunkdir/f'{i:05d}'
  if dest.exists() and dest.stat().st_size==end-start+1:return
  for attempt in range(20):
   try:
    partial=dest.with_suffix('.tmp')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset==end-start+1:
     partial.replace(dest);return
    resume=start+offset
    with session().get(resolved,headers={'Range':f'bytes={resume}-{end}'},stream=True,timeout=(30,90)) as response:
     response.raise_for_status()
     if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {resume}-'):
      raise ValueError(f'Unexpected range response: {response.status_code} {response.headers.get("Content-Range")}')
     with partial.open('ab') as out:
      for block in response.iter_content(1024*1024):out.write(block)
    if dest.with_suffix('.tmp').stat().st_size!=end-start+1:raise ValueError('Incomplete chunk')
    dest.with_suffix('.tmp').replace(dest);return
   except (requests.RequestException,ValueError) as exc:
    print('RETRY',name,i,attempt,str(exc)[:140],flush=True)
    if attempt==19:raise
    time.sleep(min(30,attempt+2))
 count=(size+CHUNK-1)//CHUNK
 print('START',name,'chunks',count,flush=True)
 with ThreadPoolExecutor(max_workers=int(os.environ.get("DOWNLOAD_WORKERS", "24"))) as pool:
  for done,future in enumerate(as_completed([pool.submit(chunk,i) for i in range(count)]),1):
   future.result()
   if done%10==0 or done==count:print('PROGRESS',name,done,count,flush=True)
 with target.with_suffix(target.suffix+'.assembling').open('wb') as out:
  for i in range(count):
   with (chunkdir/f'{i:05d}').open('rb') as source:
    while block:=source.read(8*1024*1024):out.write(block)
 target.with_suffix(target.suffix+'.assembling').replace(target)
 for path in chunkdir.iterdir():path.unlink()
 chunkdir.rmdir()
 print('DONE',name,size,flush=True)

with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(download_file,FILES))
