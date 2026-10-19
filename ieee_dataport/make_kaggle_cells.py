#!/usr/bin/env python3
"""make_kaggle_cells.py -- build the one-line notebook cells for Kaggle: each cell is
    import base64,zlib;exec(zlib.decompress(base64.b64decode("...")).decode())
and unpacks the scripts into /kaggle/working/code before running the orchestrator."""
import base64, zlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

def payload(entry, files):
    blobs = {f: base64.b64encode(zlib.compress(open(os.path.join(HERE, f), "rb").read(), 9)).decode() for f in files}
    src = "import base64,zlib,os,sys,subprocess\n"
    src += "W=os.environ.get('KAGGLE_WORKING','/kaggle/working');D=os.path.join(W,'code');os.makedirs(D,exist_ok=True)\n"
    src += "F=" + repr(blobs) + "\n"
    src += "for n,b in F.items():\n    open(os.path.join(D,n),'wb').write(zlib.decompress(base64.b64decode(b)))\n"
    src += "print('scripts unpacked:',sorted(F))\n"
    if isinstance(entry, dict):     # entry chosen by the MODE variable set in the first line of the notebook cell
        src += f"E={entry!r}.get(os.environ.get('MODE','train'))\n"
        src += "print('mode:',os.environ.get('MODE','train'),'seeds:',os.environ.get('SEEDS','0,1,2'))\n"
        src += "subprocess.run([sys.executable,os.path.join(D,E)],check=True)\n"
    else:
        src += f"subprocess.run([sys.executable,os.path.join(D,{entry!r})],check=True)\n"
    outer = base64.b64encode(zlib.compress(src.encode(), 9)).decode()
    return 'import base64,zlib;exec(zlib.decompress(base64.b64decode("' + outer + '")).decode())'

cells = {
    "prepare": payload("kaggle_prepare.py", ["ieee_audit_tools.py", "train_clip_classifier.py", "kaggle_prepare.py"]),
    "train": payload("kaggle_train.py", ["train_clip_classifier.py", "kaggle_train.py"]),
    "all": payload("kaggle_all.py", ["ieee_audit_tools.py", "train_clip_classifier.py", "kaggle_prepare.py", "kaggle_train.py", "kaggle_all.py"]),
}
for k, v in cells.items():
    open(os.path.join(HERE, f"kaggle_cell_{k}.txt"), "w").write(v)
    print(k, len(v), "chars")
