#!/usr/bin/env python3
"""kaggle_all.py -- one GPU session: preparation (kaggle_prepare.py) followed by the classifier runs
(kaggle_train.py), with a common session clock so that no run starts too late to finish."""
import os, sys, time, glob, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
os.environ.setdefault("SESSION_START", str(time.time()))
cache = os.path.join(WORK, "cache")
if not (os.path.exists(os.path.join(cache, "manifest_IEEE_normalized.json"))
        and os.path.exists(os.path.join(cache, "manifest_IEEE_as_distributed.json"))
        and glob.glob(os.path.join(WORK, "audit_outputs.zip"))):
    subprocess.run([sys.executable, os.path.join(HERE, "kaggle_prepare.py")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "kaggle_train.py")], check=True)
