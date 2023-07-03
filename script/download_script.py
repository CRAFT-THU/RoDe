import ssgetpy
import sys

download_path = "./" + sys.argv[1]
results = ssgetpy.search(rowbounds=(10000,None),colbounds=(10000,None),nzbounds=(100000,None),limit=5000)
results.download(destpath=download_path)