from memory_profiler import profile

import inspect
from gpu_mem_track import MemTracker
frame = inspect.currentframe()
gpu_tracker = MemTracker(frame)

from sentence_transformers.cross_encoder import CrossEncoder

model_save_path = ''

model = CrossEncoder(model_save_path, device="cpu", max_length=64)

gpu_tracker.track()

@profile
def do():
    with open("test.txt","r")as f, open("result.txt","w")as fw:
        for i, line in enumerate(f):
            if i%1000 == 0:print(i)
            line_lst = line.strip().split("\t")
            score = model.predict([line_lst])[0]
            fw.write(line.strip()+"\t"+str(score)+"\n")

do()
gpu_tracker.track()
