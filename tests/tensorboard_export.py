import os, glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logdir = "assets/model/logs"
evs = sorted(glob.glob(os.path.join(logdir, "**", "events.*"), recursive=True))
ea = EventAccumulator(os.path.dirname(evs[-1])); ea.Reload()

def series(tag):
    xs, ys = [], []
    for e in ea.Scalars(tag):
        xs.append(e.step); ys.append(e.value)
    return np.array(xs), np.array(ys)

def describe(name, y):
    if len(y)==0: return f"{name}: no data"
    mov = np.convolve(y, np.ones(100)/100, mode="valid") if len(y)>=100 else y
    slope = (mov[-1]-mov[0])/max(1,len(mov)-1)
    return f"{name}: last={y[-1]:.4f} min={y.min():.4f} med={np.median(y):.4f} max={y.max():.4f} trend≈{slope:.4f}/step"

for tag in ["train/loss","train/epsilon","train/episode_reward"]:
    x,y = series(tag)
    print(describe(tag,y))