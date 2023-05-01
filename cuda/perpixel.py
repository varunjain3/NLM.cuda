import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

shared = np.array([2260.5,2186.5,2206.5,4576.0,15009.5,55318.5,218099.0])
python = np.array([578.5,2478.5,9348.5,35914.0,148386.0])
cpp = np.array([1587.0,6169.0,25850.0,104115.5])
workload = np.array([32,64,128,256,512,1024,2048])

plt.figure(figsize=(10,6))
plt.plot(workload,shared,label="cuda")
plt.plot(workload[:len(python)],python,label="python-baseline")
plt.plot(workload[:len(cpp)],cpp,label="cpp-baseline")
plt.legend()
plt.ylabel("time taken (ms)")
plt.yscale("log",base=10)
plt.xlabel("image size")
plt.xscale('log', base=2)
plt.title("time take for various workloads")
plt.xticks(workload)
plt.savefig("workloads.png")


pixels = workload**2
plt.figure(figsize=(10,6))
plt.plot(workload,shared/pixels,label="cuda")
plt.plot(workload[:len(python)],python/pixels[:len(python)],label="python-baseline")
plt.plot(workload[:len(cpp)],cpp/pixels[:len(cpp)],label="cpp-baseline")
plt.title("time take per pixel across various workloads")
plt.ylabel("time taken per pixel (ms)")
plt.xlabel("image size")
plt.xscale('log', base=2)
plt.xticks(workload)
plt.legend()
plt.savefig("perpixel.png")
