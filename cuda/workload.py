import numpy as numpy
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns


runtimes = []
workloads = [32,64,128,256,512,1024,2048]

for workload in workloads:
    print(f"workload started: {workload}")
    trials = 2
    time_taken_total = 0
    for i in range(trials):
        cmd = f"export PATH=/usr/local/cuda-11.4/bin/:$PATH  && cd ./nlmeans/cuda && ./main2d {workload}"
        execute = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        execute = execute.stdout.read().decode("utf-8")
        time_taken = int(execute.strip().split()[-1][:-2])
        time_taken_total += time_taken
        
        
    runtimes.append(time_taken_total/trials)


# plotting
sns.lineplot(x=workloads, y=runtimes)
plt.xlabel("image size")
plt.ylabel("run time (ms)")
plt.xscale('log', base=2)
plt.savefig("check.png")

with open("shared_mem.txt","w") as f:
    f.writelines(",".join(map(str,workloads)) + "\n")
    f.writelines(",".join(map(str,runtimes)))