import subprocess
import os 
import matplotlib.pyplot as plt

PROG_PATH = os.path.dirname(os.path.abspath(__file__))

def profile_data(run_path):
    # run_path  = "./cuda/main2d"
    result = subprocess.run([run_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    # print(output)
    time = output.split(':')[-1].strip().split('ms')[0]
    return time

def plot_func(cpu_time_list, gpu_time_list):
    plt.plot(cpu_time_list)
    plt.plot(gpu_time_list)
    plt.xlabel('Iteration')
    plt.ylabel('Time(ms)')
    plt.title('CPU and GPU Time(ms) vs Iteration')
    plt.legend(['CPU', 'GPU'])
    plt.savefig('cpu_gpu_time.png')
    return

def main():
    profiling = os.path.join(PROG_PATH, 'profiling')
    if not os.path.exists(profiling):
        os.mkdir(profiling)
    
    cpu_time_list, gpu_time_list = [], []

    for i in range(1, 11):
        profiling_cpu = os.path.join(PROG_PATH, 'main')
        time_cpu = profile_data(profiling_cpu)
        print('CPU time: {}ms'.format(time_cpu))
        cpu_time_list.append(time_cpu)
        
    os.chdir(os.path.join(PROG_PATH, 'cuda'))
    for i in range(1, 11):
        profiling_cuda = './main2d'
        time_gpu = profile_data(profiling_cuda)
        print('GPU time: {}ms'.format(time_gpu))
        gpu_time_list.append(time_gpu)

    os.chdir(PROG_PATH)
    out_file = open(os.path.join(profiling, 'time_data.csv'), 'w')
    out_file.write("CPU(ms),GPU(ms)\n")
    for i in range(10):
        out_file.write("{},{}\n".format(cpu_time_list[i], gpu_time_list[i]))
    out_file.close()
    plot_func(cpu_time_list, gpu_time_list)

    return


main()