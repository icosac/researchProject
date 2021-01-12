import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    with open("../prova.log") as f:
        power=[]
        times=[]
        t=0
        for l in f:
            try: #Find values for power consumption
                p_cpu=re.search("CPU (\d+)/(\d+)", l).groups()
                p_gpu=re.search("GPU (\d+)/(\d+)", l).groups()
                inst_cpu=float(int(p_cpu[0])/1000.0)
                inst_gpu=float(int(p_gpu[0])/1000.0)
                power.append((inst_cpu+inst_gpu))
            except:
                print("error")
                power.append({"W" : 0, "time" : t}) 
            times.append(t)
            t+=50

    for i in range(len(power)):
        print(power[i], times[i])

    fig=plt.figure(figsize=(6.4*3, 4.8*3))
    ax=plt.step(times, power)
    plt.savefig(("provaLogFile.png"))

if __name__=="__main__":
	main()