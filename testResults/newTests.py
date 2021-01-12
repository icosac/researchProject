import matplotlib
matplotlib.use("Agg")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log
import argparse, sys

class Run:
    def __init__(self,name, test_name, discr, time, length, err, refinements, threads, functionType, jump, guessInitialAngles, power_file=""):
        self.name=name
        self.test_name=test_name
        self.discr=discr
        self.time=time
        self.length=length
        self.err=err
        self.refinements=refinements
        self.threads=threads
        self.functionType=functionType
        self.jump=jump
        self.power_file=power_file
        self.power_cons=[]
        self.n=1
        if guessInitialAngles=="true":
            self.guessInitialAngles=True
        else:
            self.guessInitialAngles=False

    def __eq__(self, other):
        return (self.test_name==other.test_name and 
                self.discr==other.discr and 
                self.refinements==other.refinements and 
                self.threads==other.threads and 
                self.guessInitialAngles==other.guessInitialAngles and 
                self.functionType==other.functionType and 
                self.jump==other.jump)

    def __str__(self):
        return (self.name+" "+self.test_name+" "+str(self.discr)+" "+str(self.time)+" "+self.power_file)

import re
def scan(r): #Consider the value taken each 50ms
    power=[]
    mean_power=0.0
    frequencies=[]
    t=0
    file_name=r.power_file
    with open(file_name) as file:
        for l in file:
            #https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3231/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/AppendixTegraStats.html
            if re.search("GR3D_FREQ", l)!=None:
                #Find values for tegrastats
                try: #Find values for power consumption
                    p_cpu=re.search("CPU (\d+)/(\d+)", l).groups()
                    p_gpu=re.search("GPU (\d+)/(\d+)", l).groups()
                    inst_cpu=float(int(p_cpu[0])/1000.0)
                    inst_gpu=float(int(p_gpu[0])/1000.0)
                    mean_power+=float(int(p_cpu[1])/1000.0)
                    mean_power+=float(int(p_gpu[1])/1000.0)
                    power.append({"W" : (inst_cpu+inst_gpu), "time" : t})
                except:
                    power.append({"W" : 0, "time" : t})    
                try: #Find values for frequencies
                    frequencies.append({"MHz" : int(re.search("GR3D_FREQ (\d+)%", l).groups()[0]), "time" : t})
                except: 
                    frequencies.append({"MHz" : int(re.search("GR3D_FREQ (\d+)%", l).groups()[0]), "time" : t})
            else:
                #Find values for NUC
                try:
                    # app=re.search("Power: (.+)", l).groups()[0].split(', ')
                    app=[float(w.split(': ')[1]) for w in re.search("Power: (.+), ", l).groups()[0].split(', ')]
                    p=0.0
                    for a in app:
                        p+=a
                    power.append({"W" : p, "time" : t})
                except Exception as e:
                    print(e)
                    power.append({"W" : 0, "time" : t})
            t+=50

    return (power, frequencies)


def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="tests.json", help="Input file. Default is tests.json")
    parser.add_argument("-n", "--names", type=str, default="", help="List of devices to print separated by comas without spaces.")
    parser.add_argument("-N", "--test_names", type=str, default="", help="List of the names of the tests to print separated by comas without spaces.")
    parser.add_argument("-r", "--ref", type=str, default="", help="List of refinements to consider. The list is coma-separated without spaces.")
    parser.add_argument("-f", "--func", type=int, default=2, help="The function to consider, default is 2.")
    parser.add_argument("-j", "--jump", type=int, default=2, help="Considered only if --func is 2. It represents the value for the parameter jump. Default is 2.")
    parser.add_argument("-g", "--guess", type=int, default=1, help="Considered only if --func is not 0. It says whether to look for specific angles (1) or not (0). Default is 1.")
    parser.add_argument("-l", "--log", action="store_true", default=False, help="Whether to apply logarithm to data or not.")
    parser.add_argument("-M", "--max", type=int, default=sys.maxsize, help="Only times below this value are kept into consideration. Default is sys.maxsize.")
    parser.add_argument("-d", "--min_discr", type=int, default=0, help="The minimum number of discretizations to consider. Default is 0.")
    parser.add_argument("-D", "--max_discr", type=int, default=sys.maxsize, help="The maximum number of discretizations to consider. Default is sys.maxsize.")
    parser.add_argument("-t", "--min_thread", type=int, default=128, help="The minimum number of threads to consider. Default is 128.")
    parser.add_argument("-T", "--max_thread", type=int, default=1024, help="The maximum number of discretizations to consider. Default is 1024.")
    options=parser.parse_args(args)
    return options

def main():
    arguments=getOptions(sys.argv[1:])
    input_file="tests.json"
    acceptAllDevices=False
    acceptAllTests=False
    acceptAllRef=False
    names=[]
    testNames=[]
    defRefinements=[]
    log_b=arguments.log
    timeLimit=sys.maxsize
    defDiscr=[0, sys.maxsize]
    defThreads=[128, 1024]
    defFunc=2
    defJump=0
    defGuessed=False

    if arguments.input!="" and arguments.input.endswith(".json"):
        input_file=arguments.input
    try:
        names=arguments.names.split(',')
    except:
        pass
    if names==[] or names[0]=="":
        acceptAllDevices=True
    try:
        testNames=arguments.test_names.split(',')
    except:
        pass
    if testNames==[] or testNames[0]=="":
        acceptAllTests=True
    try:
        defRefinements.append(int(ref) in arguments.refinements.split(','))
    except:
        pass
    if defRefinements==[] or defRefinements[0]=="":
        acceptAllRef=True
    try:
        timeLimit=int(arguments.max)
    except:
        pass
    try:
        defDiscr[0]=int(arguments.min_discr)
        defDiscr[1]=int(arguments.max_discr)
    except:
        pass
    try:
        defThreads[0]=int(arguments.min_thread)
        defThreads[1]=int(arguments.max_thread)
    except:
        pass
    try:
        defFunc=int(arguments.func)
    except:
        pass
    try:
        if defFunc==2:
            defJump=int(arguments.jump)
    except:
        pass
    try:
        if defFunc!=0:
            defGuessed=bool(arguments.guess)
    except:
        pass

    #Read data from json file and add every possible thing
    runs=[]
    lines=0
    with open(input_file) as json_file:
        for l in json_file:
            lines+=1
    n_lines=0
    with open(input_file) as json_file:
        data=json.load(json_file)
        for p in data['run']:
            print("Reading file... {:.2f}%".format(n_lines/lines*100.0), end="\r")
            n_lines+=1
            r=None
            try:
                r=Run(p['name'], p['test_name'], p['discr'], p['time'], p['length'], p['err'], p['refinements'], p['threads'], p['functionType'], p['jump'], p['guessInitialAngles'], p['power_file'])
            except:
                r=Run(p['name'], p['test_name'], p['discr'], p['time'], p['length'], p['err'], p['refinements'], p['threads'], p['functionType'], p['jump'], p['guessInitialAngles'])
            if (acceptAllDevices or (not acceptAllDevices and r.name in names)) and \
                (acceptAllTests or (not acceptAllTests and r.test_name in testNames)) and \
                (acceptAllRef or (not acceptAllRef and r.refinements in defRefinements)) and \
                r.time<timeLimit and \
                r.discr>=defDiscr[0] and r.discr<=defDiscr[1] and \
                r.threads>=defThreads[0] and r.threads<=defThreads[1] and \
                r.functionType==defFunc and (r.functionType!=2 or (r.functionType==2 and r.jump==defJump)) and \
                r.guessInitialAngles==defGuessed:
            
                if r.power_file!="":
                    (p, f)=scan(r)
                    r.power_cons.append(p)
                #Run through all runs and check if there is an equal one
                found=False
                for i in range(len(runs)):
                    if runs[i]==r:
                        runs[i].time+=r.time 
                        runs[i].n+=1
                        runs[i].power_cons.append(r.power_cons[0])
                        found=True
    
                if not found:
                    runs.append(r)
    print("                                                 \rReading file... 100%")

    #Run through all runs and check which had multiple times.
    i=0
    for r in runs:
        if r.n>1:
            r.time=(r.time/r.n)
            maxL=0
            for i in range(len(r.power_cons)): #Find the longest log
                if len(r.power_cons[i])>maxL:
                    maxL=len(r.power_cons[i])
            mean_ps=[]
            for i in range(maxL):
                mean_p=0
                nArrOk=0 #Divide just by the number of arrays large at least i
                for arr in range(len(r.power_cons)):
                    if i<len(r.power_cons[arr]):
                        mean_p+=r.power_cons[arr][i]["W"]
                        nArrOk+=1
                if i==0 and len(r.power_cons)!=nArrOk:
                    print(i, len(r.power_cons), nArrOk)
                mean_p/=nArrOk
                mean_ps.append({'W' : mean_p, 'time' : i*50})
            r.power_cons=mean_ps
        i+=1

    discrs=[]
    threads=[]
    for r in runs:
        if acceptAllRef:
            if r.refinements not in defRefinements:
                defRefinements.append(r.refinements)
        if r.discr not in discrs:
            discrs.append(r.discr)
        if r.threads not in threads:
            threads.append(r.threads)
    
    import matplotlib.colors as mcolors
    colors=['blue', 'orange', 'red', mcolors.CSS4_COLORS['limegreen'], mcolors.CSS4_COLORS['darkgreen'], 'purple', 'red', 'gray']

#    for tn in range(len(testNames)):
#        width=0.1
#        for th in threads:
#            samples=times(names, str(f) for f in )
#            for n in range(len(names)):
#                labels=times(discrs, defRefinements)
#                x=np.arange(len(labels))
#                fig, ax=plt.subplots()
#                for (d, r) in labels:
#                    values=[]
#                    for r in runs:
#                        if r.name=n and r.test_name==tn and \
#                            r.discr==d and r.refinements==r and \
#                            r.threads==th and 
#
#                    ax.barh(x, values, width, label=labels[l]) #colors=colors[n]
    i=0
    for r in runs:
        #for a in r.power_cons:
        #    print(a)
        fig=plt.figure(figsize=(6.4*3, 4.8*3))
        x=[pt['time'] for pt in r.power_cons]
        y=[pt['W'] for pt in r.power_cons]
        ax=plt.step(x, y, label=r.name)
        plt.savefig(("images/prova"+r.name+".png"), transparent=True)
        if i==10:
            break
        i+=1
        

#for l in range(len(logs)):
#                p, f, t, m=scan(logs[l])
#                ax[d].step(t, p, label=dev_names[l], color=colors[devices.index(dev_names[l])])
#                ax[d].step(t, [m for time in t], label="mean W "+dev_names[l], color=colors[devices.index(dev_names[l])])
#                if t[-1]>longest_x:
#                    longest_x=t[-1]
#                if max(p)>longest_y:
#                    longest_y=max(p)
#        

#    for i in range(len(tests)):
#        labels = 
#        x = np.arange(len(labels))  # the label locations
#        width = 0.1  # the width of the bars
#
#        rects=[]
#
#        fig=plt.figure(figsize=(6.4*3, 4.8*3), constrained_layout=True)
#        spec=fig.add_gridspec((int((len(discrs)+1)/2)+1), 2)
#        ax=[]
#        newL=len(discrs)+1
#        for d in range(newL):
#            if d==0:
#                ax.append(fig.add_subplot(spec[0, :]))
#            else:
#                x_=int((d+1)/2)
#                y_=1
#                if d%2==1:
#                    y_=0
#                ax.append(fig.add_subplot(spec[x_, y_]))
#        l=len(devices)
#        for j in range(l):
#            values=matrix[i][j]
#            rect=ax[0].bar(x+(j-(l-1)/2)*width, values, width, label=devices[j], color=colors[j])
#            rects.append(rect)
#
#        ax[0].set_ylabel('Times in ms')
#        ax[0].set_xlabel('#Discretizations')
#        ax[0].set_title(tests[i])
#        ax[0].set_xticks(x)
#        ax[0].set_xticklabels(labels)
#        ax[0].legend()
#
#        longest_x=0.0
#        longest_y=0.0
#        for d in range(1, newL):
#            ax[d].set_ylabel("Power (W)")
#            ax[d].set_xlabel("Time in ms")
#            ax[d].set_title(str(discrs[d-1])+" discretizations")
#
#            logs=[]
#            dev_names=[]
#            for r in runs:
#                if r.power_file!="" and (r.discr==discrs[d-1] and r.test_name==tests[i] and r.name in names):
#                    logs.append(r.power_file)
#                    dev_names.append(r.name)
#            names_index=[sorted(names).index(dev_n) for dev_n in dev_names]
#            for l in range(len(logs)):
#                p, f, t, m=scan(logs[l])
#                ax[d].step(t, p, label=dev_names[l], color=colors[devices.index(dev_names[l])])
#                ax[d].step(t, [m for time in t], label="mean W "+dev_names[l], color=colors[devices.index(dev_names[l])])
#                if t[-1]>longest_x:
#                    longest_x=t[-1]
#                if max(p)>longest_y:
#                    longest_y=max(p)
#        for d in range(1, newL):
#            ax[d].set_xticks(list(np.arange(0, longest_x, int(longest_x/10)))+[longest_x])
#            ax[d].set_yticks(list(np.arange(0, longest_y, longest_y/5))+[longest_y])
#            ax[d].legend()
#        # plt.show()
#        plt.savefig("images/"+(str(i)+"_b.png"), transparent=True)
#
#
def times(arr1, arr2):
    ret=[]
    for a in arr1:
        for b in arr2:
            ret.append((a,b))
    return ret

if __name__=="__main__":
    main()


