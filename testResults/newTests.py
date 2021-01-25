#import matplotlib
#matplotlib.use("Agg")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log
import argparse, sys
import re

class Run:
    def __init__(self,name, test_name, discr, time, length, err, refinements, threads, functionType, jump, guessInitialAngles, power_file="", initTime=-1.0, endTime=-1.0):
        self.name=name
        self.devName=""
        self.test_name=test_name
        self.discr=discr
        self.time=time
        self.length=length
        self.err=err
        self.refinements=refinements
        self.threads=threads
        self.functionType=int(functionType)
        self.jump=int(jump)
        self.power_file=power_file
        self.power_cons=[]
        self.n=1
        if guessInitialAngles=="true":
            self.guessInitialAngles=True
        else:
            self.guessInitialAngles=False
        self.initTime=initTime
        self.endTime=endTime

    def __eq__(self, other):
        return (self.devName==other.devName and
                self.test_name==other.test_name and 
                self.discr==other.discr and 
                self.refinements==other.refinements and 
                self.threads==other.threads and 
                self.guessInitialAngles==other.guessInitialAngles and 
                self.functionType==other.functionType and 
                self.jump==other.jump)

    def __str__(self):
        return ("dev_name: "+str(self.devName)+
                " test_name: "+str(self.test_name)+
                " discr: "+str(self.discr)+
                " refinements: "+str(self.refinements)+
                " threads: "+str(self.threads)+
                " guessInitialAngles: "+str(self.guessInitialAngles)+
                " functionType: "+str(self.functionType)+
                " jump: "+str(self.jump))
        #return (self.name+" "+self.test_name+" "+str(self.discr)+" "+str(self.time)+" "+self.power_file)

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
    parser.add_argument("-f", "--func", type=str, default="2", help="List of functions to consider. The list is coma-separated without spaces. The default is 2.")
    parser.add_argument("-j", "--jump", type=str, default="3", help="List of jumps to consider. Taken into account only if 2 is present in the list of functions. The list is coma-separated without spaces. The default is 2.")
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
    print(arguments)
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
    defFuncs=[]
    defJumps=[]
    defGuessed=True

    if arguments.input!="" and arguments.input.endswith(".json"):
        input_file=arguments.input
    try:
        names=arguments.names.split(',')
    except:
        pass
    if names==[] or names[0]=="":
        acceptAllDevices=True
        names=[]
    try:
        testNames=arguments.test_names.split(',')
    except:
        pass
    if testNames==[] or testNames[0]=="":
        acceptAllTests=True
        testNames=[]
    try:
        defRefinements.append(int(ref) for ref in arguments.refinements.split(','))
    except:
        pass
    if defRefinements==[] or defRefinements[0]=="":
        acceptAllRef=True
        defRefinements=[]
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
        print("arguments.func: "+str(arguments.func))
        for func in arguments.func.split(','):
            defFuncs.append(int(func))
    except:
        pass
    if defFuncs==[] or defFuncs[0]=="":
        defFuncs=[2]
    try:
        if 2 in defFuncs:
            try:
                for jump in arguments.jump.split(','):
                    defJumps.append(int(jump))
            except:
                pass
            if defJumps==[] or defJumps[0]=="":
                defJumps=[2]
    except:
        pass
    try:
        if defFunc!=0:
            defGuessed=bool(arguments.guess)
    except:
        pass

    print("input_file: "+str(input_file))
    print("acceptAllDevices: "+str(acceptAllDevices))
    print("acceptAllTests: "+str(acceptAllTests))
    print("acceptAllRef: "+str(acceptAllRef))
    print("names: "+str(names))
    print("testNames: "+str(testNames))
    print("defRefinements: "+str(defRefinements))
    print("log_b: "+str(log_b))
    print("timeLimit: "+str(timeLimit))
    print("defDiscr: "+str(defDiscr))
    print("defThreads: "+str(defThreads))
    print("defFuncs: "+str(defFuncs))
    print("defJumps: "+str(defJumps))
    print("defGuessed: "+str(defGuessed))

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
            
            realName=r.name.split('_')[0]

            #Check if CPU: threads==0, functype==0, jump==0 guessInitialAngles==False
            if (acceptAllDevices or (not acceptAllDevices and (realName!="" or realName!=" "))):
                #print("acceptAllDevices")
                if (acceptAllTests or (not acceptAllTests and r.test_name in testNames)):
                    #print("acceptAllTests")
                    if (acceptAllRef or (not acceptAllRef and r.refinements in defRefinements)):
                        #print("acceptAllRef")
                        if r.time<timeLimit:
                            #print("time")
                            if  r.discr>=defDiscr[0] and r.discr<=defDiscr[1]:
                                if r.threads==0 and r.jump==0 and r.functionType==0 and r.guessInitialAngles==False:
                                    found=False
                                    r.devName=realName
                                    for i in range(len(runs)):
                                        if runs[i]==r:
                                            runs[i].time+=r.time 
                                            runs[i].n+=1
                                            found=True
                                    if not found:
                                        runs.append(r)


            if (acceptAllDevices or (not acceptAllDevices and (realName!="" or realName!=" "))):
                #print("acceptAllDevices")
                if (acceptAllTests or (not acceptAllTests and r.test_name in testNames)):
                    #print("acceptAllTests")
                    if (acceptAllRef or (not acceptAllRef and r.refinements in defRefinements)):
                        #print("acceptAllRef")
                        if r.time<timeLimit:
                            #print("time")
                            if  r.discr>=defDiscr[0] and r.discr<=defDiscr[1]:
                                #print("discr")
                                if  r.threads>=defThreads[0] and r.threads<=defThreads[1]:
                                    #print("threads")
                                    #print("functionType: "+str(r.functionType)+" "+str(type(r.functionType))+" "+str(defFuncs))
                                    if  r.functionType in defFuncs:
                                        #print("jump "+str(r.jump)+" "+str(type(r.jump))+" "+str(defJumps))
                                        if (r.functionType!=2 or (r.functionType==2 and r.jump in defJumps)):
                                            if  r.guessInitialAngles==defGuessed:
                                                #print("guessInitialAngles")
                                                #if r.power_file!="":
                                                    #(p, f)=scan(r)
                                                    #r.power_cons.append(p)
                                                #Run through all runs and check if there is an equal one
                                                found=False
                                                r.devName=realName
                                                for i in range(len(runs)):
                                                    if runs[i]==r:
                                                        runs[i].time+=r.time 
                                                        runs[i].n+=1
                                                        #runs[i].power_cons.append(r.power_cons[0])
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
    funcs=[]
    jumps=[]
    for r in runs:
        if acceptAllRef:
            if r.refinements not in defRefinements:
                defRefinements.append(r.refinements)
        if r.discr not in discrs:
            discrs.append(r.discr)
        if r.threads not in threads:
            threads.append(r.threads)
        if r.functionType not in funcs:
            funcs.append(r.functionType)
        if r.functionType==2 and r.jump not in jumps:
            jumps.append(r.jump)
        if r.devName not in names and acceptAllDevices:
            names.append(r.devName)
        if r.test_name not in testNames and acceptAllTests:
            testNames.append(r.test_name)

    
    print("discrs: "+str(discrs))
    print("threads: "+str(threads))
    print("funcs: "+str(funcs))
    print("jumps: "+str(jumps))
    print("names: "+str(names))
    print("testNames: "+str(testNames))

    import matplotlib.colors as mcolors
    colors=['blue', 'orange', 'red', mcolors.CSS4_COLORS['limegreen'], mcolors.CSS4_COLORS['darkgreen'], 'purple', 'red', 'gray']

    choice=input("Do you want to show the diagram over threads and different functions? [Y/n] ")
    if choice=="Y" or choice=="y":
        for tn in range(len(testNames)): 
            break
            width=0.4
            for th in threads:
                nGraphs=0
                fig=plt.figure(figsize=(6.4*3, 4.8*3), constrained_layout=True)
                spec=fig.add_gridspec(int(((len(funcs)+len(jumps)))/2), 2)
    #           fig, ax=plt.subplots()
                ax=[]
                valuess=[]
                for j in jumps:
                    print("j: "+str(j))
                    for fn in funcs: #Create a graph for each function
                        if fn!=2 and j!=jumps[-1]:
                            continue
                        print("fn: "+str(fn))
                        ax.append(fig.add_subplot(spec[int(nGraphs/2), nGraphs%2]))
                        for n in range(len(names)):
                            values=[]
                            labels=times(discrs, defRefinements)
                            x=np.arange(len(labels))
                            for l in range(len(labels)):
                                (dis,ref)=labels[l]
                                #print("looking for: "+names[n]+" with discr: "+str(dis)+", ref: "+str(ref))
                                before=len(values)
                                for r in runs:
                                    #print(r)
                                    if r.devName==names[n]:
                                        #print("r.devName")
                                        #print("r.test_name <"+r.test_name+"> <"+testNames[tn]+"> "+str(len(testNames)))
                                        if  r.test_name==testNames[tn]: 
                                            if r.discr==dis: 
                                                #print("r.discr")
                                                if  r.refinements==ref: 
                                                    #print(" r.refinements")
                                                    if r.threads==th: 
                                                        #print("r.threads")
                                                        if  r.functionType==fn: 
                                                            #print(" r.functionType")
                                                            if fn!=2 or (fn==2 and r.jump==j):
                                                                #print("r.jump")
                                                                values.append(r)
                                #if len(values)==before:
                                #    print("not found")
                                #else:
                                #    print("found: "+str(values[-1]))
                            if values!=[]:
                                valuess.append(values)
                                #print(values)
                                ax[nGraphs].barh(x+n*width-width/len(names), [v.time for v in values], width, color=colors[n], label=names[n]) #color=colors[n]
                                ax[nGraphs].set_xlabel('Times in ms')
                                ax[nGraphs].set_yticks(x)
                                ax[nGraphs].set_yticklabels(labels)
                                if func!=2:
                                    ax[nGraphs].set_title(testNames[tn]+", threads: "+str(th)+" func: "+str(fn)+", jump: 0")
                                else:
                                    ax[nGraphs].set_title(testNames[tn]+", threads: "+str(th)+" func: "+str(fn)+", jump: "+str(j))
                                ax[nGraphs].legend()
                        nGraphs+=1
                for vv in valuess[0:2]:
                    print("<", end="")
                    for v in vv:
                        print(v, end=", ")
                    print(">")
                #ax.legend()
                plt.show()

    choice=input("Do you want to print the tables? [Y/n] ")
    if choice=="Y" or choice=="y" or choice=="":
        lthreads=input("For which number of threads? ")
        lfunc=2
        ljump=3
        lguess=True
        print("namesss: "+str(names))
        #for dv1 in names:
        for tn in testNames:
            print("\\subsection{"+tn+"}")
            print("\\begin{center}")
            print("\\begin{tabular}{c|c|c", end="")
            for n in names:
                print("|c", end="")
            print("}")
            print("\t&&&\\multicolumn{"+str(len(names))+"}{|c}{Times(ms)}\\\\")
            print("\tDisc&Ref&Err", end="")
            for n in names:
                print("&"+n, end="")
            print("\\\\\n\\hline")
            for discr in discrs:
                for ref in defRefinements:
                    for dv in range(len(names)): 
                        for r in runs:
                            if r.devName==names[dv] and r.test_name==tn and r.discr==discr and r.refinements==ref:
                                if  str(r.threads) in lthreads and r.functionType==lfunc and \
                                    r.jump==ljump and r.guessInitialAngles==lguess:
                                    if dv==0:
                                        print("{:d}&{:d}&{:.1e}&{:.1e}".format(discr, ref, r.err, r.time), end="")
                                    elif dv==len(names)-1:
                                        print("&{:.1e}\\\\".format(r.time))
                                    else:
                                        print("&{:.1e}".format(r.time), end="")


                                elif r.threads==0 and r.functionType==0 and r.jump==0 and \
                                    r.guessInitialAngles==False:
                                    if dv==0:
                                        print("{:d}&{:d}&{:.1e}&{:.1e}".format(discr, ref, r.err, r.time), end="")
                                    elif dv==len(names)-1:
                                        print("&{:.1e}\\\\".format(r.time))
                                    else:
                                        print("&{:.1e}".format(r.time), end="")
                print("\\hline")
            print("\\end{tabular}")
            print("\\end{center}\n\n\n\n\n\n\n")
    

    print("discrs: "+str(discrs))
    print("threads: "+str(threads))
    print("funcs: "+str(funcs))
    print("jumps: "+str(jumps))
    print("names: "+str(names))
    print("testNames: "+str(testNames))
    choice=input("Do you want to show the power-consumption graphs? [Y/n]")
    if choice=="Y" or choice=="y" or choice=="":
        print(names)
        lNamesC=input("These are the devices in list, do you want to print all of them? Otherwise write a coma-separated list [Y/list]")
        lNames=[]
        if lNamesC=="Y":
            lNames=names
        else:
            lNames=lNamesC.split(",")
        print(discrs)
        lNamesC=input("These are the discriminations in list, do you want to print all of them? Otherwise write a coma-separated list [Y/list]")


        fig=plt.figure(figsize=(6.4*3, 4.8*3), constrained_layout=True)
        spec=fig.add_gridspec(2, 1)
        ax=[]
                

#    import pandas
#    df = pandas.DataFrame(dict(graph=['Item one', 'Item two', 'Item three'],
#                               n=[3, 5, 2], m=[6, 1, 3])) 
#
#    ind = np.arange(len(df))
#    width = 0.4
#
#    fig, ax = plt.subplots()
#    ax.barh(ind, df.n, width, color='red', label='N')
#    ax.barh(ind + width, df.m, width, color='green', label='M')
#
#    ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
#    ax.legend()
#
#    plt.show()
#        

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
#                if r.power_file!="" and (r.discr==discrs[d-1] and r.test_name==tests[i] and r.devName in names):
#                    logs.append(r.power_file)
#                    dev_names.append(r.devName)
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


