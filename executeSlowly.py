from os import system
from time import sleep
import sys
from datetime import datetime, timedelta
#threads=[128, 256, 512]
#jumps=[2, 3, 17]
#discrs=[4, 16, 90, 360]
#refinements=[0, 1, 2, 4, 8, 16]
#functions=[2,1,0]
#tests=range(6)
#guessAngles=False
#nExecs=range(1, 3)
#name="XavierOverClock"
interval=50 #Sampling interval in ms

threads=[256]
jumps=[2, 17]
discrs=[360]
refinements=[8, 16]
functions=[2]
tests=range(4,6)
guessAngles=False
nExecs=range(1)
lastExec=-1
name=input("Insert the name of the test (device): ")

system("mkdir -p "+name)
system("tegrastats --stop && tegrastats --start --logfile tegrastats"+name+str(nExecs[0])+"to"+str(nExecs[-1])+".log")
start=datetime.now()

for nExec in nExecs:
	for thread in threads:
		for func in functions:
			if func!=0:
				guessAngles=True
			for jump in jumps:
				if func!=2 and jump!=2:
					pass
				else:
					for discr in discrs:
						for ref in refinements:
							for test in tests:
								testName=name+"_"+str(thread)+"_"+str(func)+"_"+str(jump)+"_"+str(discr)+"_"+str(ref)+"_"+str(test)+"_"+str(guessAngles)
								_start=datetime.now()
								elapsedStart=(int(((datetime.now()-start).total_seconds()*1000)/interval)+1)*interval
								system("./bin/cu/main.out "+testName+" "+str(nExec)+" "+str(test)+" "+str(discr)+" "+str(ref)+" "+str(func)+" "+str(int(guessAngles))+" "+str(thread)+" "+str(jump))
								elapsedStop=int(((datetime.now()-start).total_seconds()*1000)/interval)*interval
								with open("times.json", "a+") as f:
									f.write('{"name" : "'+testName+'_'+str(nExec)+'", "start": '+str(elapsedStart)+', "stop": '+str(elapsedStop)+'}\n')
								#sleep(30)
	lastExec=nExec

system("tegrastats --stop")
system("mv tegrastats"+name+str(nExecs[0])+"to"+str(nExecs[-1])+".log "+name+"/tegrastats"+name+str(nExecs[0])+"to"+str(lastExec)+".log")

#            sleep(30)
