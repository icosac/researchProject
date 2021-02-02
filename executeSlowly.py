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

threads=[128]
functions=[1]
jumps=[0]
discrs=[4, 16, 90, 360]
refinements=[0, 1, 2, 4, 8, 16]
tests=range(6)
guessAngles=True
nExecs=range(1)
name=sys.argv[2]

system("mkdir -p "+name)
start=datetime.now()

for nExec in nExecs:
	system("tegrastats --stop && tegrastats --start --logfile tegrastats"+name+"_"+str(nExec)+".log")
	for thread in threads:
		for func in functions:
			if func!=0:
				guessAngles=True
			for jump in jumps:
				if func!=2 and jump!=jumps[0]:
					pass
				else:
					for discr in discrs:
						for ref in refinements:
							for test in tests:
								testName=name+"_"+str(thread)+"_"+str(func)+"_"+str(jump)+"_"+str(discr)+"_"+str(ref)+"_"+str(test)+"_"+str(guessAngles)+"_"+str(nExec)
								print(testName)
								_start=datetime.now()
								elapsedStart=(int(((datetime.now()-start).total_seconds()*1000)/interval)+1)*interval
								print("elapsedStart: "+str(elapsedStart))
								system("./bin/cu/main.out "+testName+" "+str(nExec)+" "+str(test)+" "+str(discr)+" "+str(ref)+" "+str(func)+" "+str(int(guessAngles))+" "+str(thread)+" "+str(jump)+" "+str(elapsedStart))
								elapsedStop=int(((datetime.now()-start).total_seconds()*1000)/interval)*interval
								with open("times.json", "a+") as f:
									f.write('{"name" : "'+testName+'", "start": '+str(elapsedStart)+', "stop": '+str(elapsedStop)+'}\n') #This is just for backup, the same data is stored also in each run in tests.json
								#sleep(30)
	system("tegrastats --stop")
	system("mv tegrastats"+name+"_"+str(nExec)+".log "+name+"/tegrastats"+name+"_"+str(nExec)+".log")


#            sleep(30)
