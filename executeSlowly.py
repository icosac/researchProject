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
jumps=[3]
discrs=[4, 16, 90, 360]
refinements=[0, 1, 2, 4, 8, 16]
functions=[2]
tests=range(6)
guessAngles=True
nExecs=range(1, 10)
name=""
initStart=0.0
if len(sys.argv)==1:
    name=input("Insert the name of the test (device): ")
else:
    name=sys.argv[2]
    initStart=float(sys.argv[1])


system("mkdir -p "+name)
start=datetime.now()

initTimeStamp=datetime.now().timestamp()*1000.0
offset=initTimeStamp-initStart

for nExec in nExecs:
	#system("tegrastats --stop && tegrastats --start --logfile tegrastats"+name+"_"+str(nExec)+".log")
	for thread in threads:
		for func in functions:
			if func!=0:
				guessAngles=True
			for jump in jumps:
				if func!=2 and jump!=0:
					pass
				else:
					for discr in discrs:
						for ref in refinements:
							for test in tests:
								testName=name+"_"+str(thread)+"_"+str(func)+"_"+str(jump)+"_"+str(discr)+"_"+str(ref)+"_"+str(test)+"_"+str(guessAngles)+"_"+str(nExec)
								print(testName)
								_elapsedStart=(datetime.now()-start).total_seconds()*1000.0+offset
								elapsedStart=((int(_elapsedStart)/interval)+1)*interval
								system("./bin/cu/main.out "+testName+" "+str(nExec)+" "+str(test)+" "+str(discr)+" "+str(ref)+" "+str(func)+" "+str(int(guessAngles))+" "+str(thread)+" "+str(jump)+" "+str(elapsedStart))
								_elapsedStop=(datetime.now()-start).total_seconds()*1000.0+offset
								elapsedStop=int(_elapsedStop/interval)*interval
								#sleep(30)
	#system("tegrastats --stop")
	#system("mv tegrastats"+name+"_"+str(nExec)+".log "+name+"/tegrastats"+name+"_"+str(nExec)+".log")

