MATLABDIR = /usr/local/MATLAB/R2016a/
FILELOC = /home/fabian/Documents/Arbeit/Code/MyCode/LogConcave/code/newtonInit.mat
#MATLABDIR = /opt/MATLAB/R2016b/
#FILELOC = /export/home/frathke/Documents/Code/MyCode/LogConcave/code/newtonInit.mat
MATLABINCLUDE = $(MATLABDIR)/extern/include/
MATLABLIB = $(MATLABDIR)/bin/glnxa64/

#FLAGSGCC = -O3 -std=c++11 -fPIC -pedantic-errors -Wall -fopenmp -lm -D NUMCORES=$(NUMBEROFCORES)
FLAGSGCC = -O3 -funroll-loops -ftree-vectorize -std=c99 -fPIC -pedantic-errors -fopenmp -Wall -Wextra -I/usr/share/R/include/ -I. -march=native

clean:
	rm *.o 

all: newton debug

full: makeGridC.o CNS.o calcGradAVX.o calcGradC.o preCondGradAVXC.o calcGradFastFloatC.o calcGradFastC.o util.o bfgsFullC.o calcGradFloatC.o preCondGradFloatC.o
	gcc callerFull.c -o runFull -I$(MATLABINCLUDE) -L$(MATLABLIB) $(FLAGSGCC) -Wl,-rpath,$(MATLABLIB) -DFILELOC='"$(FILELOC)"' $^ -lm -lR -fopenmp -lmx -lmat -lmex 

debug: makeGridC.o CNS.o calcGradAVX.o calcGradC.o preCondGradAVXC.o calcGradFastFloatC.o calcGradFastC.o util.o bfgsFullC.o calcGradFloatC.o preCondGradFloatC.o
	gcc -g callerFull.c -o runDebug -I$(MATLABINCLUDE) -L$(MATLABLIB) $(FLAGSGCC) -O0 -Wl,-rpath,$(MATLABLIB) -DFILELOC='"$(FILELOC)"' $^ -lm -lR -fopenmp -lmx -lmat -lmex 

init: makeGridC.o calcGradFullAVXC.o calcGradFloatC.o CNS.o util.o bfgsInitC.o
	gcc -g callerInit.c -o runInit -I$(MATLABINCLUDE) -L$(MATLABLIB) $(FLAGSGCC) -L/usr/lib/R/lib/ -Wl,-rpath,$(MATLABLIB) -DFILELOC='"$(FILELOC)"' $^ -lm -lR -fopenmp -lmx -lmat -lmex 

%.o: %.c
	gcc -c -g $(FLAGSGCC) -O0 -I$(MATLABINCLUDE) $<	

