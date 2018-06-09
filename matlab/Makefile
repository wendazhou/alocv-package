cfiles:
	mex -v ../C/src/*.c -I../C/include -lmwblas -lmwlapack -c -outdir ./obj

cppfiles:
	mex -v ../C/src/*.cpp -I../C/include/ -lmwblas -lmwlapack -c -outdir ./obj

all: cfiles cppfiles
	mex -v alo_lasso_mex.c -I../C/include/ -lmwblas -lmwlapack ./obj/*.o