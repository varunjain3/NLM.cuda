make 2d:
	nvcc --resource-usage -std=c++11 main2d.cu -o main2d `pkg-config --cflags --libs opencv` -Xcompiler -fopenmp
	./main2d

make 1d: 
	nvcc -std=c++11 main.cu -o main `pkg-config --cflags --libs opencv`
	./main