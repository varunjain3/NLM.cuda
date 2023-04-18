make all:
	g++ -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`
	./main