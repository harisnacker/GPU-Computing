cuda:
	nvcc -o app main.cu kernel.cu -O2 -L/usr/X11R6/lib -lm -lpthread -lX11   

cpu:
	g++ -o app test.cpp kernelcpu.cpp -O2 -L/usr/X11R6/lib -lm -lpthread -lX11   

clean:
	rm app
