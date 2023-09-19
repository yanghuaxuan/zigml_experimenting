build: twice twice.o

twice twice.o: twice.zig
	zig build-exe $?

clean: twice 
	rm $? 
	rm *.o

.PHONY: clean build