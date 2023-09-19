build: twice twice.o gates gates.o

twice twice.o: twice.zig
	zig build-exe $?

gates gates.o: gates.zig
	zig build-exe $?

clean: twice gates
	rm $? 
	rm *.o

.PHONY: clean build