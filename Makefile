lr:LogisticRegression.o main.o 
	g++ -o lr LogisticRegression.o main.o
LogisticRegression.o:LogisticRegression.cpp LogisticRegression.h matrix.h
	g++ -c LogisticRegression.cpp 

main.o:main.cpp LogisticRegression.h
	g++ -c main.cpp
clean:
	rm -f *.o lr
