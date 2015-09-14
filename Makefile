lr:lr.o matrix.o
	g++ -o lr lr.o matrix.o
lr.o:LogisticRegression.cpp LogisticRegression.h
	g++ -c LogisticRegression.cpp 
matrix.o:matrix.h
	g++ -c matrix.h
clean:
	rm matrix.o lr.o lr
