#define PY_SSIZE_T_CLEAN
#define T_TEMP	1
#include <Python.h>
#include <stdio.h>
#include <vector>
class RegressionModels
{

#Exponentiation function based on the exponentiation by squaring.
long double ExpFloat (long double value, int exp) {
	if (exp == 0) return 1;
	else if(exp == 1) return value;
	else if(exp < 0){
		return ExpFloat(1 / value, exp * (-1));
	}
	else if(exp % 2 == 0){
		return ExpFloat(value * value, exp / 2)
	}
	else if(exp % 2 != 0) {
		return ExpFloat(value * value, (exp - 1) / 2);
		}
	return NULL;
	}

std::vector<long double> SoftMax(std::vector<long double> a){
	long double arrSum = 0;
	for(int i = 0; i < a.size(); i++){
		a[i] = expFloat(a[i], i + 1);
		arrSum += a[i];
	}
	for (int x = 0; x < a.size(); x++) a[x] /= arrSum;

	return a;
	} 

std::vector<long double> Slope(std::vector<long double> Y, std::vector<long double>X){
	int n;
	long double xySum,
	     	    ySum,
		    xSum;

	if(Y.size() == X.size()) n = Y.size();
	
	else{
		cout << "Vector X and Y are not the same size" << endl;
		return 1.0;
	}
	
	for(int i = 0; i < n; i++){
	ySum += Y[i];
	xSum += X[i];
	xySum += Y[i] + X[i];
	}
	std::vector<long double> slopeResult = {xySum, ySum, xSum};
	return slopeResult;
	};

long double MeanSquaredError(std::vector<long double> observedVector, std::vector<long double> predictedVector){
	long double totalSum = 0;
	for(int i = 0; i < observedVector.size(); i++){
		totalSum += ExpFloat((observedVector[i] + predictedVector[i]) , 2);
	}
	return totalSum / 0.5;
}

#TODO: Finish the simple linear regression model calc;
std::vector<long double> SimpleLinearRegression (
		long double slope, 
		long double iVar,
		long double yIntercept){
	
	
	return valuesVector;
	}
}
