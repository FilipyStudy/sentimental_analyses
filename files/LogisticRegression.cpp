#define PY_SSIZE_T_CLEAN
#define T_TEMP	1
#include <"Python.h">
#include <iostream>
#include <vector>

float expFloat (float value, int exp) {
	if (exp == 0) return 1;
	else if (exp == 1) return value;
	else {
		int flag = 1;
		float tempValue = value;
		while (flag < exp){
		tempValue = tempValue * value;
		flag += 1;
		}
		return tempValue;
	}
}

std::vector<float> SoftMax(std::vector<float> a){
	int totalOfMembers = a.size();
	int i = 0;
	int flag = 1;
	float arrSum = 0;
	int x = 0
	while(i < totalOfMembers){
		a[i] = expFloat(a[i], flag);
		i++;
		flag++;
	}
	for (float i: a) arrSum += i;
	while(x < a.size()){
		a[x] = a[x] / arrSum;
		x++;
	}
	return a;
} 

int main(){
	return 0;
}
