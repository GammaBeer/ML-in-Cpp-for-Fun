#include<bits/stdc++.h>
using namespace std;

float X_train[][2]={
    {0.5,1.5},
    {1,1},
    {1.5,0.5},
    {3,0.5},
    {2,2},
    {1,2.5}
};

float Y_train[]={
    0,0,0,1,1,1
};

float w[]={
    1,1
};


#define m sizeof(X_train)/sizeof(X_train[0])

#define b -3;


float sigmoid(float z){
    return 1/(1+exp(-1*z));
}

float dot(float row[]){
    float sum=0;
    for(int i=0;i<2;i++){
        sum+=row[i]*w[i];
    }
    return sum;
}

float compute_function(){
    float cost=0;
    for(int i=0;i<m;i++){
        float z=dot(X_train[i])+b;
        float func=sigmoid(z);
        float lossfunc= -Y_train[i]*log(func)-(1-Y_train[i])*log(1-func);
        cost+=lossfunc;
    }
    cost/=m;
    return cost;
}

int main(){
    //compute function
    cout<<compute_function();
    return 0;
}