#include<bits/stdc++.h>
using namespace std;

float sigmoid(float z){
    return 1/(1+exp(-1*z));
}

float dot(vector<float> row,vector<float> w){
    float sum=0;
    for(int i=0;i<2;i++){
        sum+=row[i]*w[i];
    }
    return sum;
}



vector<vector<float>> gradient_logic(int m,vector<vector<float>> X_train,float b,vector<float> Y_train,vector<float> w){
    vector<float> dj_dw={0.0,0.0};
    float dj_db=0;
    for(int i=0;i<m;i++){
        float z=dot(X_train[i],w)+b;
        float func=sigmoid(z);
        float err=func-Y_train[i];
        //cout<<"err = "<<err<<endl;
        for(int j=0;j<2;j++){
            dj_dw[j]+=err*X_train[i][j];
        }
        dj_db=dj_db+err;
    }
    for(int i=0;i<dj_dw.size();i++){
        dj_dw[i]/=m;
    }
    dj_db/=m;
    return {dj_dw,{dj_db}};
}

float cost_function(int m,vector<vector<float>> X_train,float b,vector<float> Y_train,vector<float> w){
    float cost=0;
    for(int i=0;i<m;i++){
        float z=dot(X_train[i],w)+b;
        float func=sigmoid(z);
        float lossfunc= -Y_train[i]*log(func)-(1-Y_train[i])*log(1-func);
        cost+=lossfunc;
    }
    cost/=m;
    return cost;
}

vector<float> mulalpha(vector<float> dj_dw,float alpha,vector<float> wtemp){
    for(int i=0;i<dj_dw.size();i++){
        // cout<<"dw = "<<dj_dw[i]<<endl;
        // cout<<dj_dw[i]*alpha<<endl;
        dj_dw[i]=wtemp[i]-dj_dw[i]*alpha;
    }
    return dj_dw;
}

vector<vector<float>> gradient_descent(vector<vector<float>> X_train,vector<float> Y_train,vector<float> w,float b,int m){
    vector<float> wtemp=w;
    float btemp=b;
    float alpha=0.1;
    for(int i=0;i<10000;i++){
        vector<vector<float>> gradient=gradient_logic(m,X_train,btemp,Y_train,wtemp);
        vector<float> dj_dw=gradient[0];
        float dj_db=gradient[1][0];
        wtemp=mulalpha(dj_dw,alpha,wtemp);
        btemp-=alpha*dj_db;
        if(i%1000==0) cout<<cost_function(m,X_train,btemp,Y_train,wtemp)<<endl;
    }
    return {wtemp,{btemp}};

}   

void print2darr(vector<vector<float>> v){
    for(int i=0;i<v.size();i++){
        for(int j=0;j<v[i].size();j++){
            cout<<v[i][j]<<" ";
        }
        cout<<endl;
    }
}
int main(){
    vector<vector<float>> X_train={
        {0.5,1.5},
        {1,1},
        {1.5,0.5},
        {3,0.5},
        {2,2},
        {1,2.5}
    };
    vector<float> Y_train={
        0,0,0,1,1,1
    };
    vector<float> w={
        0.0,0.0
    };
    int m=X_train.size();
    float b =0.0;
    vector<vector<float>> ans=gradient_descent(X_train,Y_train,w,b,m);
    cout<<"w =";
    for(int i=0;i<ans[0].size();i++){
        cout<<ans[0][i]<<" ";
    }
    cout<<"\nb ="<<ans[1][0]<<endl;
    // print2darr(ans);
    return 0;
}