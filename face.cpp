#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include "Data.cpp"


using namespace std;

class Face
{
public:
    Data D;
    double first_delta[960][20];
    double second_delta[20][20];
    double first_tho[20];
    double first[960][20]; // weight of edges between 1st and 2nd layer
    double second[20][20];// weight of edges between 2nd and 3rd layer
    double error,err;
    double mom;
    double eta;
    double gamma;
    double out1[960],out2[20],O[20]; // output of each layer 
    double *T;// output array , correct output for input image
    double *mat;// input array , holds the greyscale values of input image
    int lim, lim2;
    void sigmoid_1() //calculates output of sigmoid units in 1st layer
    {
        double sum;
        int i,k=0;
        while(k<960)
        {
            out1[k]=mat[k];
            k++;
        }

    }
    void sigmoid_2() // calculates output of sigmoid units in hidden layer
    {
        double sum=0.00;
        int i,k=0;
        while(k<20)
        {
            sum=0.00;
            for(i=0;i<960;i++)
            {
                sum+=first[i][k]*out1[i];

            }
            out2[k]=1/(1+exp(-sum));
            k++;
        }
    }
    void sigmoid_3() // calculates output of sigmoid units in output layer
    {
        double sum=0.00;
        int i,k=0;
        while(k<20)
        {
            sum=0.00;
            for(i=0;i<20;i++)
            {
                sum+=second[i][k]*out2[i];
            }
            O[k]=1/(1+exp(-sum));
            k++;
        }

    }
    void delta_2() // calculates change in wiegts for edges between 2nd and 3rd layer
    {
        int i=0,k=0;
        while(i<20)
        {
            k=0;
            while(k<20)
            {
                second_delta[i][k]=eta*(-((T[k]-O[k])*O[k]*(1-O[k])*out2[i])+2*gamma*second[i][k]) + mom*second_delta[i][k];
                k++;
            }
            i++;
        }
    }
    void delta_1()// calculates change in wiegts for edges between 1st and 2nd layer
    {
        double sum=0.00;
        int j,i=0,k=0;
        while(k<20)
        {
           sum=0.00;
           for(j=0;j<20;j++)
            {
                sum-=O[j]*(1-O[j])*(T[j]-O[j])*second[k][j];
            }
            first_tho[k]=out2[k]*(1-out2[k])*sum;
            k++;
        }
        while(i<960)
        {
            k=0;
            while(k<20)
            {
                first_delta[i][k]=eta*(first_tho[k]*out1[i]+2*gamma*first[i][k])+mom*first_delta[i][k];
                k++;
            }
            i++;
        }

    }
    void update_2() // updates wiegts for edges between 2nd and 3rd layer
    {
        int i,j;
        for(i=0;i<20;i++)
        {
            for(j=0;j<20;j++)
            second[i][j]=second[i][j]+second_delta[i][j];
        }
    }
    void update_1() // updates wiegts for edges between 1st and 2nd layer
    {
        int i,j;
         for(i=0;i<960;i++)
        {
            for(j=0;j<20;j++)
            first[i][j]=first[i][j]+first_delta[i][j];
        }
    }
    Face()//constructor sets weights between -0.5 and 0.5 randomly , also sets network parameters like learning rate , momentum
    {
        int i,j;
        D.Init(2);
        mom=0.3;
        eta=-0.3;
        gamma=1/(exp(18));
        error=1;
        for(i=0;i<960;i++)
        {
            for(j=0;j<20;j++)
            {
                first[i][j]=(0.5+(((rand())%1000)*1.0)/1000) -1;
                first_delta[i][j]=0;
            }
        }
        for(i=0;i<20;i++)
        {
            for(j=0;j<20;j++)
            {
                second[i][j]=(0.5+(((rand())%1000)*1.0)/1000) -1;
                second_delta[i][j]=0;
            }
            first_tho[i]=0;
        }
        lim = D.loadTrainingData();
        lim2 = D.loadValidationData(0);

    }

    void train()// runs 50 iterations each time 
    {
        int k=0;
        int j=0;
        while(j<50)
        {
            k=0;
            while(k<lim)
            {
                pair<double*,double*> p = D.getTrainingData();
                mat=p.first;
                T=p.second;
                sigmoid_1();
                sigmoid_2();
                sigmoid_3();
                delta_2();
                delta_1();
                update_2();
                update_1();
                k++;
            }
            j++;
        }
    }
    void validateFinal()
    {
        lim2 = D.loadValidationData(1);
        validate();
    }
    void validate()
    {
        int k=0;
        error=0;
        while(k<lim2)
        {
            int i=0;
            int max_pos;
            pair<double*,double*> p = D.getValidationData();
            mat=p.first;
            T=p.second;
            sigmoid_1();
            sigmoid_2();
            sigmoid_3();
            double max=O[0];
            for(i=0;i<20;i++)
            {
                if(O[i]>=max)
                {
                    max=O[i];
                    max_pos=i;
                }
            }
            if(T[max_pos]<0.6)
            {
                error++;
            }
            k++;
        }
        error=error/lim2;
    }
};

double first[960][20],second[20][20];

// void print(char *file, int l,int m, int n, bool sec,double error)
// {
//     FILE *f = fopen(file,"w");
//     int i,j;
//     fprintf(f, "Accuracy %lf%%\n",(1-error)*100);

//     fprintf(f, "Weights in first level\n");
//     for(i=0;i<l;i++)
//     {
//         for(j=0;j<m;j++)
//         {
//             fprintf(f, "%lf,",first[i][j] );
//         }
//         fprintf(f, "\n");
//     }
//     fprintf(f, "\n");

//     if(sec)
//     {
//         fprintf(f,"Weights in second level\n");
//         for(i=0;i<m;i++)
//         {
//             for(j=0;j<n;j++)
//             {
//                 fprintf(f,"%lf,",first[i][j] );
//             }
//             fprintf(f, "\n");
//         }
//     }
// }

int main()
{
    Face p;
    int i,j;
    double perror,error;

    perror = 9999;
    error = 0;

    p.validate();
    error = p.error;
    printf("Initial accuracy %lf%%\n",(1-error)*100);

    for(int z=1;true;z++)
    {
        p.train();
        p.validate();
        error = p.error;
        printf("Accuracy after %d iterations %lf%%\n",z*50*277,(1-error)*100);
        if(perror <= error) // stoping criteria , if error on validation set increases then training stops
        {
            for(i=0;i<960;i++)
                for(j=0;j<20;j++)
                    p.first[i][j]=first[i][j];
            for(i=0;i<20;i++)
                for(j=0;j<20;j++)
                    p.second[i][j]=second[i][j];
            p.validateFinal();
            error = p.error;
            printf("Accuracy on final validation set %lf%%",(1-error)*100);
            // char fl[] = "face.csv";
            // print(fl,960,20,20,true,error);
            break;
        }
        for(i=0;i<960;i++)
            for(j=0;j<20;j++)
                first[i][j]=p.first[i][j];
        for(i=0;i<20;i++)
        {
            for(j=0;j<20;j++)
            {
                second[i][j]=p.second[i][j];
            }
        }
        perror = error;
    }
}