#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include "Data.cpp"


using namespace std;

class shades
{
public:
    Data D;
    double first_delta[960][2];
    double first[960][2];
    double error,err;
    double mom;
    double eta;
    double gamma;
    double out1[960],O[2];
    double *T;
    double *mat;
    double delta;
    int lim, lim2;
    void sigmoid_1()
    {
        double sum;
        int i,k=0;
        while(k<960)
        {
            out1[k]=mat[k];
            k++;
        }

    }
    void sigmoid_2()
    {
        double sum=0.00;
        int i,k=0;
        while(k<2)
        {
            sum=0.00;
            for(i=0;i<960;i++)
            {
                sum+=first[i][k]*out1[i];

            }
            O[k]=1/(1+exp(-sum));
			k++;
        }
    }

    void delta_1()
    {
        int i=0,k=0;
        while(i<960)
        {
            k=0;
            while(k<2)
            {
                first_delta[i][k]=eta*(-((T[k]-O[k])*O[k]*(1-O[k])*out1[i])+2*gamma*first[i][k]) + mom*first_delta[i][k];
                k++;
            }
            i++;
        }
    }
    void update_1()
    {
        int i,j;
        for(i=0;i<960;i++)
        {
            for(j=0;j<2;j++)
            first[i][j]=first[i][j]+first_delta[i][j];
        }
    }
    shades()
    {
        int i,j;
        D.Init(1);
        mom=0.3;
        eta=-0.3;
        gamma=1/(exp(18));
        error=1;
        for(i=0;i<960;i++)
        {
            for(j=0;j<2;j++)
            {
                first[i][j]=(0.5+(((rand())%1000)*1.0)/1000) -1;
                first_delta[i][j]=0;
            }
        }
        lim = D.loadTrainingData();
        lim2 = D.loadValidationData(0);

    }

    void train()
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
				delta_1();
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
            double max=O[0];
            for(i=0;i<2;i++)
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


double first[960][2];

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
    shades p;
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
        if(perror <= error)
        {
            for(i=0;i<960;i++)
                for(j=0;j<2;j++)
                    p.first[i][j]=first[i][j];

            p.validateFinal();
            error = p.error;
            printf("Accuracy on final validation set %lf%%",(1-error)*100);
            // char fl[] = "pose.csv";
            // print(fl,960,6,4,true,error);
            break;
        }
        for(i=0;i<960;i++)
            for(j=0;j<2;j++)
                first[i][j]=p.first[i][j];
        perror = error;
    }
}