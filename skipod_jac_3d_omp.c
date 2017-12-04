#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))


#define  N   (2*2*2*2*2*2+2)
static const double   maxeps = 0.1e-7;
static const int itmax = 5000;
int i,j,k;
double eps;
double A [N][N][N],  B [N][N][N];

void relax();
void resid();
void init();
void verify(); 

double e;
int THREADS;
void wtime(double *t)
{
    *t = omp_get_wtime();
}

int main(int argc, char **argv)
{
    if(argc <= 1){
       return -1;
    }
	THREADS = atoi(argv[1]);
	double time_start, time_fin;
	int it;
    init();
    wtime(&time_start);
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		// printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	wtime(&time_fin);
    printf("Time: %gs\t", time_fin - time_start);
	verify();
	return 0;
}

void init()
{ 
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
} 

void relax()
{
#pragma omp parallel num_threads(THREADS) private(i,j,k) 
{
    #pragma omp for schedule(static) 
	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		B[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.;
	}
}
}

void resid()
{ 
double global = eps;
#pragma omp parallel num_threads(THREADS) shared(global)  private(i,j,k) 
{
	double local = eps;
	#pragma omp for  schedule(static)
	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		double e = fabs(A[i][j][k] - B[i][j][k]);         
		A[i][j][k] = B[i][j][k]; 
		local = Max(local,e);
	}

	#pragma omp critical
	{
		if(local > global){
			global = local;
		}
	}	
}
	eps = Max(fabs(e),global);
}

void verify()
{
	double s;
	s=0.;
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	printf("  S = %f\n",s);
}