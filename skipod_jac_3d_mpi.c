#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))


#define  N   (2*2*2*2*2*2+2)
static const double   maxeps = 0.1e-7;
static const int itmax = 2000;
int i,j,k;
double eps, sum;
double A [N][N][N],  B [N][N][N];

void relax();
void resid();
void init();
void verify(); 

double e;
int block, i1, i2;
void update(int rank,int size);
void wtime(double *t)
{
    *t = MPI_Wtime();
}

int main(int argc, char **argv)
{
	int it;
	double time_start, time_fin;
        // Initialize MPI
    int wrank, wsize;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

        // Split between processors
    block = (N-2) / wsize; 
    int check = (wrank+1 == wsize);
    i1 = block * wrank;
    i2 = check ? N-2 : block * (wrank+1); 
    init();
    if (!wrank)
        wtime(&time_start);
	
    for(it=1; it<=itmax; it++)
    {
		eps = 0.;
		relax();
        // update(wrank,wsize); //Update neighbours
		resid();
        // update(wrank,wsize); //Update neighbours
		// printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
    verify();
    if(!wrank){
        wtime(&time_fin);
        printf("Time: %gs\t", time_fin - time_start);
        printf("eps = %f\t", eps);
        printf("  S = %f\n",sum/(N*N*N));        
    }

    	// Finalize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

	return 0;
}

void init()
{ 
	for(i=i1;i<=i2+1;i++)
    for(j=0; j<=N-1; j++)
    for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
} 

void relax()
{
	for(i=i1+1;i<=i2;i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		B[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.;
	}
}

void resid()
{ 
    double global = eps;
    double local = eps;

	for(i=i1+1;i<=i2;i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		double e = fabs(A[i][j][k] - B[i][j][k]);         
		A[i][j][k] = B[i][j][k]; 
		local = Max(local,e);
	}
        //Find max local in cycle and pass it to eps
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    eps = global;

}

void verify()
{
	double s = 0.0;
	for(i=i1+1;i<=i2;i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1);
	}
        //Reduce all parts of A to check answer
    MPI_Reduce(&s,&sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
}

void update(int rank, int size)
{
    MPI_Request request_up;
    MPI_Request request_down;
    MPI_Status status;

    if(rank!=0){
        MPI_Isend(&A[(i1+1)*N*N] ,(i2-i1)*N*N,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,&request_up);
    
    }
    if(rank!=size-1){
        MPI_Isend(&A[(i1+1)*N*N],(i2-i1)*N*N,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,&request_down);
        MPI_Isend(&B[(i1+1)*N*N],(i2-i1)*N*N,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,&request_down);
    }
    if (rank != 0){
        MPI_Recv(&A[(block*(rank-1)+1)*N*N], (i2-i1)*N*N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv(&B[(block*(rank-1)+1)*N*N], (i2-i1)*N*N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
    if (rank != size-1)
        if(rank==size-2){
            MPI_Recv(&A[(block*(rank+1)+1)*N*N], ((N-2)-i1)*N*N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            MPI_Recv(&B[(block*(rank+1)+1)*N*N], ((N-2)-i1)*N*N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );        
        }else{
            MPI_Recv(&A[(block*(rank+1)+1)*N*N], (i2-i1)*N*N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            MPI_Recv(&B[(block*(rank+1)+1)*N*N], (i2-i1)*N*N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
    if (rank != 0) {
        MPI_Wait( &request_up, &status );
    }
    if (rank != size-1) {
        MPI_Wait( &request_down, &status );
    }

}