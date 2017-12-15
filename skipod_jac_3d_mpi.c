#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "mpi.h"
#define  Max(a,b) ((a)>(b)?(a):(b))
#define m_printf if (wrank == 0) printf

#define  N   (2*2*2*2*2*2+2)
static const double   maxeps = 0.1e-7;
static const int itmax = 2000;
int i,j,k;
double eps, sum;
double A [N][N][N];
double B [N][N][N];

void relax();
void resid();
void init();
void verify();

int block, startrow, lastrow;
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
    MPI_Barrier(MPI_COMM_WORLD); // wait for all process here

        // Split between processors
    block = N / wsize;
    startrow = block * wrank;
    lastrow =  block * (wrank+1) - 1;
    m_printf("Jacobian 3D started!\n");

        // Initialize matrices 
    init();
    if (!wrank)
        wtime(&time_start);

    for(it=1; it<=itmax; it++)
    {
		eps = 0.;
		relax();
        update(wrank,wsize);
		resid();
        update(wrank,wsize);
		// printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
    verify();
    if(!wrank){
        wtime(&time_fin);
        printf("Time: %gs\t", time_fin - time_start);
        printf("eps = %f\t", eps);
        printf("S = %f\n", sum);
    }

    	// Finalize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

	return 0;
}

void init()
{
	for(i=startrow; i<=lastrow; i++)
    for(j=0; j<=N-1; j++)
    for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
}

void relax()
{
	for(i=startrow; i<=lastrow; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
        if (i == 0 || i == N-1) continue;
		B[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1])/6.;
	}
}

void resid()
{
    double local_eps = eps;

	for(i=startrow; i<=lastrow; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
        if (i == 0 || i == N-1) continue;
		double e = fabs(A[i][j][k] - B[i][j][k]);
		A[i][j][k] = B[i][j][k];
		local_eps = Max(local_eps,e);
	}
        //Max local_eps from each process
    MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

}

void verify()
{
	double s = 0.0;
	for(i=startrow; i<=lastrow; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
        // Sum all parts of A to check answer
    MPI_Allreduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void update(int rank, int size)
{
    MPI_Request request[4];
    MPI_Status status[4];
        // Update neighbours
    if (rank)
        MPI_Irecv(&A[startrow-1][N][N], N*N, MPI_DOUBLE, rank-1, 1215, MPI_COMM_WORLD, &request[0]);
    if(rank)
        MPI_Isend(&A[startrow][N][N], N*N, MPI_DOUBLE, rank-1, 1216, MPI_COMM_WORLD, &request[1]);
    if(rank != size-1)
        MPI_Isend(&A[lastrow][N][N], N*N, MPI_DOUBLE, rank+1, 1215, MPI_COMM_WORLD, &request[2]);
    if (rank != size-1)
        MPI_Irecv(&A[lastrow+1][N][N], N*N, MPI_DOUBLE, rank+1, 1216, MPI_COMM_WORLD, &request[3]);

        // Wait for processes
    int ll = 4, shift = 0; // all processes 
    if(!rank) { // first process
        ll = 2;
        shift = 2;
    }
    if(rank == size-1) { // last process
        ll = 2;
    }
    MPI_Waitall(ll, &request[shift], &status[0]);
}
