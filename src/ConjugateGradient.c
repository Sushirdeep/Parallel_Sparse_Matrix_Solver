#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mmio.h"

#define NUM_THREADS 2

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////FUNCTION         :       ConjugateGradient//////////////////////////////////////////////////////////////////////////
//////////////INPUTs           :                        //////////////////////////////////////////////////////////////////////////
//////////////MAT              :       Input CSR Matix  //////////////////////////////////////////////////////////////////////////
//////////////dim              :       Matrix dimension //////////////////////////////////////////////////////////////////////////
///////////// x_k              :       Final Solution Vector//////////////////////////////////////////////////////////////////////
///////////// b                :       Input RHS Vector //////////////////////////////////////////////////////////////////////////
///////////// init_guess_x0    :       Initial guess for x ///////////////////////////////////////////////////////////////////////
///////////// tolerance        :       threshold value for tolerance /////////////////////////////////////////////////////////////
///////////// max_iterations   :       Maximum allowed number of iteration for convergance ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ConjugateGradient(MTX *MAT, double dim, double* x_k, double* b, double* init_guess_x0,  double tolerance, int max_iterations){

	// File to hold the various timing values, used for debug of the parallel solver 
	FILE* optimization = fopen("./Output/optimization.out","w");

	//omp_set_num_threads(4);
	
	// Set number of chunks to direct the open MP compiler for work sharing 
	int chunk = 1000;
	int num_cols = dim;

	//Dot product result of Apk with Apk
	double Apk_dot_pk;

	// Dot product of rk in the previous iteration
	double rk_dot_rk_prev;

	// Dot product of rk in the current iteration
	double rk_dot_rk_new;

	// Residue calculated at every iteration
	double residue = 1.0; 

	//Initial Residue 	
	double residue_init = 0.0;

	//Relative residue value used for print and convergence check
	//Tolerance set to 10^-8
        double relative_residue;

	double alpha, beta;

	//Number of iterations
	int numIterations = 0;

	//timing related declarations
	double start_time_vectorUpdates, run_time_vectorUpdates;
	double start_time_matrixVectorMult, run_time_matrixVectorMult;

	// Allocating Memories for the variables in the convergence loop
	double* r_k = (double*) malloc(dim*sizeof(double));
	double* p_k = (double*) malloc(dim*sizeof(double));
	double* Ap_k = (double*) malloc(dim*sizeof(double));


	//Matrix Vector product optimised for parallel calculations
	//Starting the timer	
	start_time_matrixVectorMult = omp_get_wtime();
	matrixVectorProduct(r_k, MAT,init_guess_x0);
	run_time_matrixVectorMult = omp_get_wtime() - start_time_matrixVectorMult;
	fprintf(optimization,"Time taken for Multiplication = %lf s\n",run_time_matrixVectorMult);
	

	// Optimised vector updates
	start_time_vectorUpdates = omp_get_wtime();
	#pragma omp parallel num_threads(NUM_THREADS)
	{ 	
		#pragma omp for schedule(static,chunk)
		for(int i=0;i<num_cols;i++) 
		{	
			r_k[i] = b[i] - r_k[i];
			x_k[i] = init_guess_x0[i];
			p_k[i] = r_k[i];
		}
	}

	run_time_vectorUpdates = omp_get_wtime() - start_time_vectorUpdates;
	fprintf(optimization,"Time taken for vector updates = %lf s\n",run_time_vectorUpdates);

	//(r0,r0)
	rk_dot_rk_prev = vectorDotProduct(r_k,r_k,dim);
	
	//Calculating the initial residue	
	residue_init = pow(rk_dot_rk_prev,0.5);

	// keep iterating the loop till one of the following condition fails
	// 1. residue calculated at every iteration < tolerance * initial residue
	// 2. The loop has exceed the maximum number of iterations
	while((residue > (tolerance * residue_init)) && (numIterations < max_iterations)){

		// Increment the number of iterations
		numIterations = numIterations + 1;

		// Matrix vector product
		matrixVectorProduct(Ap_k, MAT,p_k);

		//Dot product of Apk with pk
		Apk_dot_pk = vectorDotProduct(Ap_k,p_k,dim);

		alpha = rk_dot_rk_prev/Apk_dot_pk;

		//Parallelised Vector update calculations
		#pragma omp parallel num_threads(NUM_THREADS)
		{ 	
			#pragma omp for schedule(static,chunk)
			for(int j=0;j<num_cols;j++) 
			{
				x_k[j] = x_k[j] + (alpha * p_k[j]);
				r_k[j] = r_k[j] - (alpha * Ap_k[j]);
			}
		}

		//Dot product of rk with rk
		rk_dot_rk_new = vectorDotProduct(r_k,r_k,dim);
		 
		//Calculate residue to search for loop convergence		
		residue = pow(rk_dot_rk_new,0.5);
		
		beta = rk_dot_rk_new/rk_dot_rk_prev;
		
		//Store the dot product value; To be used for the subsequent loop
		rk_dot_rk_prev = rk_dot_rk_new;	


		//Parallelised vector update loop
		#pragma omp parallel num_threads(NUM_THREADS)
		{ 	
			#pragma omp for schedule(static,chunk)
			for(int j=0;j<num_cols;j++)
			{
				p_k[j] = r_k[j] + beta*p_k[j];
			}
		}

		relative_residue = (residue/residue_init);

		//Debug prints
		fprintf(stdout,"\nIteration No. = %d\n", numIterations);	
		fprintf(stdout,"Initial Residue = %e\t Current Residue = %e\t Relative Residue = %e\n",residue_init, residue, relative_residue);
		fprintf(optimization,"\nIteration No. = %d\tRelative residue = %e\n", numIterations, relative_residue);

	}

	fprintf(stdout,"\nNumber of Iterations = %d\n", numIterations);
	fprintf(stdout,"\nInitial Residue = %e\t Final Residue = %e\t Final Relative Residue = %e\n",residue_init, residue, relative_residue);	

	fprintf(optimization,"\nNumber of Iterations = %d\n", numIterations);
	fprintf(optimization,"\nInitial Residue = %e\t Final Residue = %e\t Final Relative Residue = %e\n",residue_init, residue, relative_residue);

	//freeing all the allocated memories to prevent memory leak
	free(r_k);
	free(p_k);
	free(Ap_k);
	
	
	fclose(optimization);

	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////FUNCTION         :       vectorDotProduct//////////////////////////////////////////////////////////////////////////
//////////////INPUTs           :                        //////////////////////////////////////////////////////////////////////////
//////////////input1           :       Input vector 1   //////////////////////////////////////////////////////////////////////////
//////////////input2           :       Input vector 2   //////////////////////////////////////////////////////////////////////////
///////////// n                :       dimension        //////////////////////////////////////////////////////////////////////////
///////////// OUTPUTS                                   //////////////////////////////////////////////////////////////////////////
///////////// dot              :       Output dot product ////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double vectorDotProduct(double* input1, double* input2, int n){
	double dot=0.0, *a,*b;
    int N,i, nthreads, tid;

    #pragma omp parallel num_threads(NUM_THREADS) reduction (+: dot) \
      shared (n,input1,input2) private (N,i, nthreads, tid, a,b)
    {
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();

        N = n/nthreads; // Min iter for all threads
        a = input1 + N*tid; // Ptrs to this threads
        b = input2 + N*tid; // chunks of X & Y

        if ( tid == nthreads-1 )
            N += n-N*nthreads;

        dot = a[0]*b[0];
        for (i=1; i<N; i++)
            dot += a[i]*b[i];
    }
    return dot;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////FUNCTION         :       matrixVectorProduct////////////////////////////////////////////////////////////////////////
//////////////INPUTs           :                        //////////////////////////////////////////////////////////////////////////
//////////////MAT              :       Input Matrix     //////////////////////////////////////////////////////////////////////////
//////////////input2           :       Input vector     //////////////////////////////////////////////////////////////////////////
///////////// output           :       Output of the multiplication //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void matrixVectorProduct(double* output, MTX *MAT, double* input){

	int i,j, ckey;
	int chunk = 1000;
		
	//Check if the input matrix is symmetric
	if((matcode[1] == 'X')&&(matcode[3] == 'S')) 
	{

		// Parallelising the multiplication and accumulation
		#pragma omp parallel num_threads(NUM_THREADS)
		{
			#pragma omp for private(ckey,j,i) schedule(static,1000)
			for(i=0;i<MAT->nrows;i++) {
				output[i] = 0.0;
				for(ckey=MAT->row_ptr[i];ckey<MAT->row_ptr[i+1];ckey++) {
					j = MAT->JA[ckey];
					//zi = zi + MAT->val[ckey] * input[j];
					output[i] = output[i] + MAT->val[ckey] * input[j];
				}
			//output[i] += zi;
			}
		}

		
		for(int i=0;i<MAT->nrows;i++) 
			for(int ckey=MAT->row_ptr[i];ckey<MAT->row_ptr[i+1], (MAT->JA[ckey]) != i;ckey++) {
				j = MAT->JA[ckey];
				output[j] += MAT->val[ckey] * input[i];;
			}
	}
	else
	{
		fprintf(stderr,"\n Not a symmetric Matrix. CG method not applicable\n");
		exit(1);
	}
		
return;
}


