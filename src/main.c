#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mmio.h"

int main(){

	int matrixChoice;
	int numColumns;
	char* matrix;

	double start_time, run_time;
	
	// File Handler for the result vector
	FILE *result;

	// Print statements displaying the different matrices used for testing process
	// Using the matrices from the University of Florida Sparse matrix collection 
	printf("Please enter the number for the input matrix: \n");
	printf("1. cvxbqp1.mtx \n");
	printf("2. cant.mtx \n");
	printf("3. apache1.mtx \n");
	printf("4. G2_circuit.mtx \n");
	printf("5. ecology2.mtx \n");
	printf("6. BenElechi1.mtx \n");
	printf("7. denormal.mtx \n");
	printf("8. tmt_sym.mtx \n");
	printf("9. s3rmt3m3.mtx \n");
	printf("10. thermomech_TK.mtx \n");
	printf("11. G3_circuit.mtx \n");

	scanf("%d",&matrixChoice);
	
	switch(matrixChoice) {
		case 1: 
			matrix = "cvxbqp1.mtx";
			break;
			
		case 2: 
			matrix = "cant.mtx";
			break;
			
		case 3: 
			matrix = "apache1.mtx";
			break;

		case 4: 
			matrix = "G2_circuit.mtx";
			break;

		case 5: 
			matrix = "ecology2.mtx";
			break;

		case 6: 
			matrix = "BenElechi1.mtx";
			break;

		case 7: 
			matrix = "denormal.mtx";
			break;

		case 8: 
			matrix = "tmt_sym.mtx";
			break;

		case 9: 
			matrix = "s3rmt3m3.mtx";
			break;

		case 10: 
			matrix = "thermomech_TK.mtx";
			break;

		case 11: 
			matrix = "G3_circuit.mtx";
			break;
			
		default:
			matrix = "s3rmt3m3.mtx";
			break;		
	}

	printf("Matrix %d selected",matrixChoice);

	//Declaring input and Output paths
        char inMatrix[100];
	char outMatrix[100];
	strcpy(inMatrix,"./Matrices/");
        strcpy(outMatrix,"./Output/");

	// Input Matrix 
	char* inputMatrix = strcat(inMatrix,matrix);
	
	// Result matrix
	char ResultMatrix[100];
	strcpy(ResultMatrix,"result.out");

	// Declare Matrix Structuref
	MTX MAT;

	// Read Matrix from MatrixMarket format
	Read_Mat(inputMatrix, &MAT);

	// Storing the dimension of the input matrix array
	numColumns = MAT.ncols;

	// Convert into CSR format.
	convertToCSR(&MAT);

	// Allocating Memories for the arrays 
	double* x = (double*) malloc(numColumns*sizeof(double));
	double* b = (double*) malloc(numColumns*sizeof(double)); 
	double* x_k = (double*) malloc(numColumns*sizeof(double)); 
	double* init_guess_x0 = (double*) malloc(numColumns*sizeof(double)); 

	//Initialising the solution vector and the initial guess
	// Initial guess is taken as 0.0
	// Solution vector is initialised to 1.0 to calculate the RHS vector b
	for(int i=0;i<numColumns;i++) {
		x[i] = 1.0;
		init_guess_x0[i] = 0.0;
	}	
	
	// Calculating the RHS vector b using the sample solution vector
	matrixVectorProduct(b, &MAT, x);	

	//Initialising the residue to calculate the initial relative residue during the conjugate gradient step
	double residue = 1;

	//Maximum  value for tolerance; Calculated after iteratively running the CG step to arrive at a suitable
	// value such that the final output vector is correct
	double tolerance = 1e-8;
	
	//Maximum iteration allowed before convergence is assumed to fail for a given matrix
	int max_iterations = 100000;


	//Calculating the total time consumed by the Conjugate gradient step	
	start_time = omp_get_wtime();
	ConjugateGradient(&MAT, numColumns, x_k, b, init_guess_x0, tolerance, max_iterations);
	run_time = omp_get_wtime() - start_time;
	fprintf(stdout,"Time for computation = %lf s\n",run_time);

	// Printing out the result vector v
	result = fopen(ResultMatrix,"w");

	for(int j=0;j<numColumns;j++){
		fprintf(result,"%1.16e\n",x_k[j]);
	}
	

	fclose(result);

	//freeing the allocated memories
	free(x);
	free(x_k);
	free(init_guess_x0);
	free(b);
return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////MATRIX PROCESSING FUNCTIONS/////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* 
      Functions are based on Matrix Market I/O examples.
*/ 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////FUNCTION         :       Read_Mat           ////////////////////////////////////////////////////////////////////////
//////////////INPUTs           :                        //////////////////////////////////////////////////////////////////////////
//////////////MAT              :       Input Matrix     //////////////////////////////////////////////////////////////////////////
//////////////file             :       Input Matrix file /////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Read_Mat(char* file, MTX *MAT){
    int ret_code;

    FILE *f;   

    if ((f = fopen(file, "r")) == NULL) 
        exit(1);

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &MAT->nrows, &MAT->ncols, &MAT->nz)) !=0)
        exit(1);


    /* reseve memory for matrices */

    MAT->IA = (int *) malloc(MAT->nz * sizeof(int));
    MAT->JA = (int *) malloc(MAT->nz * sizeof(int));
    MAT->val = (double *) malloc(MAT->nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for(int i=0; i<MAT->nz; i++)
    {
        if(fscanf(f, "%d %d %lg\n", &MAT->IA[i], &MAT->JA[i], &MAT->val[i])==0)
		fprintf(stderr,"\nFailed to read input Matrix\n");
        MAT->IA[i]--;  /* adjust from 1-based to 0-based */
        MAT->JA[i]--;
    }

return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////FUNCTION         :       convertToCSR       ////////////////////////////////////////////////////////////////////////
//////////////INPUTs           :                        //////////////////////////////////////////////////////////////////////////
//////////////MAT              :       Input Matrix     //////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void convertToCSR(MTX *MAT){

	sortCSR(MAT);

	int init=0;	
	
	MAT->n_row_ptr = MAT->nrows+1;
	MAT->row_ptr = (int *) malloc(MAT->n_row_ptr * sizeof(int));
	
	MAT->row_ptr[0] = 0;


	for(int j=1;j<MAT->n_row_ptr-1;j++){
		for(int i=init;i<MAT->nz;i++){
			if(MAT->IA[i+1] != MAT->IA[i]){
				MAT->row_ptr[j] = i+1;
				init = i+1;
				break;
			}
		}
	}

	MAT->row_ptr[MAT->n_row_ptr-1] = MAT->nz;
	
	matcode[1] = 'X';
	strcpy(crd,"CSR");	
return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////FUNCTION         :       sortCSR          ////////////////////////////////////////////////////////////////////////
//////////////INPUTs           :                        //////////////////////////////////////////////////////////////////////////
//////////////MAT              :       Input Matrix     //////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void sortCSR(MTX *MAT){
   
	//Flag to determine if swapping is done
	int flag;

	// Temporaries to swap the row index values
	int tmp_1;
	
	// Temporaries to swap the column index values
	int tmp_2;

	// Temporary to swap the value
	double tmp_3;

	//Break if no more swaps are required and matrix is sorted
	do{
		flag = 0;
		//iterate through all the row elements in the matrix
		for(int i=0;i<MAT->nz-1;i++){
			if((MAT->IA[i+1])<(MAT->IA[i])){

				//Swapping the row index value
				tmp_1 = MAT->IA[i];
				MAT->IA[i] = MAT->IA[i+1];
				MAT->IA[i+1] = tmp_1;

				//Swapping the column index value
				tmp_2 = MAT->JA[i];
				MAT->JA[i] = MAT->JA[i+1];
				MAT->JA[i+1] = tmp_2;

				//Swapping the actual value of the matrix in the Val array of the CSR
				tmp_3 = MAT->val[i];
				MAT->val[i] = MAT->val[i+1];
				MAT->val[i+1] = tmp_3;
				flag = 1;
			}
		}
    	}while(flag!=0);

return;
}

