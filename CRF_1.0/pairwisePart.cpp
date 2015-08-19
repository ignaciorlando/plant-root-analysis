#include "mex.h" 
#include <cstdio>
#include "densecrf.cpp"
#include <iostream>
#include <fstream>


// *****************************************
// Gateway routine
void mexFunction(int nlhs, mxArray * plhs[],    // output variables
                int nrhs, const mxArray * prhs[]) // input variables
{
    
    // Macros declarations 
    // For the outputs
    #define PAIRSWISEKERNELS_OUT   plhs[0]
    // For the inputs
    #define HEIGHT_IN              prhs[0]
    #define WIDTH_IN               prhs[1]
    #define LABELING_IN            prhs[2]
    #define PAIRWISEFEATURES_IN    prhs[3]
    #define NUMPAIRWISES_IN        prhs[4]
	#define HSTD_IN 			   prhs[5]
    #define WSTD_IN        		   prhs[6]
    #define THETA_P_X_IN        		   prhs[7]
    #define THETA_P_Y_IN        		   prhs[8]
    
    // Check the input parameters
    if (nrhs < 1 || nrhs > 9)
        mexErrMsgTxt("Wrong number of input parameters.");
    
    // Get the size of the image
    int height = (int) mxGetScalar(HEIGHT_IN);
    int width = (int) mxGetScalar(WIDTH_IN); 
    
    // Get the labeling
    short * labelingIn = (short *) mxGetData(LABELING_IN);
    
    // Get the pairwise features
    float * pairwiseFeaturesIn = (float *) mxGetData(PAIRWISEFEATURES_IN);
    
	// Get the standard deviations of the (x,y) coordinates
    float hstd_in = (float) mxGetScalar(HSTD_IN);
	float wstd_in = (float) mxGetScalar(WSTD_IN);
	
    // Get the theta_p
    int theta_p_x = (float) mxGetScalar(THETA_P_X_IN);
    int theta_p_y = (float) mxGetScalar(THETA_P_Y_IN);
    
    // Create the CRF to compute the pairwise potentials. height and width 
    // represents the size of the image, and 2 is the amount of classes
    // (background and vessels) 
    DenseCRF2D crf(height, width, 2);
    
    // Get the pairwise features
    int numPairwiseFeatures = (int) mxGetScalar(NUMPAIRWISES_IN);
    float * pairwiseFeatures = (float *) mxGetData(PAIRWISEFEATURES_IN);
    // Assign the pairwise features
    for (int pairw = 0; pairw < numPairwiseFeatures; pairw++) {
        // Encode the pairwise features
        float * single_PairwiseFeatures = new float[width * height * 3];
		float mean_h = height / 2;
		float mean_w = width / 2;
        for (int h = 0; h < height; h++) {    
            for (int w = 0; w < width; w++) {
                single_PairwiseFeatures[(w+h*width)*3+0] = pairwiseFeatures[h + height * (w + width * pairw)];
				single_PairwiseFeatures[(w+h*width)*3+1] = (((float) (h - mean_h)) / hstd_in) / theta_p_y; 
				single_PairwiseFeatures[(w+h*width)*3+2] = (((float) (w - mean_w)) / wstd_in) / theta_p_x;
            }
        }
        //Assign the pairwise features
        crf.addPairwiseEnergy(single_PairwiseFeatures, 3, 1.0);
        delete [] single_PairwiseFeatures;
    }
    
    // Declare the output variables
    mwSize dims[3] = {height, width, numPairwiseFeatures};
    PAIRSWISEKERNELS_OUT = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    float * pairwiseKernels = (float *) mxGetData(PAIRSWISEKERNELS_OUT);
    
    // Compute the pairwise energy
    for (int pairw = 0; pairw < numPairwiseFeatures; pairw++) {
        float * pairwiseEnergy = new float[width * height];
        crf.pairwiseEnergy(labelingIn, pairwiseEnergy, pairw);
        for (int w = 0; w < width; w++) {
            for (int h = 0; h < height; h++) {  
                pairwiseKernels[h + height * (w + width * pairw)] = (float) pairwiseEnergy[w + width * h];
            }
        }
        delete [] pairwiseEnergy;
    }
    
    return;
    
}