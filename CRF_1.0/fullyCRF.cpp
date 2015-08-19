#include "mex.h" 
#include "densecrf.h"
#include <cstdio>
#include <cmath>
#include "util.h"
#include <iostream>
#include <fstream>


float * encodeUnaryPotentials(float * input, int height, int width, int numClasses) {
    float * unaryPotentials = new float[height*width*numClasses];

    for (int w_ = 0; w_ < width; w_++) {
        for (int h_ = 0; h_ < height; h_++) {
            
            for (int i = 0; i < numClasses; i++) { 
                unaryPotentials[numClasses * (h_ + height * w_) + i] = input[h_ + height * (w_ + width * i)];
            }
            
        }
    }
    return unaryPotentials;
    
}



// *****************************************
// Gateway routine
void mexFunction(int nlhs, mxArray * plhs[],    // output variables
                int nrhs, const mxArray * prhs[]) // input variables
{
    
    // Macros declarations 
    // For the outputs
    #define SEGMENTATION_OUT       plhs[0]
    // For the inputs
    #define HEIGHT_IN              prhs[0]
    #define WIDTH_IN               prhs[1]
    #define UNARYPOTENTIALS_IN     prhs[2]
    #define PAIRWISEPOTENTIALS_IN  prhs[3]
    #define WEIGHT_IN              prhs[4]   
    #define THETA_POSITION         prhs[5]

    
    // Check the input parameters
    if (nrhs < 1 || nrhs > 6)
        mexErrMsgTxt("Wrong number of input parameters.");

    // Get the size of the matrix
    int height = (int) mxGetScalar(HEIGHT_IN);
    int width = (int) mxGetScalar(WIDTH_IN);
    
    // Get the theta_p
    float theta_Position = (float) mxGetScalar(THETA_POSITION);
   
    // Get the pairwise potentials
    float * pairwises = (float *) mxGetData(PAIRWISEPOTENTIALS_IN);
    
    // Get the weights and the parameters for the pairwise potentials
    float weight = (float) mxGetScalar(WEIGHT_IN);  
    
    // Setup the CRF model
	DenseCRF2D crf(height, width, 2);
    
    // Set the unary potential as an array of size width*height*(#classes)
	// packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
    float * unaryPotentialsIn = (float *) mxGetData(UNARYPOTENTIALS_IN);
    float * unary = encodeUnaryPotentials(unaryPotentialsIn, height, width, 2);
	crf.setUnaryEnergy(unary);
    
    // Assign the pairwise energies to the CRF

    // Encode the feature
    float * feature = new float[height*width*3];
    for (int w_=0; w_<width; w_++) {
        for (int h_=0; h_<height; h_++) {
            feature[(w_+h_*width)*3+0] = pairwises[(w_+h_*width)];
            feature[(w_+h_*width)*3+1] = h_ / theta_Position;
            feature[(w_+h_*width)*3+2] = w_ / theta_Position;
        }
    }
    //Assign the pairwise
    crf.addPairwiseEnergy(feature, 3, weight, NULL); 
    
    // Do the inference
    SEGMENTATION_OUT = mxCreateNumericMatrix(height, width, mxINT16_CLASS, mxREAL);
    short * map = (short *) mxGetData(SEGMENTATION_OUT);
	crf.map(10, map);
    
    delete [] unary;
    delete [] feature;
    //delete [] map;
    
    return;
    
}
