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
    #define SEGMENTATION_OUT    plhs[0]
    #define P1_OUT              plhs[1]
    #define P2_OUT              plhs[2]
    // For the inputs
    #define IMAGE_IN            prhs[0]
    #define UNARYPOTENTIALS_IN  prhs[1]
    #define N_IN                prhs[2]
    #define M_IN                prhs[3]
    #define NUMCLASSES_IN       prhs[4]
    #define W1_IN               prhs[5]
    #define W2_IN               prhs[6]  
    #define THETAALPHA_IN      prhs[7]
    #define THETABETA_IN       prhs[8] 
    #define THETAGAMMA_IN      prhs[9] 
    
    // Check the input parameters
    if (nrhs < 1 || nrhs > 10)
        mexErrMsgTxt("Wrong number of input parameters.");
    
    // Declare the variables to be used
    float * unaryPotentialsIn;
    int N;
    int M;
    unsigned char * imageIn;
    int numClasses;
    float w1;
    float w2;
    float thetaAlpha;
    float thetaBeta;
    float thetaGamma;
    const mwSize *dims;
    
    // Get the image 
    imageIn = (unsigned char *) mxGetData(IMAGE_IN);
    
    // Get the scores and its dimensions
    unaryPotentialsIn = (float *) mxGetData(UNARYPOTENTIALS_IN);
    dims = mxGetDimensions(UNARYPOTENTIALS_IN);

    // Get the size of the matrix
    N = (int) mxGetScalar(N_IN);
    M = (int) mxGetScalar(M_IN);
      
    // Get the number of classes
    numClasses = (int) mxGetScalar(NUMCLASSES_IN);
    
    // Get the weights and the parameters for the pairwise potentials
    w1 = (float) mxGetScalar(W1_IN);
    w2 = (float) mxGetScalar(W2_IN);
    thetaAlpha = (float) mxGetScalar(THETAALPHA_IN);
    thetaBeta = (float) mxGetScalar(THETABETA_IN);     
    thetaGamma = (float) mxGetScalar(THETAGAMMA_IN);
    
    // Setup the CRF model
	DenseCRF2D crf(N, M, numClasses);

    // Get the scores in the correct order
    float * unary = encodeUnaryPotentials(unaryPotentialsIn, N, M, numClasses);
    
    // Specify the unary potential as an array of size M*N*(#classes)
	// packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
	crf.setUnaryEnergy( unary );

    // Add a color dependent term (feature = xyrgb)
    unsigned char * im = imageIn;
	crf.addPairwiseBilateral(thetaAlpha, thetaAlpha, thetaBeta, thetaBeta, thetaBeta, im, w1);
    
	// Add a color independent term (feature = pixel location 0..N-1, 0..M-1)
	crf.addPairwiseGaussian(thetaGamma, thetaGamma, w2);
    
    // Create the output matrix for the segmentation
    SEGMENTATION_OUT = mxCreateNumericMatrix(N, M, mxINT16_CLASS, mxREAL);
    short * map = (short *) mxGetData(SEGMENTATION_OUT);
    // Do the inference
	crf.map(10, map);
    
    P1_OUT = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL);
    float * p1out = (float *) mxGetData(P1_OUT);
    P2_OUT = mxCreateNumericMatrix(N, M, mxSINGLE_CLASS, mxREAL);
    float * p2out = (float *) mxGetData(P2_OUT);
    crf.pairwiseEnergy(map, p1out, 0);
    crf.pairwiseEnergy(map, p2out, 1);
    
    return;
    
}
