
function y = fullyCRF_wrapped(unaryPotentials, pairwiseFeatures, weights, thetaPosition)
    
    % Get the segmentation
    y = fullyCRF(int32(size(unaryPotentials, 1)), int32(size(unaryPotentials, 2)), ...
        single(unaryPotentials), single(pairwiseFeatures), ...
        single(weights), single(thetaPosition));
    
    % Remove fake detections outside the field of view
    y = double(y);

end