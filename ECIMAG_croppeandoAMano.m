saveroot = 'C:\Users\Nacho\Dropbox\RootSegmentation\ECImag\pics\OTRO\';

% Abrir la imagen
originalImage = imread('C:\Users\Nacho\Dropbox\RootSegmentation\Images\SKMBT_C36414021715140_0014.jpg');

% Croppear la región donde está la raíz
figure
cropped = imcrop(originalImage);
cropped = uint8(((double(cropped) - double(min(cropped(:)))) / (double(max(cropped(:))) - double(min(cropped(:))))) * 255);
close all
figure, imshow(cropped, [min(cropped(:)) max(cropped(:))]);
title('Cropped image');
imwrite(uint8(cropped), strcat(saveroot,'cropped.png'));

% Eliminar las hojas
withoutRoot = imerode(cropped, strel('square',3));
leaves = imdilate(withoutRoot, strel('square',3));
figure, imshow(leaves, [min(leaves(:)) max(leaves(:))]);
title('Leaves');
imwrite(leaves, strcat(saveroot,'leaves.png'));
cropped = cropped - leaves;
figure, imshow(cropped, [min(cropped(:)) max(cropped(:))]);
title('Cropped image without leaves');
imwrite(cropped,strcat(saveroot,'withoutLeaves.png'));

% Realzo las raíces
riccis = zeros(size(cropped, 1), size(cropped, 2), 6);
count = 1;
for i = 3 : 2 : 15 
    [ricci] = Ricci2007(imcomplement(double(cropped)), i, i);
    riccis(:,:,count) = ricci(:,:,1);
    count = count + 1;
end
cropped = max(riccis, [], 3);
figure, imshow(cropped, [min(cropped(:)) max(cropped(:))]);
title('Enhanced with Ricci line detector');
ricci = uint8(((double(cropped) - double(min(cropped(:)))) / (double(max(cropped(:))) - double(min(cropped(:))))) * 255);
imwrite(ricci, strcat(saveroot,'ricci.png'));

% Generar los potenciales unarios
connected = cropped;
connected = double(connected);
connected = (connected / max(connected(:))) * 6;
U = zeros(size(connected,1), size(connected,2), 2);
U(:,:,1) = connected;
U(:,:,2) = imcomplement(connected);

% Generar los potenciales pairwise
P = connected;

% Obtener la segmentación
theta_p = 1;
segmentation = fullyCRF_wrapped(U, P / theta_p, [3], 100000);
figure, imshow(segmentation, [min(segmentation(:)) max(segmentation(:))]);
title('Segmentation');
imwrite(segmentation, strcat(saveroot,'segmentation.png'));

% Unir las regiones desconexas
changes = 1;
refinedSegmentation = logical(segmentation);
% while (changes ~= 0)
%     previous = refinedSegmentation;
%     refinedSegmentation = bwmorph(previous, 'bridge');
%     changes = sum(sum(abs(double(refinedSegmentation) - double(previous))));
% end
refinedSegmentation = imclose(refinedSegmentation, ones(3));
figure, imshow(refinedSegmentation);
title('Refined segmentation - Bridge');
imwrite(refinedSegmentation, strcat(saveroot,'closedSegmentation.png'));

% Esqueletización
skeletonization = bwmorph(refinedSegmentation, 'skel', 4);
figure, imshow(skeletonization);
title('Skeletonization');
imwrite(skeletonization, strcat(saveroot,'skeletonization.png'));

% Limpiar las regiones con area menor a algo
skeletonization = bwareaopen(logical(skeletonization), 20);
figure, imshow(skeletonization);
title('Refined skeletonization - Removed isolated regions');
imwrite(skeletonization, strcat(saveroot,'finalSkeletonization.png'));

% Representación como grafo
nodes = skeletonization;
edges = cell(size(skeletonization));