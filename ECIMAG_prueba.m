
% Reducir el ruido de la imagen para facilitar la segmentación posterior
%I = medfilt2(I, [2 2]);
%I = stdfilt(I, [1 0 1; 1 0 1; 1 0 1]);
%figure, imshow(I, [min(I(:)) max(I(:))]);
%title('Cropped image without leaves - Filtered');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

% Abrir la imagen
originalImage = imread('C:\Users\Usuario\Dropbox\RootSegmentation\Images\SKMBT_C36414021715140_0014.jpg');

% Croppear la región donde está la raíz
figure
cropped = imcrop(originalImage);
cropped = uint8(((double(cropped) - double(min(cropped(:)))) / (double(max(cropped(:))) - double(min(cropped(:))))) * 255);
close all
figure, imshow(cropped, [min(cropped(:)) max(cropped(:))]);
title('Cropped image');

% Eliminar las hojas
withoutRoot = imerode(cropped, strel('square',3));
leaves = imdilate(withoutRoot, strel('square',3));
figure, imshow(leaves, [min(leaves(:)) max(leaves(:))]);
title('Leaves');
I = cropped - leaves;
figure, imshow(I, [min(I(:)) max(I(:))]);
title('Cropped image without leaves');

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

% Reducir el ruido de la imagen para facilitar la segmentación posterior
I = stdfilt(cropped, [0 1 0; 1 0 0; 0 0 0]);
figure, imshow(cropped, [min(cropped(:)) max(cropped(:))]);
title('Cropped image without leaves - Filtered');

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
segmentation = fullyCRF_wrapped(U, P, [3], 100000);
figure, imshow(segmentation, [min(segmentation(:)) max(segmentation(:))]);
title('Segmentation');

% Unir las regiones desconexas
changes = 1;
refinedSegmentation = logical(segmentation);
while (changes ~= 0)
    previous = refinedSegmentation;
    refinedSegmentation = bwmorph(previous, 'bridge');
    changes = sum(sum(abs(double(refinedSegmentation) - double(previous))));
end
figure, imshow(refinedSegmentation);
title('Refined segmentation - Bridge');

% Esqueletización
skeletonization = bwmorph(refinedSegmentation, 'skel', 4);
figure, imshow(skeletonization);
title('Skeletonization');

% Limpiar las regiones con area menor a algo
skeletonization = bwareaopen(logical(skeletonization), 20);
figure, imshow(skeletonization);
title('Refined skeletonization - Removed isolated regions');

% Representación como grafo
nodes = skeletonization;
edges = cell(size(skeletonization));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear; clc; close all;
% 
% % Abrir la imagen
% originalImage = imread('C:\Users\Usuario\Dropbox\RootSegmentation\Images\SKMBT_C36414021715140_0014.jpg');
% 
% % Croppear la región donde está la raíz
% figure
% cropped = imcrop(originalImage);
% cropped = uint8(((double(cropped) - double(min(cropped(:)))) / (double(max(cropped(:))) - double(min(cropped(:))))) * 255);
% close all
% figure, imshow(cropped, [min(cropped(:)) max(cropped(:))]);
% title('Cropped image');
% 
% % Eliminar las hojas
% withoutRoot = imerode(cropped, strel('square',3));
% leaves = imdilate(withoutRoot, strel('square',3));
% figure, imshow(leaves, [min(leaves(:)) max(leaves(:))]);
% title('Leaves');
% I = cropped - leaves;
% figure, imshow(I, [min(I(:)) max(I(:))]);
% title('Cropped image without leaves');
% 
% % Realzo las raíces
% riccis = zeros(size(cropped, 1), size(cropped, 2), 6);
% count = 1;
% for i = 3 : 2 : 15 
%     [ricci] = Ricci2007(imcomplement(double(cropped)), i, i);
%     riccis(:,:,count) = ricci(:,:,1);
%     count = count + 1;
% end
% cropped = max(riccis, [], 3);
% figure, imshow(cropped, [min(cropped(:)) max(cropped(:))]);
% title('Enhanced with Ricci line detector');
% 
% % Generar los potenciales unarios
% connected = cropped;
% connected = double(connected);
% connected = (connected / max(connected(:))) * 6;
% U = zeros(size(connected,1), size(connected,2), 2);
% U(:,:,1) = connected;
% U(:,:,2) = imcomplement(connected);
% 
% % Generar los potenciales pairwise
% P = connected;
% 
% % Obtener la segmentación
% segmentation = fullyCRF_wrapped(U, P, [3], 100000);
% figure, imshow(segmentation, [min(segmentation(:)) max(segmentation(:))]);
% title('Segmentation');
% 
% % Unir las regiones desconexas
% changes = 1;
% refinedSegmentation = logical(segmentation);
% while (changes ~= 0)
%     previous = refinedSegmentation;
%     refinedSegmentation = bwmorph(previous, 'bridge');
%     changes = sum(sum(abs(double(refinedSegmentation) - double(previous))));
% end
% figure, imshow(refinedSegmentation);
% title('Refined segmentation - Bridge');
% 
% % Esqueletización
% skeletonization = bwmorph(refinedSegmentation, 'skel', 4);
% figure, imshow(skeletonization);
% title('Skeletonization');
% 
% % Limpiar las regiones con area menor a algo
% skeletonization = bwareaopen(logical(skeletonization), 20);
% figure, imshow(skeletonization);
% title('Refined skeletonization - Removed isolated regions');
% 
% % Representación como grafo
% nodes = skeletonization;
% edges = cell(size(skeletonization));


