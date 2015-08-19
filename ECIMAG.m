clear; clc; close all;

root = 'G:\Dropbox\RootSegmentation\Images\';
savePath = 'G:\Dropbox\RootSegmentation\ECImag\results_theta\';
imageName = 'SKMBT_C36414021715140_00';
firstImage = 14;
lastImage = 27;

for i = firstImage : lastImage
    
    % Abrir la imagen
    originalImage = imread(strcat(root, imageName, num2str(i), '.jpg'));
    
    % Crear directorio
    imageResultDirectory = strcat(savePath, imageName, num2str(i), '\');
    mkdir(imageResultDirectory);
    
    % Por cada conjunto de raices
    for j = 1 : 3
 
        close all;
        
        % Crear path para guardar los resultados
        saveroot = strcat(imageResultDirectory, num2str(j), '\');
        mkdir(saveroot);
        
        % Abrir la máscara
        mask = logical(imread(strcat(root, imageName, num2str(i), '_mask', num2str(j), '.png')));
        
        % Enmascarar la imagen
        [i_min, i_max, j_min, j_max] = getBoundingBox(mask);
        cropped = originalImage(i_min : i_max, j_min : j_max);
        figure, imshow(cropped, [min(cropped(:)) max(cropped(:))]);
        title('Cropped image');
        imwrite(cropped,strcat(saveroot,'cropped.png'));
        
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
        for k = 3 : 2 : 15 
            [ricci] = Ricci2007(imcomplement(double(cropped)), k, k);
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
        connected = (connected / max(connected(:)));
        U = zeros(size(connected,1), size(connected,2), 2);
        w_u = 2;
        U(:,:,1) = connected * w_u;
        U(:,:,2) = imcomplement(connected * w_u);

        % Generar los potenciales pairwise
        P = connected;

        % Estimar theta_x
        maxNumberPairwises = (size(P,1) * size(P,2));
        numMedians = 10;
        if (numMedians * 10000) > maxNumberPairwises
            numMedians = floor(maxNumberPairwises / 10000);
            if numMedians == 0
                theta_x = abs((median(pdist(P(:)))));
            else
                medians = zeros(numMedians,1);
                for k = 1 : numMedians
                    medians(k) = abs((median(pdist(randsample(P(:), 10000)))));
                end
                theta_x = median(medians);
            end
        else
            medians = zeros(numMedians,1);
            for k = 1 : numMedians
                medians(k) = abs((median(pdist(randsample(P(:), 10000)))));
            end
            theta_x = median(medians);
        end
        
        % Obtener la segmentación
        theta_p = 1;
        w_p = 1;
        segmentation = fullyCRF_wrapped(U, P / theta_x, w_p, theta_p);
        figure, imshow(segmentation, [min(segmentation(:)) max(segmentation(:))]);
        title('Segmentation');
        imwrite(segmentation, strcat(saveroot,'segmentation.png'));

        % Unir las regiones desconexas
        changes = 1;
        refinedSegmentation = logical(segmentation);
        refinedSegmentation = imclose(refinedSegmentation, ones(3));
        figure, imshow(refinedSegmentation);
        title('Refined segmentation - Bridge');
        imwrite(refinedSegmentation, strcat(saveroot,'closedSegmentation.png'));

        % Esqueletización
        skeletonization = bwmorph(refinedSegmentation, 'skel', 8);
        figure, imshow(skeletonization);
        title('Skeletonization');
        imwrite(skeletonization, strcat(saveroot,'skeletonization.png'));

        % Limpiar las regiones con area menor a algo
        skeletonization = bwareaopen(logical(skeletonization), 20);
        figure, imshow(skeletonization);
        title('Refined skeletonization - Removed isolated regions');
        imwrite(skeletonization, strcat(saveroot,'finalSkeletonization.png'));
        
        
    end
    
    
end









