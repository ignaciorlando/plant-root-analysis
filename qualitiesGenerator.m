root = 'C:\Users\Usuario\Dropbox\RootSegmentation\Images\';
skel = 'C:\Users\Usuario\Dropbox\RootSegmentation\ECImag\results_theta\';

imageName = 'SKMBT_C36414021715140_00';
firstImage = 14;
lastImage = 27;

%groundtruthmasks = cell(lastImage - firstImage);
%skeletonizations = cell(lastImage - firstImage);

qualities = zeros(3, lastImage - firstImage);

for i = firstImage : lastImage
    
    %leo la máscara del gtruth
    Igt = imread(strcat(root, imageName, num2str(i), '_gtmask1', '.png'));
    %figure, imshow(Igt);
    %title('Igt');
    for j = 1:3
        
        %leo la máscara del sector de la raíz a evaluar
        mask = imread(strcat(root, imageName, num2str(i), '_mask',num2str(j), '.png'));
%         figure, imshow(mask);
%         title('mask');
        [i_min, i_max, j_min, j_max] = getBoundingBox(mask);
        
        %cropeo el pedazo de imagen que me interesa
        cropped = Igt(i_min : i_max, j_min : j_max);
%         figure, imshow(cropped);
%         title('Cropped');
        %leo la esqueletización
        Iskel = imread(strcat(skel, imageName, num2str(i), '\', num2str(j),'\','skeletonization','.png'));
%         figure, imshow(Iskel);
%         title('Iskel');
        %calculo la intersección entre la esqueletización y el ground truth
%         
        i
        sizeIntersec
        intersec = Iskel .* cropped;
        %cuento en cuántos pixeles coinciden
        sizeIntersec = sum(intersec(:));
        %calculo |esqueletización|
        sizeRoot = sum(Iskel(:));
        %calculo el hit ratio
        qualities(j,i - firstImage + 1) = sizeIntersec / sizeRoot;
    end
    
end

qualities = qualities';