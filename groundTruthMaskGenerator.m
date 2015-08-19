
root = 'C:\Users\Mante\Dropbox\RootSegmentation\Images\';
imageName = 'SKMBT_C36414021715140_00';
firstImage = 14;
lastImage = 18;

for i = firstImage : lastImage
    
    %I = imread(strcat(root, imageName, num2str(i),'_gt2', '.png'));
    %I = imread(strcat(root, imageName, num2str(i),'_gt1', '.jpg'));
    
    % para encis 
    I = imread('C:\Users\Usuario\Dropbox\RootSegmentation\Images\SKMBT_C36414021715140_0018_gt1.png');
    Ir = I(:,:,1);
    Ig = I(:,:,2);
    Ib = I(:,:,3);
    
%   imshow(Ir);
%   imshow(Ig);
%   imshow(Ib);
    mask = Ir == 255 & Ig == 0 & Ib == 0;
    %mask = Ir ~= Ig & Ig ~= Ib & Ir ~= Ib & Ir > 250;
    %imshow(mask);
    
    %mask = imdilate(mask,strel('square',2));
    %mask = imfill(mask,'holes'); %PARA ENCIS
    %imwrite(mask, strcat(root, imageName, num2str(i), '_gtmask', '.png'));
    %imwrite(mask, strcat(root, imageName, num2str(i), '_gtmask1', '.png'));
    %para encis 
    imwrite(mask, 'C:\Users\Usuario\Dropbox\RootSegmentation\Images\SKMBT_C36414021715140_0018_gtmask1.png');

    close all;
    
end