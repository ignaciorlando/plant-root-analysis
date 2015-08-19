
root = 'C:\Users\Nacho\Dropbox\RootSegmentation\Images\';
imageName = 'SKMBT_C36414021715140_00';
firstImage = 19;
lastImage = 19;

for i = firstImage : lastImage
    
    I = imread(strcat(root, imageName, num2str(i), '.jpg'));
    
    for j = 1 : 1
        figure, imshow(I);
        h = imrect(gca);
        bw = h.createMask;
        imwrite(bw, strcat(root, imageName, num2str(i), '_mask', num2str(j), '.png'));
        close all;
    end
    
    
end