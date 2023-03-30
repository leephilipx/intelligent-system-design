img = imread('viz_outputs/retina2_ksize1_tavg30_172-201.bmp');

% PSF = fspecial('average', 5);
PSF = fspecial('disk', 3);
NSR = 0.5;
wnr_img = deconvwnr(img, PSF, NSR);

edge1 = edge(img, 'canny', [0.05 0.15], 2);
edge2 = edge(wnr_img, 'canny', [0.02 0.15], 2);
overlayed = img+uint8(edge2*255);
imwrite(overlayed, 'good_canny_disk.bmp');

subplot(1, 5, 1);
imshow(img);
title('Original Image');

subplot(1, 5, 2);
imshow(wnr_img);
title('Restored Image');

subplot(1, 5, 3);
imshow(edge1);
title('Canny (Original)');

subplot(1, 5, 4);
imshow(edge2);
title('Canny (Restored)');

subplot(1, 5, 5);
imshow(overlayed);
title('Overlayed Edges');