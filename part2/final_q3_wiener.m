img = imread('viz_outputs/tavg_20/172-191.bmp');

PSF = fspecial('average', 5);
% PSF = fspecial('disk', 3);
% PSF = fspecial('gaussian', 5, 1);
NSR = 0.3;
wnr_img = deconvwnr(img, PSF, NSR);

canny_20 = edge(wnr_img, 'canny', [0.05 0.15], 2);
canny_15 = edge(wnr_img, 'canny', [0.05 0.15], 1.5);

figure(2);
subplot(1, 3, 1); imshow(wnr_img);
title('After Wiener Deconvolution');
subplot(1, 3, 2); imshow(canny_20);
title('Canny with [0.05, 0.15], sigma=2');
subplot(1, 3, 3); imshow(canny_15);
title('Canny with [0.05, 0.15], sigma=1.5');

mkdir('viz_outputs\tavg_20_matlab')
save('viz_outputs\tavg_20_matlab\canny_wiener_avg5.mat', 'wnr_img', 'canny_20', 'canny_15');