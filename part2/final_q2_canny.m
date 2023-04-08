img = imread('viz_outputs/tavg_20/172-191.bmp');
canny_20 = edge(img, 'canny', [0.05 0.15], 2);
canny_15 = edge(img, 'canny', [0.05 0.15], 1.5);

figure(1);
subplot(1, 3, 1); imshow(img);
title('Image');
subplot(1, 3, 2); imshow(canny_20);
title('Canny with [0.05, 0.15], sigma=2');
subplot(1, 3, 3); imshow(canny_15);
title('Canny with [0.05, 0.15], sigma=1.5');

mkdir('viz_outputs\tavg_20_matlab')
save('viz_outputs\tavg_20_matlab\canny_only.mat', 'img', 'canny_20', 'canny_15');