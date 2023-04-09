files = ls('viz_outputs\tavg_20\*.bmp');
mkdir('viz_outputs\tavg_20_matlab_edges')

NSR = 0.3;
PSF_avg = fspecial('average', 5);
PSF_dsk = fspecial('disk', 3);

for i = 1:length(files)

    filename_in = strcat('viz_outputs\tavg_20\', files(i, :));
    filename_out = strcat('viz_outputs\tavg_20_matlab_edges\', files(i, 1:end-4));

    img = imread(filename_in);
    wnr_avg = deconvwnr(img, PSF_avg, NSR);
    wnr_dsk = deconvwnr(img, PSF_dsk, NSR);
    
    canny_nml = edge(img, 'canny', [0.05 0.15], 1.5);
    canny_avg = edge(wnr_avg, 'canny', [0.05 0.15], 1.5);
    canny_dsk = edge(wnr_dsk, 'canny', [0.05 0.15], 1.5);
    
    save(filename_out, 'img', 'wnr_avg', 'wnr_dsk', 'canny_nml', 'canny_avg', 'canny_dsk');
    
end