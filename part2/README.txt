
### OUTPUTS ###

All outputs are in the viz_outputs directory

tavg_20                         Temporal averaged images with window size 20
tavg_20_matlab                  Intermediate results for use in visualize_matlab.ipynb
tavg_20_matlab_edges            Images after Wiener deconvolution and Canny results from MATLAB
tavg_20_final_edges_only        Final edge image output
tavg_20_final_edges_overlayed   Final edge image overlayed onto tavg_20

XXX.gif                         GIF of images without shifting
XXX_shifted.gif                 GIF of images with shifting (w.r.t. tavg_20)


### CODE ###

final_qX.ipynb/m
- Contains codes and visualisation related to the question 1, 2, and 3 respectively (Sections 2, 3, 4 in report).

final_all_python/matlab
- Contains code for use in processing the entire retina2 dataset to final edges.

visualize_matlab.ipynb
- Visualize results obtained from MATLAB.
