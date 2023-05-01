# Things we can try out

Important notes:
- Please remember to save your model weights (applies to all parts)!
- Do we need any normalisation prior to classifier?
- FDDB: http://vis-www.cs.umass.edu/fddb/
- Fisherface paper: https://cseweb.ucsd.edu/classes/wi14/cse152-a/fisherface-pami97.pdf

Face detection:
- [x] Haar cascade basic code
- [x] DL-based face det basic code
- [x] Benchmark haar cascade parameters on FDDB
- [x] Provide inference times for both methods (on batch_size=1)

Dimensionality reduction:
- [x] PCA on Olivetti faces
- [x] PCA on LFW people (see comment in jupyter notebook)
- [x] PCA without first 3 components (ref paper)
- [x] Fisherface (ref paper)

Face recognition:
- [x] Use deep learning classifier
- [x] AdaBoost classifier
- [x] SVM classifier, others?
- [x] Fit all models onto own dataset

Integration:
- [x] Asynchorous VideoCapture and OpenCV window
- [x] Axial rotation along face symmetry line (needs eye keypoints)
- [x] Integrate all components with labels
- [x] Add keyboard toggle functionality