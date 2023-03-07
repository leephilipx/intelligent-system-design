# Things we can try out

Important notes:
- Please remember to save your model weights (applies to all parts)!
- Do we need any normalisation prior to classifier?
- FDDB: http://vis-www.cs.umass.edu/fddb/
- Fisherface paper: https://cseweb.ucsd.edu/classes/wi14/cse152-a/fisherface-pami97.pdf

Face detection:
- [x] Haar cascade basic code
- [x] DL-based face det basic code
- [ ] Benchmark haar cascade parameters on FDDB
- [ ] Benchmark SSD on FDDB
- [ ] Provide inference times for both methods (on batch_size=1)

Dimensionality reduction:
- [x] PCA on Olivetti faces
- [ ] PCA on LFW people (see comment in jupyter notebook)
- [ ] PCA without first 3 components (ref paper)
- [ ] Fisherface (ref paper)

Face recognition:
- [x] Use deep learning classifier
- [ ] AdaBoost classifier
- [ ] SVM classifier, others?
- [ ] Fit all models onto own dataset

Integration:
- [ ] Asynchorous VideoCapture and OpenCV window
- [ ] Axial rotation along face symmetry line (needs eye keypoints)
- [ ] Integrate all components with labels
- [ ] Add keyboard toggle functionality