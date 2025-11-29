\# Automated Memory Card Game with Computer Vision



An autonomous system that detects, classifies, and matches cards in a physical Memory game using computer vision techniques.



\## Features

\- \*\*Card Detection\*\*: Contour-based segmentation with morphological operations

\- \*\*Classification\*\*: SIFT feature matching to distinguish face-up vs face-down cards

\- \*\*Pair Matching\*\*: Feature-based comparison to identify matching pairs

\- \*\*Performance\*\*: 100% accuracy under controlled lighting conditions



\## System Architecture

1\. \*\*Image Preprocessing\*\*: Grayscale conversion → Gaussian blur → Morphological operations → Edge detection

2\. \*\*Card Segmentation\*\*: Contour detection and filtering based on area and shape

3\. \*\*Feature Extraction\*\*: SIFT keypoint detection and descriptor computation

4\. \*\*Classification\*\*: Feature matching against reference templates (20+ matches threshold)

5\. \*\*Game Logic\*\*: Automated turn-based gameplay simulation



