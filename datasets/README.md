# DATASET

## Description

This dataset consists of 9 test WSI images taken from the [CAMELYON16](https://camelyon16.grand-challenge.org/) dataset. WSI involves digitally scanning a tissue slide containing thin tissue samples for microscopic examination and storing them as digital images. WSI is a technology used to digitally scan and archive images at high resolution. Since the images are high resolution, they will be processed in patches in this repo.

## Content

The dataset consists of 9 WSI images in the `wsi` folder. The images are named as `test_001.tif`, `test_002.tif`, ..., `test_009.tif`.

* `data_analysis.ipynb`: A notebook that visualizes the WSI images and their extracted thumbnails. The thumbnails are saved in the thumbnails folder.

`patch_embedding_extractor.ipynb`: A notebook that extracts embeddings from the patches of WSI images using a pre-trained [Google Path Foundation](https://research.google/blog/health-specific-embedding-tools-for-dermatology-and-pathology/) model. The embeddings are saved in the embeddings folder. Patch sizes are 224, 512, and 1024. Patches are resized to 224x224 before embedding extraction.

`clustering_dataset.ipynb`: A notebook that clusters the embeddings extracted from WSI patches using the KMeans algorithm. The clustering process helps clean sensor artifacts and background noise. The clustered embeddings are saved in the clustered_embeddings folder.

`selected_embeddings` folder: Contains the embeddings of patches selected from the clustered embeddings.

`merged_embeddings` folder: Contains merged embeddings of the selected patches.
