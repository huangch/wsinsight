#!/usr/bin/env bash
set -euo pipefail

# Download all 12 Xenium breast biomarker datasets:
# - one *_outs.zip (Xenium output package)
# - one *_he_image.ome.tif (matching H&E image)

# S1 Top
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Top/Human_Breast_Biomarkers_S1_Top_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Top/Human_Breast_Biomarkers_S1_Top_he_image.ome.tif

# S1 Mid
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Mid/Human_Breast_Biomarkers_S1_Mid_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Mid/Human_Breast_Biomarkers_S1_Mid_he_image.ome.tif

# S1 Bot
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Bot/Human_Breast_Biomarkers_S1_Bot_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S1_Bot/Human_Breast_Biomarkers_S1_Bot_he_image.ome.tif

# S2 Top
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S2_Top/Human_Breast_Biomarkers_S2_Top_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S2_Top/Human_Breast_Biomarkers_S2_Top_he_image.ome.tif

# S2 Mid
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S2_Mid/Human_Breast_Biomarkers_S2_Mid_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S2_Mid/Human_Breast_Biomarkers_S2_Mid_he_image.ome.tif

# S2 Bot
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S2_Bot/Human_Breast_Biomarkers_S2_Bot_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S2_Bot/Human_Breast_Biomarkers_S2_Bot_he_image.ome.tif

# S3 Top
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S3_Top/Human_Breast_Biomarkers_S3_Top_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S3_Top/Human_Breast_Biomarkers_S3_Top_he_image.ome.tif

# S3 Mid
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S3_Mid/Human_Breast_Biomarkers_S3_Mid_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S3_Mid/Human_Breast_Biomarkers_S3_Mid_he_image.ome.tif

# S3 Bot
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S3_Bot/Human_Breast_Biomarkers_S3_Bot_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S3_Bot/Human_Breast_Biomarkers_S3_Bot_he_image.ome.tif

# S4 Top
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S4_Top/Human_Breast_Biomarkers_S4_Top_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S4_Top/Human_Breast_Biomarkers_S4_Top_he_image.ome.tif

# S4 Mid
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S4_Mid/Human_Breast_Biomarkers_S4_Mid_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S4_Mid/Human_Breast_Biomarkers_S4_Mid_he_image.ome.tif

# S4 Bot
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/4.0.0/Human_Breast_Biomarkers_S4_Bot/Human_Breast_Biomarkers_S4_Bot_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/4.0.0/Human_Breast_Biomarkers_S4_Bot/Human_Breast_Biomarkers_S4_Bot_he_image.ome.tif

# Xenium_V1_FFPE_Human_Breast_IDC_With_Addon
curl --retry 3 -O https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Breast_IDC_With_Addon/Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_outs.zip
curl --retry 3 -O https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_V1_FFPE_Human_Breast_IDC_With_Addon/Xenium_V1_FFPE_Human_Breast_IDC_With_Addon_he_image.ome.tif
