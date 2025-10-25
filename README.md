# SIADS 699 - MADS Capstone
## Financial Form Text Extractor

### Overview
We are attempting to build a state of the art, full stack, text extraction application that performs on scanned images of forms. Our application's architecture will include manually labelled (Label Studio) training images (PNG, JPEG, PDFs) with bounded boxes of "header", "body" and "footer" text. This data will be used to fine tune a convolutional neural network (Yolov8). Our application will then extract text only from the body of forms (Tesseract 5). This extracted data will be stored in a relational database (postgreSQL). Our application will ultimately follow a micro-services pattern, implemented with Docker, and tied together in a web application front-end (Streamlit).

### Setup
```{bash}
.
├── data
│   ├── input
│   │   ├── ground-truth
│   │   └── training
│   └── output
├── docs
└── src
    ├── docker
    └── models
```

### Data
    - [Google Drive](https://drive.google.com/drive/folders/1ibqk_GzowWrwybOqg8wA88Q95gKQnrM1?usp=share_link)

### Documentation
    - [rvl-cdip-invoice](https://huggingface.co/datasets/chainyo/rvl-cdip-invoice)
    - [HuggingFace Co-Lab](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/save_load_dataset.ipynb#scrollTo=091FrwQDXQiM)
    - [Yolov8](https://arxiv.org/html/2408.15857)
    - [Tesseract 5](https://tesseract-ocr.github.io/tessdoc/)