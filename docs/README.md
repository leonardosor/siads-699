# SIADS 699 - MADS Capstone
## Financial Form Text Extractor

### Overview
We are attempting to build a state of the art, full stack, text extraction application that performs on scanned images of forms. Our application's architecture will include manually labelled (Label Studio) training images (PNG, JPEG, PDFs) with bounded boxes of "header", "body" and "footer" text. This data will be used to fine tune a convolutional neural network (Yolov8). Our application will then extract text only from the body of forms (Tesseract 5). This extracted data will be stored in a relational database (postgreSQL). Our application will ultimately follow a micro-services pattern, implemented with Docker, and tied together in a web application front-end (Streamlit).

### Repository Architecture
```{bash}
.
├── .devcontainer
├── data
│   ├── input
│   │   ├── ground-truth
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   └── output
├── docs
└── src
    ├── docker
    └── models
```

### Data
- [Google Drive][1]

### Docker Container
The image build includes a PostgreSQL Alpine 15 database, Debian Frontend and Ultralytics (PyTorch, YOLO, OpenCV with GPU) support. As part of the PostgreSQL set-up we're including a database init file that creates a few tables with pre-set columns and data types as a general outline of MLOps data.

### Documentation
- [rvl-cdip-invoice][2]
- [HuggingFace Co-Lab][3]
- [Yolov8][4]
- [Tesseract 5][5]

[1]: https://drive.google.com/drive/folders/1ibqk_GzowWrwybOqg8wA88Q95gKQnrM1?usp=share_link
[2]: https://huggingface.co/datasets/chainyo/rvl-cdip-invoice
[3]: https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/save_load_dataset.ipynb#scrollTo=091FrwQDXQiM
[4]: https://arxiv.org/html/2408.15857
[5]: https://tesseract-ocr.github.io/tessdoc/
