# Contextual Pill Recognition on VAIPE with Prescription Image and Medical Graph Context
# Nhận dạng thuốc theo ngữ cảnh trên VAIPE với ảnh prescription và ngữ cảnh đồ thị y khoa

## 1. Introduction
## 1. Giới thiệu

This project studies contextual pill recognition on the VAIPE dataset by following and improving the idea of the PIKA paper, Image based Contextual Pill Recognition with Medical Knowledge Graph Assistance.

Dự án này nghiên cứu bài toán nhận dạng thuốc theo ngữ cảnh trên bộ dữ liệu VAIPE, bám theo và cải tiến ý tưởng của bài báo PIKA, Image based Contextual Pill Recognition with Medical Knowledge Graph Assistance.

The main goal of the project is not only to classify pill crop images, but also to use prescription level context and graph based relational information to improve recognition performance on visually similar pill classes.

Mục tiêu chính của dự án không chỉ là phân loại ảnh crop viên thuốc, mà còn khai thác ngữ cảnh ở mức prescription và thông tin quan hệ dạng đồ thị để cải thiện khả năng nhận dạng các lớp thuốc có đặc trưng thị giác gần giống nhau.

In the final experiment, the best fine tuned model achieved a Validation Macro F1 score of 0.8671, which is higher than the best reported F1 score of 0.8571 in the PIKA paper.

Trong thực nghiệm cuối cùng, mô hình tốt nhất sau fine tune đạt Validation Macro F1 bằng 0.8671, cao hơn kết quả F1 tốt nhất là 0.8571 được công bố trong bài báo PIKA.

---

## 2. Motivation
## 2. Động lực nghiên cứu

Pill recognition is a difficult computer vision problem because many pills have very similar color, shape, and size. A model that only uses visual appearance often struggles when different classes look almost identical.

Nhận dạng thuốc là một bài toán khó trong thị giác máy tính vì nhiều viên thuốc có màu sắc, hình dạng và kích thước rất giống nhau. Mô hình chỉ dùng đặc trưng thị giác thường gặp khó khăn khi các lớp thuốc gần như giống hệt nhau.

In real medical settings, pills do not appear independently. They are usually associated with a prescription, and the co occurrence of medicines in the same prescription provides useful contextual information.

Trong bối cảnh y tế thực tế, các viên thuốc không xuất hiện độc lập. Chúng thường gắn với một đơn thuốc, và sự đồng xuất hiện của nhiều loại thuốc trong cùng một prescription mang lại thông tin ngữ cảnh rất hữu ích.

This project explores how to combine:
1. pill image features
2. prescription image features
3. graph based context derived from co occurring pill labels

Dự án này nghiên cứu cách kết hợp:
1. đặc trưng ảnh viên thuốc
2. đặc trưng ảnh prescription
3. ngữ cảnh đồ thị được xây dựng từ các nhãn thuốc đồng xuất hiện

---

## 3. Reference paper
## 3. Bài báo tham chiếu

Reference paper:
Tài liệu tham chiếu:

[Image based Contextual Pill Recognition with Medical Knowledge Graph Assistance](https://arxiv.org/abs/2208.02432)

The paper reports the best F1 score of 0.8571 with ResNet 18 plus PIKA.

Bài báo công bố kết quả tốt nhất đạt F1 bằng 0.8571 với backbone ResNet 18 kết hợp PIKA.

---

## 4. Dataset
## 4. Bộ dữ liệu

This project uses the VAIPE dataset.

Dự án sử dụng bộ dữ liệu VAIPE.

Dataset related link:
Liên kết liên quan đến dữ liệu:

[VAIPE Pill Resource](https://smarthealth.vinuni.edu.vn/resources/)

In the experiments of this project, the training data structure includes:
Trong các thực nghiệm của dự án, dữ liệu huấn luyện bao gồm:

- pill images
- pill labels in JSON format
- prescription images
- pill to prescription mapping through pill_pres_map.json

- ảnh pill
- nhãn pill dạng JSON
- ảnh prescription
- ánh xạ giữa pill và prescription thông qua file pill_pres_map.json

The raw dataset is processed into pill crops and metadata files for different model variants.

Dữ liệu gốc được xử lý thành các ảnh crop viên thuốc và các file metadata cho nhiều phiên bản mô hình khác nhau.

---

## 5. Project objective
## 5. Mục tiêu dự án

The project has three main objectives:
Dự án có ba mục tiêu chính:

1. Reproduce the contextual pill recognition setting on VAIPE
2. Compare multiple contextual modeling strategies
3. Build a final model that improves over the baseline and surpasses the best reported result of the PIKA paper

1. Tái hiện bài toán nhận dạng thuốc theo ngữ cảnh trên VAIPE
2. So sánh nhiều chiến lược mô hình hóa ngữ cảnh khác nhau
3. Xây dựng mô hình cuối cùng tốt hơn baseline và vượt kết quả tốt nhất được công bố trong bài báo PIKA

---

## 6. Model variants explored
## 6. Các phiên bản mô hình đã khảo sát

Several model variants were implemented and evaluated.

Nhiều phiên bản mô hình đã được xây dựng và đánh giá.

### 6.1 Baseline image only
### 6.1 Baseline chỉ dùng ảnh

This version uses only the cropped pill image for classification.

Phiên bản này chỉ sử dụng ảnh viên thuốc sau khi crop để phân loại.

### 6.2 PIKA like v1
### 6.2 PIKA like v1

This version combines:
- pill crop image
- prescription image

Phiên bản này kết hợp:
- ảnh crop viên thuốc
- ảnh prescription

This was the first strong contextual baseline and already improved a lot over image only classification.

Đây là baseline ngữ cảnh đầu tiên có hiệu quả rõ rệt và đã cải thiện đáng kể so với mô hình chỉ dùng ảnh viên thuốc.

### 6.3 PIKA v2
### 6.3 PIKA v2

This version uses:
- pill crop image
- structured context labels from the same prescription

Phiên bản này sử dụng:
- ảnh crop viên thuốc
- vector ngữ cảnh có cấu trúc từ các nhãn thuốc trong cùng prescription

### 6.4 PIKA v3
### 6.4 PIKA v3

This version combines:
- pill crop image
- prescription image
- structured context labels

Phiên bản này kết hợp:
- ảnh crop viên thuốc
- ảnh prescription
- vector ngữ cảnh có cấu trúc

### 6.5 PIKA graph v1
### 6.5 PIKA graph v1

This version introduces:
- pill crop image
- graph context built from co occurring labels in prescriptions

Phiên bản này đưa thêm:
- ảnh crop viên thuốc
- ngữ cảnh đồ thị được xây dựng từ các nhãn thuốc đồng xuất hiện trong prescription

### 6.6 Best PIKA model
### 6.6 Mô hình Best PIKA

The strongest version before fine tuning combines:
- pill crop image
- prescription image
- graph context built with PPMI
- gated fusion and interaction features

Phiên bản mạnh nhất trước fine tune kết hợp:
- ảnh crop viên thuốc
- ảnh prescription
- ngữ cảnh đồ thị được xây dựng bằng PPMI
- gated fusion và interaction features

### 6.7 Fine tuned best model
### 6.7 Mô hình tốt nhất sau fine tune

This version starts from the best checkpoint and fine tunes it with a smaller learning rate and adjusted class balancing strategy.

Phiên bản này bắt đầu từ checkpoint tốt nhất và fine tune với learning rate nhỏ hơn cùng chiến lược cân bằng lớp phù hợp hơn.

This is the final model of the project.

Đây là mô hình cuối cùng của dự án.

---

## 7. Final results
## 7. Kết quả cuối cùng

### 7.1 Comparison across model versions
### 7.1 So sánh giữa các phiên bản mô hình

| Model | Validation Macro F1 | Validation Accuracy |
|---|---:|---:|
| Baseline image only | 0.4801 | 0.5774 |
| PIKA like v1 | 0.6054 | 0.5959 |
| PIKA v2 | 0.5957 | 0.6051 |
| PIKA v3 | 0.5882 | 0.5505 |
| PIKA graph v1 | 0.5654 | 0.5843 |
| Best PIKA before fine tuning | 0.8380 | 0.6285 |
| Best PIKA after fine tuning | 0.8671 | 0.8903 |
| PIKA paper best reported result | 0.8571 | not reported in the same comparison setting |

| Mô hình | Validation Macro F1 | Validation Accuracy |
|---|---:|---:|
| Baseline chỉ dùng ảnh | 0.4801 | 0.5774 |
| PIKA like v1 | 0.6054 | 0.5959 |
| PIKA v2 | 0.5957 | 0.6051 |
| PIKA v3 | 0.5882 | 0.5505 |
| PIKA graph v1 | 0.5654 | 0.5843 |
| Best PIKA trước fine tune | 0.8380 | 0.6285 |
| Best PIKA sau fine tune | 0.8671 | 0.8903 |
| Kết quả tốt nhất của bài báo PIKA | 0.8571 | không công bố theo cùng thiết lập so sánh |

### 7.2 Main conclusion
### 7.2 Kết luận chính

The final fine tuned model reached:
Mô hình cuối cùng sau fine tune đạt:

- Validation Macro F1 = 0.8671
- Validation Accuracy = 0.8903

- Validation Macro F1 = 0.8671
- Validation Accuracy = 0.8903

This result is higher than the best reported F1 score of the reference PIKA paper.

Kết quả này cao hơn F1 tốt nhất được công bố trong bài báo PIKA tham chiếu.

---

## 8. Why the final model works better
## 8. Vì sao mô hình cuối cùng tốt hơn

The strongest model benefits from three complementary information sources:
Mô hình mạnh nhất hưởng lợi từ ba nguồn thông tin bổ sung cho nhau:

1. Visual detail from cropped pill images
2. Global prescription context from prescription images
3. Graph context from co occurring pill classes in prescriptions

1. Chi tiết thị giác từ ảnh crop viên thuốc
2. Ngữ cảnh toàn cục từ ảnh prescription
3. Ngữ cảnh đồ thị từ các lớp thuốc đồng xuất hiện trong prescription

In addition, the fine tuning stage helped correct weak performance on several large classes. One important improvement was class 105, which became a much stronger class after fine tuning and contributed significantly to the final score.

Ngoài ra, giai đoạn fine tune đã giúp sửa hiệu năng yếu ở một số lớp lớn. Một cải thiện rất quan trọng là lớp 105, lớp này mạnh lên rõ rệt sau fine tune và đóng góp đáng kể vào kết quả cuối cùng.

---


## 9. Repository structure
## 9. Cấu trúc repository

vaipe-contextual-pill-recognition/
├── README.md
├── requirements.txt
├── prepare_cropped_dataset.py
├── train_baseline.py
├── build_pika_metadata.py
├── train_pika_baseline.py
├── build_pika_context_metadata.py
├── train_pika_v2_context_labels.py
├── build_pika_v3_metadata.py
├── train_pika_v3_triple_context.py
├── build_pika_graph_data.py
├── train_pika_graph.py
├── build_best_pika_data.py
├── train_best_pika_model.py
├── finetune_best_pika_model.py

10. Main pipeline
10. Quy trình chính

The final best performing pipeline is:
Quy trình tốt nhất của mô hình cuối cùng gồm:

Download the VAIPE dataset
Build pill crops and metadata
Build a PPMI graph from prescription level co occurrence
Train the best PIKA model
Fine tune the best checkpoint
Tải bộ dữ liệu VAIPE
Tạo ảnh crop viên thuốc và metadata
Xây dựng đồ thị PPMI từ quan hệ đồng xuất hiện ở mức prescription
Huấn luyện mô hình Best PIKA
Fine tune từ checkpoint tốt nhất

The final model uses:
Mô hình cuối cùng sử dụng:

pill image encoder
prescription image encoder
graph encoder
gated fusion with interaction features
classifier head
bộ mã hóa ảnh viên thuốc
bộ mã hóa ảnh prescription
bộ mã hóa đồ thị
gated fusion với interaction features
tầng phân loại cuối
11. Environment
11. Môi trường thực hiện

The project was mainly developed and tested on Google Colab.

Dự án chủ yếu được phát triển và kiểm thử trên Google Colab.

Core libraries:
Các thư viện chính:

Python
PyTorch
torchvision
timm
pandas
numpy
scikit learn
Pillow
tqdm
12. Installation
12. Cài đặt

Clone the repository and install dependencies:
Clone repository và cài đặt thư viện:

git clone https://github.com/2274802010922/vaipe-contextual-pill-recognition.git
cd vaipe-contextual-pill-recognition
pip install -r requirements.txt
13. Dataset download on Google Colab
13. Tải dữ liệu trên Google Colab

Example:
Ví dụ:

!pip install -q kagglehub
import kagglehub, os

path = kagglehub.dataset_download("tommyngx/vaipepill2022")
print("Dataset path:", path)

os.environ["VAIPE_TRAIN_ROOT"] = f"{path}/public_train"
os.environ["PIKA_BEST_OUTPUT_ROOT"] = "/content/processed_pika_best"
14. How to run
14. Cách chạy
14.1 Baseline
14.1 Baseline
python prepare_cropped_dataset.py
python train_baseline.py
14.2 PIKA like baseline
14.2 PIKA like baseline
python build_pika_metadata.py
python train_pika_baseline.py
14.3 Structured context version
14.3 Phiên bản structured context
python build_pika_context_metadata.py
python train_pika_v2_context_labels.py
14.4 Triple context version
14.4 Phiên bản triple context
python build_pika_v3_metadata.py
python train_pika_v3_triple_context.py
14.5 Graph based version
14.5 Phiên bản graph based
python build_pika_graph_data.py
python train_pika_graph.py
14.6 Final best model
14.6 Mô hình tốt nhất cuối cùng
python build_best_pika_data.py
python train_best_pika_model.py
14.7 Fine tuning from the best checkpoint
14.7 Fine tune từ checkpoint tốt nhất
python finetune_best_pika_model.py

If you are using Google Colab, make sure the following environment variable points to the saved checkpoint:

Nếu bạn dùng Google Colab, hãy chắc rằng biến môi trường sau trỏ đúng đến checkpoint đã lưu:

import os
os.environ["PIKA_BASE_CHECKPOINT"] = "/content/vaipe-contextual-pill-recognition/outputs_best_pika/best_pika_model.pth"
15. Notes about checkpoint saving
15. Ghi chú về lưu checkpoint

To avoid losing the best model after a runtime reset, save the following files to Google Drive:

Để tránh mất mô hình tốt nhất sau khi runtime reset, nên lưu các file sau vào Google Drive:

best_pika_model.pth
best_pika_metadata.csv
graph_labels.json
graph_pmi.npy

These files are enough to resume from the best model and run fine tuning later.

Các file này là đủ để khôi phục mô hình tốt nhất và chạy fine tune ở các lần sau.

16. Key observations from experiments
16. Các nhận xét chính từ thực nghiệm
Image only classification is not sufficient for this problem
Prescription image gives a strong contextual boost
Simple structured context alone is not enough
A graph branch only becomes effective when combined properly with strong visual context
Fine tuning is the final step that pushed the model beyond the paper result
Mô hình chỉ dùng ảnh là chưa đủ cho bài toán này
Ảnh prescription mang lại cải thiện ngữ cảnh rõ rệt
Structured context đơn giản là chưa đủ
Nhánh graph chỉ phát huy hiệu quả khi được kết hợp đúng với ngữ cảnh thị giác mạnh
Fine tune là bước cuối cùng giúp mô hình vượt kết quả của bài báo
17. Future improvements
17. Hướng phát triển tiếp theo

Possible future directions include:
Một số hướng phát triển tiếp theo gồm:

more detailed ablation on graph construction
stronger graph encoder
improved visualization and explainability
confusion matrix and per class F1 analysis
deployment as an interactive demo application
ablation chi tiết hơn về cách xây dựng graph
graph encoder mạnh hơn
cải thiện trực quan hóa và khả năng giải thích
phân tích confusion matrix và F1 theo từng lớp
triển khai thành ứng dụng demo tương tác
18. Acknowledgment
18. Lời cảm ơn và ghi nhận

This project is based on the idea of contextual pill recognition introduced by the PIKA paper and uses the VAIPE dataset as the experimental foundation.

Dự án này được xây dựng dựa trên ý tưởng nhận dạng thuốc theo ngữ cảnh của bài báo PIKA và sử dụng bộ dữ liệu VAIPE làm nền tảng thực nghiệm.
