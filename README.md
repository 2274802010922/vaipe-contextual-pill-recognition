# Nghiên cứu và cải tiến mô hình nhận dạng thuốc theo ngữ cảnh trên bộ dữ liệu VAIPE

## 1. Giới thiệu

Dự án này nghiên cứu bài toán nhận dạng thuốc theo ngữ cảnh trên bộ dữ liệu VAIPE, bám theo và cải tiến ý tưởng của bài báo **Image-based Contextual Pill Recognition with Medical Knowledge Graph Assistance (PIKA)**.

Mục tiêu của dự án không chỉ là phân loại ảnh viên thuốc sau khi crop, mà còn khai thác thêm:
- ngữ cảnh từ ảnh prescription
- ngữ cảnh đồ thị từ các thuốc đồng xuất hiện
- chiến lược fine-tuning nhiều giai đoạn

Kết quả cuối cùng của dự án đạt **Validation Macro F1 = 0.9011**, vượt kết quả tốt nhất được báo cáo trong bài báo PIKA là **0.8571**.

---

## 2. Bối cảnh bài toán

Nhận dạng thuốc là một bài toán khó trong thị giác máy tính vì nhiều viên thuốc có:
- màu sắc gần giống nhau
- hình dạng tương tự
- kích thước nhỏ
- ít đặc trưng thị giác nổi bật

Nếu chỉ dùng ảnh viên thuốc, mô hình thường dễ nhầm giữa các lớp có đặc trưng rất gần nhau.

Trong thực tế, thuốc không xuất hiện độc lập mà thường đi kèm trong cùng một đơn thuốc. Vì vậy, ngoài ảnh viên thuốc, các nguồn thông tin sau có thể hỗ trợ mạnh cho quá trình nhận dạng:
- ảnh prescription
- tập thuốc đồng xuất hiện trong cùng prescription
- quan hệ giữa các lớp thuốc dưới dạng graph

Đây chính là hướng tiếp cận contextual pill recognition mà dự án này theo đuổi.

---

## 3. Bài báo tham chiếu

Bài báo tham chiếu chính của dự án:

**Image-based Contextual Pill Recognition with Medical Knowledge Graph Assistance**  
Link: `https://arxiv.org/abs/2208.02432`

Theo bài báo, mô hình PIKA tốt nhất đạt **F1-score = 0.8571** với backbone ResNet-18.

---

## 4. Bộ dữ liệu

Dự án sử dụng bộ dữ liệu **VAIPE**.

Một số liên kết liên quan:
- Dataset Kaggle: `https://www.kaggle.com/datasets/tommyngx/vaipepill2022`
- VAIPE resource: `https://smarthealth.vinuni.edu.vn/resources/`

Cấu trúc dữ liệu huấn luyện chính gồm:
- ảnh pill
- nhãn pill dạng JSON
- ảnh prescription
- file ánh xạ `pill_pres_map.json`

Dữ liệu gốc được xử lý thành:
- các ảnh crop viên thuốc
- metadata cho nhiều phiên bản mô hình
- graph ngữ cảnh dựa trên đồng xuất hiện giữa các lớp thuốc

---

## 5. Mục tiêu dự án

Dự án có ba mục tiêu chính:

1. Tái hiện bài toán nhận dạng thuốc theo ngữ cảnh trên bộ dữ liệu VAIPE
2. So sánh nhiều cách mô hình hóa ngữ cảnh khác nhau
3. Xây dựng mô hình cuối cùng có kết quả tốt hơn baseline và vượt bài báo PIKA

---

## 6. Các phiên bản mô hình đã triển khai

### 6.1. Baseline chỉ dùng ảnh
Phiên bản đầu tiên chỉ sử dụng ảnh viên thuốc sau khi crop để phân loại.

### 6.2. PIKA-like v1
Phiên bản này kết hợp:
- ảnh viên thuốc
- ảnh prescription

Đây là phiên bản contextual đầu tiên mang lại cải thiện rõ rệt so với baseline.

### 6.3. PIKA v2
Phiên bản này sử dụng:
- ảnh viên thuốc
- structured context labels từ cùng prescription

### 6.4. PIKA v3
Phiên bản này kết hợp:
- ảnh viên thuốc
- ảnh prescription
- structured context labels

### 6.5. PIKA graph v1
Phiên bản này sử dụng:
- ảnh viên thuốc
- graph context được xây dựng từ các nhãn thuốc đồng xuất hiện

### 6.6. Best PIKA model
Đây là mô hình mạnh nhất trước fine-tuning, kết hợp:
- ảnh viên thuốc
- ảnh prescription
- graph context theo PPMI
- gated fusion
- interaction features

### 6.7. Fine-tune v1
Phiên bản fine-tune đầu tiên được huấn luyện tiếp từ checkpoint tốt nhất của Best PIKA model.

### 6.8. Fine-tune v2
Phiên bản này tiếp tục nâng cấp chiến lược fine-tuning với:
- differential learning rates
- effective number class weights
- cosine learning rate scheduler
- gradient clipping
- early stopping

### 6.9. Fine-tune v3
Phiên bản này giữ toàn bộ cải tiến mạnh của v2, đồng thời bổ sung:
- lưu best checkpoint
- lưu resume checkpoint sau mỗi epoch
- khả năng resume training nếu Colab bị ngắt

Đây là phiên bản tốt nhất hiện tại của dự án.

---

## 7. Kết quả thực nghiệm

### 7.1. Kết quả qua các giai đoạn

| Phiên bản mô hình | Validation Macro F1 | Validation Accuracy |
|---|---:|---:|
| Baseline chỉ dùng ảnh | 0.4801 | 0.5774 |
| PIKA-like v1 | 0.6054 | 0.5959 |
| PIKA v2 | 0.5957 | 0.6051 |
| PIKA v3 | 0.5882 | 0.5505 |
| PIKA graph v1 | 0.5654 | 0.5843 |
| Best PIKA trước fine-tune | 0.8380 | 0.6285 |
| Fine-tune v1 | 0.8671 | 0.8903 |
| Fine-tune v2 | 0.8936 | 0.9243 |
| Fine-tune v3 | 0.9011 | 0.9334 |
| PIKA paper | 0.8571 | không báo cáo theo cùng thiết lập |

### 7.2. Kết quả tốt nhất hiện tại

Mô hình tốt nhất của dự án là **Fine-tune v3**, đạt:

- **Best Val Macro F1 = 0.9011**
- **Validation Accuracy = 0.9334**

Kết quả này cao hơn mốc **0.8571** của bài báo PIKA.

---

## 8. Vì sao mô hình cuối cùng mạnh hơn

Một số điểm nổi bật giúp mô hình cuối cùng đạt kết quả cao hơn:

- sử dụng đồng thời **ảnh viên thuốc**, **ảnh prescription** và **graph context**
- xây dựng graph ngữ cảnh bằng **PPMI**
- dùng **gated fusion** thay vì ghép đặc trưng đơn giản
- bổ sung **interaction features** giữa các nhánh
- fine-tune nhiều giai đoạn từ checkpoint tốt nhất
- cải thiện rõ ở các lớp khó và các lớp có nhiều mẫu

---

## 9. Điểm nổi bật của v3 so với v2

So với phiên bản v2, phiên bản v3 có các điểm nổi bật sau:

- bổ sung **resume training**
- tự động lưu **resume checkpoint sau mỗi epoch**
- có thể tiếp tục huấn luyện nếu Google Colab bị ngắt
- giúp quá trình fine-tune ổn định và an toàn hơn
- tiếp tục cải thiện kết quả từ **0.8936** lên **0.9011**

---
