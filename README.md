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
## 8.1. Công nghệ kế thừa từ bài báo PIKA và các điểm khác biệt giúp mô hình đạt kết quả tốt hơn

Dự án này được phát triển dựa trên định hướng của bài báo **PIKA** trong bài toán **contextual pill recognition**. Thay vì xem nhận dạng thuốc là một bài toán phân loại ảnh độc lập cho từng viên thuốc, cách tiếp cận này khai thác thêm **ngữ cảnh đơn thuốc** và **quan hệ giữa các thuốc xuất hiện cùng nhau** để hỗ trợ dự đoán. Đây cũng là tư tưởng cốt lõi xuyên suốt trong mô hình được xây dựng trong repo.

### 8.1.1. Những thành phần được kế thừa từ bài báo PIKA

Trước hết, dự án kế thừa quan điểm rằng bài toán nhận dạng thuốc không nên chỉ dựa vào đặc trưng thị giác của từng ảnh crop. Trong thực tế, các viên thuốc thường xuất hiện trong cùng một prescription và tồn tại những mối liên hệ ngữ cảnh có ý nghĩa. Vì vậy, ngoài nhánh xử lý ảnh viên thuốc, mô hình còn khai thác thêm thông tin từ toàn bộ prescription để hỗ trợ quá trình nhận dạng.

Thứ hai, dự án kế thừa ý tưởng biểu diễn quan hệ giữa các lớp thuốc dưới dạng **đồ thị ngữ cảnh**. Cách biểu diễn này cho phép mô hình học không chỉ từ ảnh mà còn từ tri thức về việc những thuốc nào thường xuất hiện cùng nhau trong thực tế. Đây là một thành phần quan trọng trong tinh thần của PIKA, đặc biệt hữu ích đối với các lớp thuốc có đặc điểm thị giác gần giống nhau.

Thứ ba, dự án kế thừa cách tiếp cận **kết hợp đặc trưng thị giác và đặc trưng ngữ cảnh** trước bước phân loại cuối cùng. Nói cách khác, mô hình không đưa ra quyết định dựa trên một nguồn dữ liệu duy nhất, mà dựa trên biểu diễn hợp nhất của nhiều nguồn thông tin khác nhau. Nhờ đó, hệ thống có khả năng dự đoán toàn diện hơn và ổn định hơn.

Từ các thành phần kế thừa trên, project vẫn giữ được định hướng nghiên cứu cốt lõi của bài báo PIKA, đồng thời mở rộng thêm nhiều điểm cải tiến để nâng cao hiệu năng.

### 8.1.2. Khác biệt thứ nhất: nâng cấp backbone cho nhánh ảnh viên thuốc

Một trong những khác biệt quan trọng nhất của repo này so với bài báo tham chiếu là việc nâng cấp backbone của nhánh ảnh viên thuốc. Trong khi mô hình tham chiếu sử dụng backbone nhẹ hơn, project này sử dụng **EfficientNetV2-S** để trích xuất đặc trưng thị giác.

Việc sử dụng backbone mạnh hơn giúp mô hình học tốt hơn các đặc trưng tinh vi của viên thuốc, bao gồm:

- hình dạng
- màu sắc
- độ tương phản
- đường viền
- texture bề mặt
- các chi tiết nhỏ khó phân biệt giữa các lớp

Điều này đặc biệt quan trọng trong bài toán pill recognition, vì nhiều lớp thuốc có ngoại hình rất giống nhau và chỉ khác ở những tín hiệu thị giác rất nhỏ. Nhờ backbone mạnh hơn, nhánh ảnh trong project tạo ra biểu diễn đặc trưng giàu thông tin hơn trước khi kết hợp với ngữ cảnh.

### 8.1.3. Khác biệt thứ hai: sử dụng đồng thời ba nhánh thông tin

Khác với cách tiếp cận chỉ tập trung mạnh vào một nguồn ngữ cảnh, mô hình cuối cùng trong project được thiết kế theo hướng **đa nhánh**, bao gồm:

- **nhánh ảnh viên thuốc**
- **nhánh ảnh prescription**
- **nhánh graph context**

Điểm đáng chú ý là ảnh prescription không chỉ được xem là dữ liệu phụ, mà trở thành một **nhánh ngữ cảnh độc lập** trong mô hình. Nhánh này giúp mô hình khai thác bối cảnh tổng thể của đơn thuốc, chẳng hạn như bố cục ảnh, tín hiệu toàn cục của prescription và thông tin về các thuốc cùng xuất hiện trong cùng một mẫu.

Việc sử dụng song song cả ba nhánh giúp mô hình học được đồng thời:

- đặc trưng cục bộ của từng viên thuốc
- ngữ cảnh hình ảnh của toàn bộ prescription
- quan hệ ngữ nghĩa giữa các lớp thuốc

Đây là một mở rộng quan trọng so với baseline chỉ sử dụng ảnh viên thuốc, và cũng là một trong những nguyên nhân chính giúp mô hình cuối cùng đạt hiệu năng cao hơn.

### 8.1.4. Khác biệt thứ ba: xây dựng graph context bằng PPMI

Một cải tiến nổi bật của project nằm ở cách xây dựng đồ thị ngữ cảnh. Thay vì chỉ sử dụng tần suất đồng xuất hiện thô giữa các lớp thuốc, repo này xây dựng đồ thị dựa trên **PPMI (Positive Pointwise Mutual Information)**.

Ưu điểm của PPMI là làm nổi bật những cặp thuốc có mối liên hệ ngữ cảnh thực sự mạnh, đồng thời giảm ảnh hưởng của các quan hệ xuất hiện nhiều nhưng ít mang giá trị phân biệt. Nhờ đó, đồ thị thu được không chỉ là một ma trận đếm đơn thuần, mà trở thành một dạng **tri thức ngữ cảnh có tính chọn lọc cao hơn**.

Sau khi xây dựng PPMI graph, project tiếp tục sử dụng:

- **GCN hai lớp**
- **context attention**

để mã hóa và chọn lọc thông tin đồ thị. Cơ chế này giúp mô hình không chỉ lưu giữ quan hệ giữa các thuốc mà còn học được phần ngữ cảnh nào quan trọng nhất đối với từng đầu vào cụ thể. Đây là một thành phần rất quan trọng góp phần cải thiện hiệu quả của mô hình.

### 8.1.5. Khác biệt thứ tư: mở rộng cơ chế fusion bằng gated fusion và interaction features

Một điểm khác biệt lớn khác của project nằm ở cơ chế hợp nhất đặc trưng. Thay vì chỉ nối trực tiếp các vector đặc trưng từ nhiều nhánh, mô hình sử dụng **gated fusion** để điều chỉnh mức đóng góp của từng nguồn thông tin.

Cơ chế này cho phép mô hình tự học:

- khi nào nên ưu tiên ảnh viên thuốc
- khi nào nên khai thác mạnh hơn ảnh prescription
- khi nào graph context đóng vai trò quan trọng hơn

Nhờ đó, quá trình fusion trở nên linh hoạt hơn so với phép nối đặc trưng thông thường.

Ngoài ra, project còn bổ sung **interaction features**, tức là không chỉ giữ các đặc trưng độc lập mà còn mô hình hóa tương tác giữa chúng, ví dụ:

- **pill feature × prescription feature**
- **pill feature × graph context**
- **prescription feature × graph context**

Nhờ cơ chế này, biểu diễn cuối cùng giàu thông tin hơn, phản ánh tốt hơn mối quan hệ giữa các nguồn dữ liệu và tăng khả năng phân biệt ở tầng phân loại cuối.

### 8.1.6. Khác biệt thứ năm: chiến lược fine-tuning nhiều giai đoạn

Một khác biệt rất quan trọng giữa project này và mô hình tham chiếu là chiến lược huấn luyện. Dự án không dừng lại ở việc huấn luyện một mô hình duy nhất, mà tiếp tục tối ưu thông qua nhiều giai đoạn fine-tuning có kiểm soát.

Các phiên bản fine-tune trong repo đã áp dụng nhiều kỹ thuật tối ưu như:

- **differential learning rates**
- **effective number class weights**
- **label smoothing**
- **cosine learning rate scheduler**
- **gradient clipping**
- **early stopping**
- **resume checkpoint**

Những kỹ thuật này không làm thay đổi hoàn toàn kiến trúc mô hình, nhưng giúp quá trình học ổn định hơn, giảm nguy cơ overfitting, hạn chế phá vỡ representation tốt đã học được trước đó và khai thác hiệu quả hơn các checkpoint mạnh.

Đặc biệt, các giai đoạn fine-tuning đã giúp mô hình được cải thiện dần qua từng phiên bản, từ:

- **0.8380** ở bản Best PIKA trước fine-tune
- **0.8671** ở phiên bản fine-tune tiếp theo
- **0.8936** ở phiên bản cải tiến sau đó
- **0.9011** ở phiên bản tốt nhất

Điều này cho thấy chiến lược huấn luyện nhiều giai đoạn đóng vai trò quan trọng trong việc đẩy hiệu năng mô hình lên mức cao hơn.

### 8.1.7. Tác động của các khác biệt này lên kết quả cuối cùng

Nhờ các cải tiến nêu trên, mô hình trong repo không chỉ tái hiện lại định hướng nghiên cứu của bài báo PIKA mà còn mở rộng nó theo hướng mạnh hơn về mặt thực nghiệm.

Cụ thể, kết quả tốt nhất của bài báo tham chiếu là:

- **F1-score = 0.8571**

Trong khi đó, phiên bản tốt nhất của project là **Fine-tune v3**, đạt:

- **Validation Macro F1 = 0.9011**
- **Validation Accuracy = 0.9334**

Khoảng chênh lệch này cho thấy các cải tiến trong repo không chỉ mang ý nghĩa kỹ thuật ở mức kiến trúc hay tối ưu huấn luyện, mà thực sự tạo ra tác động tích cực lên hiệu năng nhận dạng.

### 8.1.8. Kết luận

Tóm lại, project này kế thừa tư tưởng cốt lõi của bài báo PIKA ở việc kết hợp thông tin hình ảnh và ngữ cảnh để nhận dạng thuốc, nhưng được mở rộng theo nhiều hướng quan trọng. Cụ thể, mô hình sử dụng backbone mạnh hơn cho nhánh ảnh viên thuốc, khai thác rõ ràng hơn ngữ cảnh prescription, xây dựng graph context bằng PPMI, mở rộng cơ chế fusion theo hướng linh hoạt hơn và áp dụng chiến lược fine-tuning nhiều giai đoạn để tối ưu hiệu quả học.

Chính các khác biệt đó đã tạo nền tảng để mô hình cuối cùng không chỉ bám sát tinh thần của bài báo tham chiếu mà còn đạt kết quả thực nghiệm cao hơn.

## 9. Điểm nổi bật của v3 so với v2

So với phiên bản v2, phiên bản v3 có các điểm nổi bật sau:

- bổ sung **resume training**
- tự động lưu **resume checkpoint sau mỗi epoch**
- có thể tiếp tục huấn luyện nếu Google Colab bị ngắt
- giúp quá trình fine-tune ổn định và an toàn hơn
- tiếp tục cải thiện kết quả từ **0.8936** lên **0.9011**

---
