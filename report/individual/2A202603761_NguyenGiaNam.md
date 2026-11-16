# Báo cáo cá nhân — Day 07: Retrieval & Context Engineering

- MSSV: 2A202603761
- Họ và tên: Nguyễn Gia Nam
- Split: DEV 17 câu, TEST 7 câu
- Cấu hình cuối: Hybrid RRF + Seed-First Neighbor Expansion
- Model: gemini-3.5-flash-lite

## 1. Dữ liệu và benchmark

### 1.1. Nguồn dữ liệu

Sử dụng dữ liệu từ Lab 06, nguồn `python_wang_2023`, thuộc tài liệu Introduction to Computer Programming with Python.

- Corpus Lab 07: 200 chunks, chiến lược `by_paragraph`.
- Phạm vi: 45 trang có dữ liệu, trang PDF 190–235.
- Vector store: ChromaDB, collection `lab06_chunks` (403 total Lab 06 chunks; 200 by_paragraph chunks used in Lab 07).
- Embedding: MiniLM, 384 dimensions thông qua `MyEmbedder("minilm")` của Lab 06.
- Mô hình embedding chạy bằng ONNX.
- Quyền sử dụng tài liệu và phiên bản thư viện: cần đối chiếu metadata gốc trước khi ghi thông số chính thức.

Dữ liệu được tái sử dụng từ Lab 06 thay vì tạo embedding mới bằng API.

### 1.2. Benchmark

Tổng cộng 24 câu hỏi:

| Split | Answerable | Unanswerable | Tổng |
|---|---:|---:|---:|
| DEV | 15 | 2 | 17 |
| TEST | 5 | 2 | 7 |
| Tổng | 20 | 4 | 24 |

Mỗi câu answerable có Gold Evidence theo source, page và evidence text. Các câu unanswerable có `gold=[]` và yêu cầu Agent tìm kiếm rồi từ chối khi không đủ bằng chứng.

DEV dùng để so sánh, tối ưu và lựa chọn cấu hình. TEST chỉ chạy sau khi khóa cấu hình.

### 1.3. Kiểm tra unanswerable

Bốn câu q21–q24 hỏi các tính năng Python mới hoặc asyncio.TaskGroup.

- Quét 200 chunks: không tìm thấy bằng chứng trực tiếp.
- q24 có một chunk chứa `asynchronous` nhưng chỉ mô tả `StopAsyncIteration`, không giải thích TaskGroup.
- Việc quét 501 trang PDF trước đó chưa phát hiện từ khóa tương ứng với q21/q22.
- Hành vi Agent: cả bốn câu đều `abstain`.

Giới hạn: kiểm tra từ khóa và phạm vi corpus chưa thay thế hoàn toàn việc xác nhận thủ công trên tài liệu gốc. Do đó, điểm Abstention Correctness chính thức còn chờ xác minh.

## 2. Retrieval và thực nghiệm

### 2.1. Các phương pháp

Đã thử nghiệm:

1. TF-IDF: truy xuất dựa trên từ khóa.
2. Dense Retrieval: truy xuất vector bằng MiniLM và ChromaDB.
3. Hybrid Retrieval: kết hợp TF-IDF và Dense với Reciprocal Rank Fusion (RRF k=60).
4. Context Engineering: thay đổi candidate depth, context budget và mở rộng chunks lân cận.

### 2.2. So sánh Retrieval trên DEV và TEST

| Configuration | Method, top-k, filter, rerank, budget | Hypothesis / change | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Latency / cost |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF; top 3; budget 800 | Lexical baseline | DEV | 2/15 (13.3%) | 4/15 (26.7%) | 3/15 (20.0%) | Median 0.191 ms; local $0 |
| Method 2 | Dense MiniLM; Chroma; top 3; budget 800 | Semantic retrieval | DEV | 2/15 (13.3%) | 5/15 (33.3%) | 2/15 (13.3%) | Median 7.307 ms; local $0 |
| Method 3 | Hybrid RRF k=60; depth 3; top 3; budget 800 | Lexical + dense fusion | DEV | 3/15 (20.0%) | 5/15 (33.3%) | 3/15 (20.0%) | Chưa ghi nhận |
| Improvement 1 | Hybrid RRF; depth 10; top 3; budget 1600 | Increase candidate depth | DEV | 5/15 (33.3%) | 7/15 (46.7%) | 7/15 (46.7%) | Median 6.414 ms; local $0 |
| Improvement 2 | Hybrid depth 10; Seed-First; budget 1600 | Neighbor expansion | DEV | Chưa đo riêng | Chưa đo riêng | 9/15 (60.0%) | Chưa đo riêng |
| Selected configuration | Hybrid depth 10; Seed-First; budget 2400 | Preserve seeds, then add neighbors | DEV | Chưa đo riêng | Chưa đo riêng | 11/15 (73.3%) | Thử nghiệm context offline |
| Selected configuration | Hybrid depth 10; Seed-First; top 3; budget 2400 | Frozen DEV-selected configuration | TEST | 2/5 (40.0%) | 2/5 (40.0%) | 2/5 (40.0%) | Mean 15.386 ms; local retrieval |
| Baseline | TF-IDF; top 3; budget 800 | Held-out baseline | TEST | 4/5 (80.0%) | 4/5 (80.0%) | 4/5 (80.0%) | Mean 0.278 ms; local retrieval |

The Context Hit comparison uses the original strict evidence-matching rule. The 11/15 DEV result was measured offline with the original DEV questions. The Agent generates its own search queries, so offline retrieval results and Agent results are not directly interchangeable.

On held-out TEST, lexical retrieval outperformed the selected Hybrid method. This result was retained without changing the frozen configuration.

### 2.3. Context Engineering

Thử nghiệm cho thấy tăng candidate depth giúp cải thiện khả năng tìm thấy evidence. Tuy nhiên, không phải lúc nào tăng depth cũng cải thiện Context Hit do giới hạn số lượng chunks và ngân sách context.

Chiến lược Seed-First gồm:

1. Giữ ưu tiên các chunks được xếp hạng cao.
2. Tìm đoạn văn liền trước và liền sau trên cùng trang.
3. Bổ sung neighbors khi còn ngân sách.
4. Không đưa chunks trùng lặp vào context.
5. Chỉ chấp nhận citation thuộc các chunks thực sự xuất hiện trong context.

Theo phép đo strict, cấu hình Hybrid depth 10 budget 1600 đạt 7/15, còn Seed-First budget 2400 đạt 11/15.

Kết quả cải thiện 4 câu, tương ứng 26,7 điểm phần trăm. Tuy nhiên, so sánh này đồng thời thay đổi chiến lược đóng gói và ngân sách. So sánh cùng budget 1600 cho mức tăng từ 7/15 lên 9/15.

Chọn Seed-First 2400 làm cấu hình cuối trên DEV.

### 2.4. Các trường hợp thất bại

- q03: yêu cầu giải thích cách viết khối lệnh sau dấu hai chấm của `if`; Agent từ chối dù corpus có Gold Evidence.
- q07: yêu cầu giải thích lợi ích của `for` khi lặp lại thao tác; Agent chưa truy xuất đủ bằng chứng để trả lời.

Hai trường hợp thể hiện hạn chế của Retrieval khi câu hỏi được diễn đạt bằng tiếng Việt, trong khi corpus là tiếng Anh.

### 2.5. TEST Retrieval

Retrieval-only TEST used five answerable questions, without Gemini calls. TF-IDF achieved 4/5 Context Evidence Rate, while Hybrid + Seed-First achieved 2/5.

The selected method performed worse on exact error-name queries in this small TEST subset. This is reported as a limitation, not used for additional TEST-driven tuning.



### 2.6. Danh gia context TEST theo thang 0/1/2

Ket qua cham thu cong 5 cau TEST answerable cho moi cau hinh:

| Cau hinh | Tong diem | Diem trung binh | So cau dat 2 diem |
|---|---:|---:|---:|
| TF-IDF baseline | 9/10 | 1.80/2 | 4/5 |
| Hybrid + Seed-First | 8/10 | 1.60/2 | 3/5 |

Danh gia noi dung khac voi Strict Context Hit. Vi du q20 co context
giai thich `finally` duoc thuc thi bat ke co exception hay khong,
nhung khong trung hoan toan Gold Evidence da gan.

Chi tiet diem va ly do: `submission_evidence/retrieval/test_context_human_review.csv`.

<!-- LAB07_CONTEXT_REVIEW -->

## 3. Agentic RAG và đánh giá

### 3.1. Thiết kế Agent

Các thành phần:

- Controller: Gemini, model `gemini-3.5-flash-lite`.
- Retriever: Hybrid TF-IDF + Dense MiniLM.
- Fusion: RRF.
- Context Engineering: Seed-First Neighbor Expansion.
- `top_k=3`.
- `max_searches=3`.
- `budget=2400` ký tự.

Agent có thể chọn `search` để truy xuất bằng chứng hoặc `answer` để trả lời.

Câu trả lời chỉ được chấp nhận khi có citation hợp lệ trong context. Khi không đủ bằng chứng, Agent có thể chủ động abstain. Trường hợp citation sai hoặc thiếu bị từ chối.

Lỗi provider được ghi nhận riêng. Một số phản hồi Gemini có JSON không hợp lệ, cần bước sửa JSON; đây là chi phí và giới hạn thực tế của hệ thống.

### 3.2. Bảng đánh giá Agent

| Configuration / split | Tool use | Model calls | Grounded answer 0/1/2 and mean | Valid citations | Answers with citations | Correct abstention | False abstention |
|---|---|---|---|---|---|---|---|
| Hybrid + Seed-First / DEV | 17/17 | 35 total | 0:2, 1:0, 2:13; mean 1.733/2 | 18/18 | 13/13 | 2/2 behavioral; chờ xác minh thủ công | 2/15 |
| Hybrid + Seed-First / TEST | 7/7 | 15 total | 0:1, 1:0, 2:4; mean 1.600/2 | 4/4 | 4/4 | 2/2 behavioral; chờ xác minh thủ công | 1/5 |

The provisional metric JSON assigns `abstention_correct=false` to cases pending human confirmation. That value must not be presented as a verified model failure.

### 3.3. Phân tích theo từng câu hỏi

| Query ID | Question type | Tool calls / outcome | Answer and citations | Review label / reason |
|---|---|---|---|---|
| q01 (DEV) | Answerable | 1 search; answered | Trong Python, câu lệnh if được sử dụng để thực hiện các lựa chọn [by_paragraph:python_wang_2023_p190_p...; citations: by_paragraph:python_wang_2023_p190_para3, by_paragraph:python_wang_2023_p210_para1, by_paragraph:python_wang_2023_p210_para0 | Grade 2; citation supported |
| q02 (DEV) | Answerable | 1 search; answered | Cấu trúc if-elif-else dùng để giải quyết các bài toán có nhiều sự lựa chọn (multiple selections) [by_p...; citations: by_paragraph:python_wang_2023_p210_para1, by_paragraph:python_wang_2023_p192_para1 | Grade 2; citation supported |
| q03 (DEV) | Answerable | 2 search; abstain | Không đủ bằng chứng để trả lời. | Grade 0; false abstention |
| q04 (DEV) | Answerable | 1 search; answered | Theo bảng quy đổi điểm trong chương 3, điểm từ 90 đến 100 được xếp loại A+ [by_paragraph:python_wang_2...; citations: by_paragraph:python_wang_2023_p208_para1 | Grade 2; citation supported |
| q05 (DEV) | Answerable | 1 search; answered | Trong ví dụ chuyển điểm số sang điểm chữ, chương trình sử dụng cấu trúc câu lệnh if-elif-else [by_para...; citations: by_paragraph:python_wang_2023_p194_para1 | Grade 2; citation supported |
| q06 (DEV) | Answerable | 1 search; answered | Các nhánh elif trong ví dụ chấm điểm không cần ghi rõ cận trên vì đoạn mã tận dụng cấu trúc câu lệnh i...; citations: by_paragraph:python_wang_2023_p195_para0 | Grade 2; citation supported |
| q07 (DEV) | Answerable | 1 search; abstain | Không đủ bằng chứng để trả lời. | Grade 0; false abstention |
| q08 (DEV) | Answerable | 1 search; answered | Trong ví dụ tạo bảng nhân hình tam giác, chúng ta cần hai vòng lặp vì vòng lặp bên ngoài được sử dụng ...; citations: by_paragraph:python_wang_2023_p196_para2 | Grade 2; citation supported |
| q09 (DEV) | Answerable | 1 search; answered | Vòng lặp for lồng nhau là hai vòng lặp được đặt bên trong nhau, trong khi việc sử dụng zip trên hai ra...; citations: by_paragraph:python_wang_2023_p199_para1 | Grade 2; citation supported |
| q10 (DEV) | Answerable | 1 search; answered | Trong vòng lặp for, câu lệnh break được sử dụng để thoát khỏi vòng lặp ngay lập tức khi có điều kiện g...; citations: by_paragraph:python_wang_2023_p199_para2, by_paragraph:python_wang_2023_p204_para3 | Grade 2; citation supported |
| q11 (DEV) | Answerable | 1 search; answered | Sách khuyên nên dùng vòng lặp while thay vì vòng lặp for trong các trường hợp mà số lần lặp chưa biết ...; citations: by_paragraph:python_wang_2023_p201_para0, by_paragraph:python_wang_2023_p205_para0 | Grade 2; citation supported |
| q12 (DEV) | Answerable | 1 search; answered | Trong ví dụ nhập điểm khi số lượng sinh viên không biết trước, vòng lặp sử dụng điều kiện `if (mark >=...; citations: by_paragraph:python_wang_2023_p207_para0 | Grade 2; citation supported |
| q13 (DEV) | Answerable | 1 search; answered | Để đảm bảo vòng lặp while kết thúc như mong đợi, điều kiện lặp phải được thay đổi bên trong khối mã củ...; citations: by_paragraph:python_wang_2023_p205_para0 | Grade 2; citation supported |
| q14 (DEV) | Answerable | 1 search; answered | Ví dụ tính điểm trung bình của lớp có 30 sinh viên có thể dùng vòng lặp for vì số lượng sinh viên tron...; citations: by_paragraph:python_wang_2023_p206_para1 | Grade 2; citation supported |
| q15 (DEV) | Answerable | 1 search; answered | Bắt Exception chung không xác định được chính xác lỗi chia cho 0 vì `Exception` là lớp cha (superclass...; citations: by_paragraph:python_wang_2023_p218_para0 | Grade 2; citation supported |
| q21 (DEV) | Unanswerable | 1 search; abstain | Không đủ bằng chứng để trả lời. | Abstain; chờ xác minh thiếu bằng chứng |
| q22 (DEV) | Unanswerable | 1 search; abstain | Không đủ bằng chứng để trả lời. | Abstain; chờ xác minh thiếu bằng chứng |
| q16 (TEST) | Answerable | 1 search; answered | Lỗi ZeroDivisionError xuất hiện khi số chia của một phép chia hoặc phép toán chia lấy dư (modulo) là 0...; citations: by_paragraph:python_wang_2023_p219_para3 | Grade 2; citation supported |
| q17 (TEST) | Answerable | 1 search; answered | Lỗi IndexError xuất hiện khi chỉ số (index) của một chuỗi (sequence) nằm ngoài phạm vi cho phép [by_pa...; citations: by_paragraph:python_wang_2023_p222_para0 | Grade 2; citation supported |
| q18 (TEST) | Answerable | 1 search; answered | Khi sử dụng một biến chưa được định nghĩa, Python sẽ báo lỗi NameError [by_paragraph:python_wang_2023_...; citations: by_paragraph:python_wang_2023_p223_para1 | Grade 2; citation supported |
| q19 (TEST) | Answerable | 1 search; answered | TabError xảy ra khi việc thụt lề (indentation) bao gồm các tab và khoảng trắng không nhất quán [by_par...; citations: by_paragraph:python_wang_2023_p226_para1 | Grade 2; citation supported |
| q20 (TEST) | Answerable | 2 search; abstain | Không đủ bằng chứng để trả lời. | Grade 0; false abstention |
| q23 (TEST) | Unanswerable | 1 search; abstain | Không đủ bằng chứng để trả lời. | Abstain; chờ xác minh thiếu bằng chứng |
| q24 (TEST) | Unanswerable | 1 search; abstain | Không đủ bằng chứng để trả lời. | Abstain; chờ xác minh thiếu bằng chứng |

Full text and citation evidence are preserved in the saved traces. DEV average latency was 5666.14 ms/question; TEST average latency was 6223.53 ms/question. Gemini monetary cost was not calculated.

### 3.4. Kết quả DEV chi tiết

| Metric | Kết quả |
|---|---:|
| Tổng số câu | 17 |
| Provider Errors | 0 |
| Tool Use | 17/17 |
| Grounded Answer Grade 0 | 2 |
| Grounded Answer Grade 1 | 0 |
| Grounded Answer Grade 2 | 13 |
| Grounded Answer Mean | 1,733/2 |
| Citation Validity | 18/18 |
| Answer Citation Coverage | 13/13 |
| False Abstention | 2/15 |
| Model Calls | 35 |
| Latency trung bình | 5.666 ms/câu |

Các trường hợp q09 và q15 trước đó từng abstain, nhưng sau Seed-First đã trả lời được với citation phù hợp. Đây là kết quả quan sát trên các lần chạy DEV, không chứng minh mọi lần gọi LLM đều cho kết quả giống nhau.

### 3.5. Kết quả TEST chi tiết

TEST gồm 7 câu và chỉ chạy sau khi chọn cấu hình.

| Metric | Kết quả |
|---|---:|
| Tổng số câu | 7 |
| Provider Errors | 0 |
| Tool Use | 7/7 |
| Grounded Answer Grade 0 | 1 |
| Grounded Answer Grade 1 | 0 |
| Grounded Answer Grade 2 | 4 |
| Grounded Answer Mean | 1,600/2 |
| Citation Validity | 4/4 |
| Answer Citation Coverage | 4/4 |
| False Abstention | 1/5 |
| Model Calls | 15 |
| Latency trung bình | 6.224 ms/câu |

Các câu q16–q19 trả lời đúng về ZeroDivisionError, IndexError, NameError và TabError với citation hỗ trợ.

q20 là trường hợp false abstention: bằng chứng về việc khối `finally` luôn được thực thi nằm ở trang 234, nhưng không xuất hiện trong context cuối của Agent. Gemini không thể trả lời với citation hợp lệ.

q23, q24 được Agent từ chối phù hợp với nhãn benchmark. Nhãn abstention đúng vẫn chờ xác nhận thủ công toàn bộ bằng chứng.

### 3.6. Đánh đổi và hạn chế

- Hybrid và Neighbor Expansion tăng độ phủ bằng chứng nhưng có thể làm context dài hơn.
- Việc đưa thêm bằng chứng không bảo đảm Agent luôn trả lời đúng.
- Retrieval theo ngữ nghĩa vẫn gặp khó khăn với câu hỏi tiếng Việt và tài liệu tiếng Anh.
- JSON output của Gemini không phải lúc nào cũng hợp lệ.
- Kết quả DEV và TEST phụ thuộc vào câu hỏi, thứ tự tìm kiếm, model và điều kiện API.
- Chi phí USD của Gemini chưa được tổng hợp thành số liệu đáng tin cậy; báo cáo số lần gọi model thay thế.
- Các chỉ số abstention chính thức cần biên bản xác nhận nhãn unanswerable.



### 3.7. So sanh Agent Baseline va Hybrid tren TEST

| Chi so | TF-IDF Baseline | Hybrid + Seed-First |
|---|---:|---:|
| Tool use | 7/7 | 7/7 |
| Grounded Answer Mean | 1.600/2 | 1.600/2 |
| Citation Validity | 4/4 | 4/4 |
| False Abstention | 1/5 | 1/5 |
| Provider errors | 0 | 0 |
| Model calls | 14 | 15 |
| Average latency | 15305.09 ms | 6223.53 ms |

Both configurations answered q16-q19 and abstained on q20.
The q20 abstention is incorrect because relevant evidence exists.
For q23 and q24, abstention behavior was observed, but corpus-wide
absence confirmation remains provisional.

Agent latency includes Gemini API response time; these results
do not establish that one retrieval method is inherently faster.

Evidence: `submission_evidence/agent/baseline_test_metrics.json`
and `submission_evidence/agent/baseline_test_reviews_provisional.json`.

<!-- LAB07_AGENT_BASELINE_COMPARISON -->

## 4. Thử thách nâng cao — Bonus Multimodal RAG V6

<!-- BONUS V6 FINAL REPORT -->

### 4.1. Mục tiêu và dữ liệu

Phần Bonus mở rộng Lab 07 từ truy xuất đoạn văn sang
Retrieval-Augmented Generation trên nhiều dạng nội dung
trong sách *Introduction to Computer Programming with
Python* của Harris Wang (2023).

- Nguồn dữ liệu: `python_wang_2023`.
- Corpus V6: 1.006 records.
- Vector store: ChromaDB, collection `bonus_multimodal_v6`.
- Embedding: MiniLM, 384 chiều.
- Kiểu record gồm text, table, image, diagram và chart.
- Mỗi record chứa ID, source, trang PDF và nội dung truy xuất.
- Các hình/sơ đồ/biểu đồ được biểu diễn bằng caption mô tả
  dựa trên việc kiểm tra thủ công các trang PDF.

Hệ thống **không sử dụng vision model để trực tiếp phân tích
pixel ảnh**. Kết quả với nhóm hình ảnh vì vậy phản ánh khả
năng truy xuất và suy luận trên caption được biên soạn,
không chứng minh năng lực Visual Question Answering
end-to-end.

### 4.2. Retrieval và Context Engineering

Cấu hình Bonus sử dụng:

1. TF-IDF để tìm kiếm theo từ khóa.
2. Dense Retrieval dùng MiniLM và ChromaDB.
3. Hybrid Retrieval kết hợp hai phương pháp bằng RRF.
4. Routing theo loại thông tin được suy ra từ truy vấn.
5. Top-k = 5, candidate depth = 10 trong công cụ Bonus.

Benchmark DEV V4 (sử dụng với corpus V6) có 21 câu,
gồm 18 câu answerable và 3 câu unanswerable.

| DEV Retrieval | All Evidence Hit@5 |
|---|---:|
| Hybrid gốc | 11/18 (61,1%) |
| Hybrid + Routing | 18/18 (100,0%) |

Sau routing, All Evidence Hit@3 đạt 15/18 (83,3%).
Ba câu unanswerable không được tính vào mẫu số
Retrieval Hit@k.

Kết quả này đo khả năng tìm đủ Gold Evidence của
Retrieval, không phải Answer Accuracy của Agent.

### 4.3. Benchmark TEST V6 đã đóng băng

Bộ TEST được đóng băng trước khi chạy Agent chính thức
và không bị chỉnh sửa theo kết quả TEST.

| Loại câu | Số lượng |
|---|---:|
| Table | 6 |
| Diagram | 5 |
| Chart | 5 |
| Multi-hop | 4 |
| Text | 4 |
| Unanswerable | 6 |
| **Tổng** | **30** |

Có 24 câu answerable và 6 câu unanswerable.
Các câu answerable được gán Gold Evidence theo record ID,
trang PDF và loại nội dung.

File benchmark:
`bonus/output/benchmark_test_30_v6_FROZEN.json`

SHA-256:

`8afea6f54074f6dadf2559e985f6db52d8d9ed33b48eb2bf400d256868bf52fc`

Việc kiểm tra tự động không phát hiện lỗi cấu trúc
hoặc cảnh báo trùng lặp theo các tiêu chí đã triển khai.
Tuy nhiên, sáu nhãn unanswerable vẫn ở trạng thái
**provisional**: không tìm thấy bằng chứng qua quá trình
rà soát hiện có, nhưng chưa có xác minh độc lập toàn diện
về sự vắng mặt thông tin trong tài liệu.

### 4.4. Đánh giá Agent trên DEV

Model: `gemini-3.5-flash-lite`.

Agent có thể tìm kiếm tối đa hai lần cho mỗi câu,
sau đó trả lời kèm citation hoặc abstain khi
không đủ bằng chứng.

Lượt DEV ban đầu gặp lỗi 429 RESOURCE_EXHAUSTED
do giới hạn Gemini API Free Tier. Lỗi vận hành này
được ghi nhận riêng, không tính thành lỗi kiến thức.

Một lượt DEV khác được chạy với cơ chế pacing
và retry, lưu tách biệt với lượt đầu.

| DEV Agent | Kết quả |
|---|---:|
| Tổng số câu | 21 |
| Answered | 17 |
| Abstained | 4 |
| API errors | 0 |
| Đúng trong 18 câu answerable | 17/18 (94,4%) |
| Abstain đúng theo nhãn tạm thời | 3/3 |
| Tổng đúng sơ bộ | 20/21 (95,2%) |

Trường hợp thất bại đáng chú ý là `bonus_dev_007`:
Agent abstain khi được hỏi về xu hướng dân số trong
Figure 8-6, dù record liên quan đã có trong evidence.

Kết quả này cho thấy việc truy xuất được bằng chứng
không đồng nghĩa Agent sẽ quyết định trả lời.

### 4.5. Kết quả TEST chính thức

TEST V6 được chạy với benchmark đã đóng băng,
sử dụng runner có giới hạn tốc độ và retry API.
Kết quả lượt TEST đầu tiên được lưu lại,
không điều chỉnh Agent dựa trên câu trả lời TEST.

| Chỉ số TEST | Kết quả |
|---|---:|
| Tổng số câu | 30 |
| Answered | 22 |
| Abstained | 8 |
| API errors | 0 |
| Answer Accuracy (answerable) | 20/24 (83,3%) |
| Citation hỗ trợ câu trả lời đúng | 19/20 (95,0%) |
| Abstention đúng theo nhãn tạm thời | 6/6 |
| Tổng đúng sơ bộ | 26/30 (86,7%) |

Các chỉ số trên được tính từ bản chấm thủ công
lưu riêng sau lượt chạy TEST. Chỉ số 26/30 phụ
thuộc vào việc chấp nhận sáu nhãn unanswerable
đang ở trạng thái provisional.

**TEST Answer Accuracy là 83,3%, chưa đạt ngưỡng 90%
nếu xét riêng 24 câu có đáp án.** Không được suy diễn
từ DEV Retrieval 100% thành TEST Accuracy 100%.

### 4.6. Phân tích lỗi TEST

Có bốn câu answerable chưa đạt:

- `bonus_v4_test_table_01`: không trả lời được
  từ khóa `break` dù câu hỏi có đáp án trong tài liệu.
- `bonus_v4_test_table_02`: không đưa ra được
  kết quả liên quan đến `license()`.
- `bonus_v4_test_multi_hop_01`: abstain thay vì
  tính ra kết quả `12`.
- `bonus_v4_test_multi_hop_03`: abstain thay vì
  kết hợp bằng chứng về `Canvas` và
  `create_rectangle`.

Ngoài ra, câu `bonus_v4_test_text_02` trả lời
đúng kiểu `complex`, nhưng citation được đánh giá
là chưa hỗ trợ trực tiếp cho phép chia.

Các lỗi trên cho thấy còn tồn tại tình huống
Retrieval bỏ sót Gold Evidence, hoặc Agent thiếu
căn cứ để đưa ra kết luận dù có thông tin liên quan.

### 4.7. Hạn chế và khả năng tái lập

- Caption hình ảnh được kiểm tra và biên soạn thủ công;
  hệ thống không triển khai mô hình thị giác.
- Chất lượng OCR/table extraction có thể ảnh hưởng
  đến độ chính xác và xếp hạng truy xuất.
- Routing dựa trên từ khóa trong truy vấn, chưa phải
  bộ phân loại loại tài liệu được học tự động.
- Các câu hỏi về cùng một figure có thể sử dụng
  chung Gold Evidence; vì vậy không hoàn toàn
  độc lập về nguồn.
- Nhãn unanswerable chưa được xác minh độc lập
  toàn diện trên toàn bộ tài liệu.
- Điểm số phụ thuộc benchmark, model và điều kiện API.
- Việc chạy lại LLM có thể cho kết quả khác;
  không sử dụng TEST để tối ưu rồi chấm lại như
  một lượt TEST độc lập.

Các tệp quan trọng:

- `bonus/hybrid_index_v6.py`
- `bonus/eval_dev_v6.py`
- `bonus/run_v6_agent.py`
- `bonus/run_v6_agent_rate_limited.py`
- `bonus/output/benchmark_dev_21_v4.json`
- `bonus/output/benchmark_test_30_v6_FROZEN.json`
- `bonus/output/v6/dev_retrieval_first_run.json`
- `bonus/output/v6/test_freeze_manifest.json`
- `bonus/output/v6/agent_test_first_run/summary.json`
- `bonus/output/v6/agent_test_first_run/human_scoring_summary.json`
- `bonus/output/v6/agent_test_first_run/human_review_scored.csv`

Cần cấu hình `.env`, truy cập được nguồn PDF
và ChromaDB/embedding tương ứng để tái lập
đầy đủ các lượt chạy. Không đưa API key vào GitHub.

## Appendix - Bonus V6 Strict Scoring

<!-- BONUS_V6_STRICT_SCORING -->

| Metric | Result |
|---|---:|
| Correct answers on answerable questions | 20/24 (83.3%) |
| Correct answers with supported citations | 19/24 |
| Correct abstentions (provisional labels) | 6/6 |
| Preliminary total correctness | 26/30 (86.7%) |
| Preliminary strict pass rate | 25/30 (83.3%) |

Strict pass requires both a correct answer and a
supporting citation.

The six unanswerable labels remain provisional.

The 90% threshold requires at least 27/30 passes.
The current TEST V6 results do not establish
eligibility for the additional 15 bonus points.

The first TEST run is preserved without
post-TEST tuning.

## 5. Khả năng tái lập

Source code chính:

- `run_gemini_agent.py`: chạy Agent với Gemini.
- `src/day6_retriever.py`: kết nối dữ liệu và vector store Lab 06.
- `src/dense_eval.py`: đánh giá Dense Retrieval.
- `src/hybrid_depth_eval.py`: đánh giá Hybrid RRF.
- `src/seed_first_agent.py`: Context Engineering và Agent Loop.
- `eval_dev_agent.py`: đánh giá DEV.
- `eval_test_seedfirst.py`: chạy TEST holdout.

Kết quả được lưu cục bộ tại:

- `starter/results/agent-dev-seedfirst-final/`
- `starter/results/agent-test-seedfirst-final/`
- `starter/results/hybrid-experiments/`
- `starter/results/final_context_metrics.json`

Cần có corpus, ChromaDB và mô hình embedding tương ứng từ Lab 06. Không commit API key hoặc `.env`.

## 6. Kết luận

Lab 07 đã triển khai Agentic RAG kết hợp Hybrid Retrieval và Seed-First Context Engineering. Các thí nghiệm DEV cho thấy Context Hit cải thiện từ 7/15 lên 11/15 khi thay đổi chiến lược đóng gói và tăng context budget.

Trên TEST, hệ thống đạt Grounded Answer Mean 1,60/2 và 4/4 citation được đánh giá là có căn cứ. Hạn chế rõ nhất là Retrieval vẫn có thể bỏ sót evidence dù tài liệu có câu trả lời.

Hướng cải thiện sau bài lab gồm truy vấn song ngữ, tăng khả năng tìm candidate phù hợp và đánh giá định lượng riêng cho độ trễ/chi phí khi mở rộng context.

Báo cáo giữ nguyên kết quả TEST lần đầu, không điều chỉnh thuật toán theo TEST.

Phần Bonus V6 mở rộng sang bảng, sơ đồ, biểu đồ và câu hỏi multi-hop. DEV Retrieval đạt 18/18 All Evidence Hit@5; TEST đạt 20/24 câu answerable đúng (83,3%) và 19/20 citation hỗ trợ câu trả lời đúng. Kết quả abstention vẫn dựa trên các nhãn chưa được xác minh độc lập toàn diện.

<!-- LAB07_FORM_TABLES_V1 -->
