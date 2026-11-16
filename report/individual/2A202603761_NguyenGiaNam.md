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

## 4. Thử thách nâng cao

Chưa thực hiện đánh giá riêng với bảng, ảnh, biểu đồ và multi-hop. Không báo cáo điểm cho các hạng mục chưa thử nghiệm.

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

<!-- LAB07_FORM_TABLES_V1 -->
