Họ tên: Ngô Gia Quốc
`dataset_version`: d10-21e4070301

Môi trường chạy: Kaggle Notebook, GPU T4 (dùng 1 card), transformers 5.19.0 · peft 0.21.2 · bitsandbytes 0.50.2. Toàn bộ E0–E4 chạy trong một lần "Save & Run All" (version 357955219), tổng 4 giờ 26 phút.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – | – | 0.086 | 0.062–0.115 | 0.90 | 0.110 | 0.024 | 0.203 |
| E1 | LoRA LLM | 33,030,144 | 8.05 | 43.2 | 0.269 | 0.232–0.310 | 1.19 | 0.182 | 0.192 | 0.513 |
| E2 | + projector | 142,135,296 | 8.54 | 43.6 | 0.349 | 0.307–0.394 | 1.04 | 0.220 | 0.262 | 0.636 |
| E3 | + LoRA ViT | 148,426,752 | 8.73 | 51.5 | **0.447** | 0.407–0.490 | 0.86 | 0.326 | 0.366 | 0.696 |
| E4 | coord_scale = norm1000 (chính là E3, chấm lại) | 148,426,752 | 8.73 | 51.5 | 0.447 | 0.407–0.490 | 0.86 | 0.326 | 0.366 | 0.696 |
| E4 | coord_scale = pixel (E3, đổi nhãn + prompt + `--coord pixel`) | 148,426,752 | 8.70 | 51.0 | 0.392 | 0.350–0.435 | 0.94 | 0.274 | 0.304 | 0.625 |

`parse_valid_rate`: E0 0.926 · E1 0.994 · E2, E3, E4 1.000. Mean IoU của các cặp khớp: 0.603 (E0) → 0.677 (E1) → 0.682 (E2) → 0.693 (E3).
VRAM đỉnh đo bằng `torch.cuda.max_memory_allocated()` từ lúc nạp model đến hết 300 bước; thời gian chỉ tính vòng huấn luyện (chưa tính suy luận test ≈ 15 phút mỗi setup).

Ảnh minh hoạ: `report/figures/e0_examples.png`, `e1_examples.png`, `e3_examples.png` (mỗi ảnh 3 dự đoán tốt nhất + 3 tệ nhất theo F1 từng ảnh; GT nét đứt xanh lá, dự đoán nét liền theo màu lớp).

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B ≈ 4,44B tham số: ViT ≈ 415M (24 block, hidden 1024), trong đó projector gồm `visual.merger` ≈ 27M và `visual.deepstack_merger_list` ≈ 82M (3 merger lấy đặc trưng ở lớp 5, 11, 17 của ViT để "bơm" thẳng vào các lớp đầu của LLM); phần LLM + embedding ≈ 4,0B (36 lớp, hidden 2560).

Số tôi đo được từ chính các lần chạy:
- **E1** chỉ cập nhật LoRA (r=16) trên 252 module của LLM (`q,k,v,o,gate,up,down` × 36 lớp) = **33,03M** tham số. ViT và projector đóng băng.
- **E2** = E1 + mở toàn bộ projector: 142,14M − 33,03M = **109,11M** tham số projector (khớp 27M + 82M).
- **E3** = E2 + LoRA trên 96 module của ViT (`attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2` × 24 block): 148,43M − 142,14M = **6,29M** tham số LoRA ViT. Tổng 348 module LoRA.

Như vậy E1 → E2 tăng số tham số gấp 4,3 lần nhưng VRAM chỉ tăng 0,5 GB; E2 → E3 chỉ thêm 4% tham số nhưng tăng 8 phút vì gradient phải đi ngược qua cả 24 block ViT.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` = **0.926** (288/311 ảnh parse được). Model gốc không bọc ```json (0/311 output có fence) và hiểu đúng thang 0–1000 (toạ độ lớn nhất 1000). Hai lỗi chính:

1. **Lặp vô hạn rồi bị cắt → JSON hỏng.** Cả 23 output hỏng đều cùng một kiểu: model "rải" một vết nứt thành chuỗi box crack nhỏ nối tiếp nhau cho tới khi hết `max_new_tokens=256`, nên JSON không có `]` đóng. Ví dụ thô (đã cắt):
   `[{"label": "crack", "bbox_2d": [575, 420, 600, 440]}, {"label": "crack", "bbox_2d": [575, 440, 600, 460]}, {"label": "crack", "bbox_2d": [575, 460, 600, 480]}, …`
2. **Bịa box / sai vị trí và sai lớp.** Trong 320 box hợp lệ, 203 box là crack nhưng F1 crack chỉ 0.024 (P = 0.030); 25 box chiếm > 25% khung hình trong khi box GT trung vị chỉ ≈ 0,7% diện tích. Model nhận ra *có* hư hại (mean IoU của các cặp khớp vẫn 0.60) nhưng không biết quy ước bao box của bộ dữ liệu này. Ví dụ output hợp lệ nhưng sai: `[{"label": "manhole", "bbox_2d": [770, 640, 843, 684]}]`. Kết quả: P = 0.122, R = 0.067, F1 = 0.086, 0.90 box sai trên mỗi ảnh.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Tôi lấy đúng file dự đoán của E1 (model trả toạ độ 0–1000) rồi chấm lại bằng `--coord pixel` (`work/e1_as_pixel.json`): **F1 từ 0.269 rơi về 0.000** (P = R = 0, toàn bộ 519 box thành FP, FP/ảnh 1.19 → 1.67), dù `parse_valid_rate` vẫn 0.994. Lý do: khi coi số là pixel, trọng tài đổi `x·1000/640` và `y·1000/360`, tức kéo box ra 1,56 lần theo chiều ngang và 2,78 lần theo chiều dọc, đồng thời đẩy box xuống dưới/sang phải. Với box nhỏ như ở đây, IoU với GT gần như luôn < 0.5, nên không còn TP nào. Đây là lỗi "im lặng": JSON hợp lệ, số trông hợp lý, chỉ F1 bằng 0.

Chiều ngược lại tôi kiểm chứng bằng E4: huấn luyện hẳn với nhãn và prompt ở thang pixel (chấm `--coord pixel`) cho F1 = 0.392, thấp hơn 0.447 của norm1000 (xem Câu 10). Qwen3-VL được tiền huấn luyện grounding trên thang 0–1000 tương đối, không phụ thuộc kích thước ảnh, nên giữ thang này thì tận dụng được kiến thức có sẵn; đổi sang pixel buộc model học lại một quy ước mới chỉ với 300 bước. Tôi cũng kiểm tra round-trip ở bước 2: YOLO → JSON 0–1000 → parser của trọng tài → pixel khớp box gốc với sai số ≤ 1 px.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Mỗi ví dụ có 220 token ảnh + khoảng 90 token prompt/template, trong khi đáp án chỉ khoảng 20 token cho mỗi box (ảnh mẫu 9 box → chuỗi đầy đủ 594 token). Tôi gán `labels = -100` cho toàn bộ tiền tố (đã assert prompt là tiền tố đúng của chuỗi đầy đủ) và kiểm tra: giải mã riêng các token có nhãn ra đúng chuỗi JSON đáp án + `<|im_end|>`.

Nếu không mask:
- **Token ảnh** là `<|image_pad|>` lặp lại 220 lần: dự đoán chúng vô nghĩa (đầu vào thật là `pixel_values`), nhưng chúng chiếm phần lớn chuỗi, nên loss và gradient bị chi phối bởi việc học lặp token đệm.
- **Prompt** giống hệt nhau ở mọi ví dụ, nên model nhanh chóng thuộc lòng nó; loss nhìn thì giảm đẹp nhưng phần lớn là do học thuộc prompt, che mất tín hiệu của phần cần học là toạ độ và nhãn.
- Kết quả là tín hiệu học bị loãng đi nhiều lần trong cùng 300 bước, và model có thể học cả việc "nói" như user. Với mask, toàn bộ gradient dồn vào đúng hành vi cần học; loss của E1 giảm từ 1.28 (bước 10) xuống 0.56 (bước 300).

Tôi không chạy thử biến thể không mask (hết ngân sách GPU), nên phần này là lập luận chứ không có số đo riêng.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Lớp `Linear4bit` lưu trọng số dưới dạng mã NF4 đã nén (2 giá trị 4-bit trong 1 byte `uint8`) kèm hằng số scale theo khối. Đó không phải tham số liên tục nên không có gradient đúng nghĩa và AdamW không cập nhật trực tiếp được; chỉ có thể gắn adapter (LoRA) lên trên. Muốn "mở đầy đủ" projector thì trọng số của nó phải là tensor thường.

**Tôi đã gặp đúng lỗi này.** Cấu hình theo GUIDE là `llm_int8_skip_modules=["visual", "lm_head"]`, nhưng trong transformers 5.19 tên module được so khớp từ gốc, nên `"visual"` không khớp `model.visual.merger…` và **toàn bộ ViT kể cả projector vẫn bị lượng tử hoá**. Triệu chứng duy nhất là con số: lần chạy thử E3 báo 93,9M tham số huấn luyện thay vì ≈ 148M. Phần hụt (≈ 54,6M) đúng bằng một nửa 109M, vì tensor `uint8` nén chỉ có n/2 phần tử. Khi đó `p.data.float()` biến các byte nén thành số thực vô nghĩa và E2/E3 thực chất huấn luyện trên rác. Cách sửa: đổi thành `llm_int8_skip_modules=["model.visual", "lm_head"]` và thêm assert trong `load_base()` rằng `visual.merger.linear_fc1` là `nn.Linear` fp16. Sau khi sửa, E3 có đúng 148.426.752 tham số huấn luyện.

Projector phải ở **float32** vì:
1. Huấn luyện fp16 dùng `GradScaler`, mà `unscale_()` từ chối gradient fp16 (tôi gặp đúng lỗi "Attempting to unscale FP16 gradients").
2. Cập nhật AdamW với lr ≤ 2e-4 thường nhỏ hơn bước lượng tử của fp16 quanh giá trị trọng số, nên cập nhật bị làm tròn mất. Bản master fp32 giữ được các bước nhỏ này; trong forward, autocast vẫn tính ở fp16.

Thêm một bẫy của torch mới: `p.data = p.data.float()` đổi dtype nhưng `p.grad_dtype` vẫn là fp16, nên grad của bias projector vẫn ra fp16. Tôi sửa bằng cách đặt `p.grad_dtype = torch.float32` cho các tham số projector.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Ảnh là frame cắt từ video dashcam, nên hai frame liền nhau gần như trùng khít (cùng ổ gà, cùng nắp cống, chỉ lệch vài pixel). Chia ngẫu nhiên thì gần như mọi ảnh test đều có "anh em sinh đôi" trong train. Model chỉ cần nhớ vị trí box của từng đoạn đường là đạt F1 cao, trong khi đó là đo **khả năng nhớ** chứ không phải khả năng tổng quát sang đoạn đường mới. `prepare_data.py` gom ảnh theo phiên quay (ngày + giờ trong tên file) và gán cả phiên vào một split; tôi đã assert không phiên nào xuất hiện ở hai split.

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1.422 | 1.074 | 1.750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

Hệ quả của việc chia theo phiên là phân bố lớp lệch: val chỉ có 66 pothole, test có 121. Tôi không chạy thử chia ngẫu nhiên nên không có con số "F1 giả" để so sánh.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
| | F1 | Δ F1 | Δ VRAM | Δ thời gian |
|---|---|---|---|---|
| E1 → E2 (+ projector) | 0.269 → 0.349 | +0.080 | +0.49 GB | +0.4 phút |
| E2 → E3 (+ LoRA ViT) | 0.349 → 0.447 | +0.098 | +0.19 GB | +7.9 phút |
| E1 → E3 | 0.269 → 0.447 | +0.178 | +0.68 GB | +8.3 phút |

- **Mở projector gần như miễn phí**: thêm 109M tham số fp32 nhưng chỉ tốn thêm 0,5 GB và không tốn thêm thời gian, vì gradient chỉ dừng ở projector chứ không đi vào ViT. Tuy vậy CI95 của E1 (0.232–0.310) và E2 (0.307–0.394) vẫn chồng nhau một đoạn nhỏ, nên riêng bước này chưa đủ chắc chắn ở mức 95%.
- **LoRA trên ViT mang lại cải thiện lớn nhất**: cải thiện đồng đều cả 3 lớp (pothole 0.220 → 0.326, crack 0.262 → 0.366, manhole 0.636 → 0.696) và giảm FP/ảnh từ 1.04 xuống 0.86. CI95 của E2 và E3 (0.407–0.490) không chồng nhau. Chi phí là thêm 18% thời gian, do backprop phải đi qua 24 block ViT.
- E3 hơn E1 rõ ràng: hai khoảng CI cách nhau gần 0,1.

Kết luận: **E3 đáng đồng tiền nhất** trong bài này. VRAM đỉnh 8,7 GB vẫn dư nhiều so với 15 GB của T4. Với bài toán box nhỏ và đặc trưng thị giác chuyên biệt (vết nứt trên mặt đường), việc cho encoder thị giác thích nghi quan trọng hơn chỉ chỉnh phần ngôn ngữ.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 bootstrap (theo ảnh, 1000 lần lấy mẫu) rộng khoảng **±0,04 quanh F1**: E1 0.232–0.310 (rộng 0.078), E2 0.307–0.394 (0.087), E3 0.407–0.490 (0.083), E4-pixel 0.350–0.435 (0.085). Hệ quả:
- **Kết luận đứng vững**: E0 < E1 < E3, và E2 < E3 (các khoảng tách rời nhau).
- **Kết luận chưa chắc**: E1 so với E2 (chồng nhau ở 0.307–0.310) và E3 so với E4-pixel (chồng nhau ở 0.407–0.435). Chênh lệch khoảng 0,05–0,08 F1 ở đây nằm sát ngưỡng nhiễu.
- **Theo từng lớp còn kém tin cậy hơn**: test chỉ có 121 pothole, nên F1 pothole 0.326 của E3 chỉ dựa trên 31 TP.

CI thật có lẽ còn rộng hơn con số trên, vì bootstrap theo ảnh giả định các ảnh độc lập, trong khi 311 ảnh test đến từ ít phiên quay và các frame trong cùng phiên tương quan mạnh với nhau. Bootstrap theo phiên sẽ trung thực hơn. Ngoài ra mỗi setup chỉ chạy 1 seed (42), nên chưa tính được phần biến thiên do huấn luyện.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Với E3, **pothole kém nhất** (F1 0.326) và crack chỉ hơn một chút (0.366); manhole tốt nhất (0.696).

- **Pothole (P 0.449, R 0.256)**: lỗi chủ yếu là **bỏ sót** (90 FN so với 38 FP). Ổ gà thường nhỏ, ở xa camera, dễ lẫn với vết vá, bóng cây hay vũng nước. Train có 1.074 box pothole nhưng tập trung trong ít phiên quay, nên đặc trưng ít đa dạng. Model chỉ dự đoán 69 box pothole cho 121 GT, tức thận trọng quá mức.
- **Crack (P 0.358, R 0.375)**: lỗi chủ yếu do **ranh giới mơ hồ**. Vết nứt dài và mảnh, không có điểm đầu cuối rõ ràng, nên cùng một vết nứt có thể được gán nhãn thành 1 box lớn hoặc vài box nhỏ. Hàng dưới của `report/figures/e3_examples.png` cho thấy đúng hai kiểu lỗi này: (a) model bao một đoạn ngắn trong box GT dài nên IoU < 0.5 → vừa FP vừa FN; (b) model tách một vết nứt thành 3 box (FP = 3, FN = 1). E3 dự đoán 313 box crack nhưng có tới 201 FP. Model gốc cũng gặp vấn đề tương tự (Câu 2: E0 lặp vô hạn chuỗi box crack).
- **Manhole (P 0.785, R 0.626)**: hình dạng đều (chữ nhật hoặc tròn, mép kim loại rõ), kích thước ổn định, nên dễ nhất. Có một ca nhầm manhole thành crack khi nắp cống nằm trong vùng mặt đường nứt (khung giữa hàng dưới của ảnh E3).

So sánh `e0_examples.png` → `e1_examples.png` → `e3_examples.png` cho thấy box ngày càng bám sát GT; ở E0, ngay cả các ảnh "tốt nhất" cũng chỉ trúng những box manhole lớn.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Tôi đổi **đúng một biến: thang toạ độ** (`coord_scale`), giữ nguyên setup tốt nhất E3 (LoRA LLM + projector + LoRA ViT, r=16, 300 bước, seed 42). Giá trị `pixel` đổi cùng lúc nhãn (toạ độ nguyên theo pixel ảnh 640×360), prompt ("integer pixel coordinates…") và cách chấm (`--coord pixel`). Giá trị `norm1000` chính là lần chạy E3, được chấm lại thành `e4_coord_scale_norm1000.json`, như GUIDE cho phép.

| coord_scale | F1 | CI95 | F1 pothole / crack / manhole | Mean IoU |
|---|---|---|---|---|
| norm1000 | 0.447 | 0.407–0.490 | 0.326 / 0.366 / 0.696 | 0.693 |
| pixel | 0.392 | 0.350–0.435 | 0.274 / 0.304 / 0.625 | 0.682 |

Thang pixel kém hơn 0.055 F1, và **cả ba lớp đều giảm** (0.05–0.07 mỗi lớp). Tài nguyên gần như giống hệt (8,70 GB, 51 phút), nên khác biệt đến từ cách biểu diễn toạ độ chứ không phải từ năng lực tính toán. Giải thích hợp lý: thang 0–1000 khớp với quy ước grounding lúc tiền huấn luyện của Qwen3-VL, còn thang pixel bắt model học lại một ánh xạ mới, lại không đẳng hướng (x tới 640, y tới 360).

**Mức tin cậy**: hai khoảng CI95 chồng nhau ở đoạn 0.407–0.435, nên ở mức 95% tôi **chưa thể khẳng định** pixel kém hơn. Hướng giảm nhất quán ở cả 3 lớp là bằng chứng gợi ý, không phải kết luận chắc chắn. Muốn chắc chắn cần chạy nhiều seed, hoặc dùng tập test lớn hơn và bootstrap theo phiên. Kết luận an toàn: không có lý do để rời thang 0–1000 mặc định.

## Nhìn lại

- **Ba lỗi môi trường mà công cụ chấm không bắt được**, vì cả ba đều không làm hỏng JSON hay file bằng chứng:
  1. `llm_int8_skip_modules=["visual"]` không có tác dụng trong transformers 5.19, nên projector bị lượng tử hoá 4-bit mà không có lỗi nào báo ra. Nếu không để ý con số tham số (93,9M thay vì 148M), E2/E3 vẫn chạy và vẫn sinh ra evidence "hợp lệ". Tôi rút ra bài học: luôn đối chiếu số tham số huấn luyện với ước tính tính tay.
  2. Torch mới giữ `grad_dtype = fp16` sau khi đổi `.data` sang fp32, khiến GradScaler báo lỗi.
  3. Image mới nhất của Kaggle có sẵn `torchao 0.10`, khiến peft 0.21 báo `ImportError` khi tạo LoRA. Phiên tương tác dùng image được ghim nên không gặp, chỉ bản chạy commit mới lộ ra. Tôi xử lý bằng cách gỡ torchao.

  Smoke test (2 bước với setup nặng nhất E3) chạy trước khi train thật đã bắt được cả ba lỗi.
- **Lỗi dữ liệu**: 1 nhãn train bị parser của trọng tài từ chối vì box quá nhỏ, sau khi làm tròn về thang 0–1000 thì x1 = x2 (suy biến). Lỗi này không ảnh hưởng đáng kể nhưng cho thấy độ phân giải 0–1000 không đủ cho những box rất nhỏ.
- **Điều tôi chưa hiểu hết**: vì sao chỉ *bias* của projector bị kẹt `grad_dtype` fp16 còn weight thì không. Tôi cũng chưa định lượng được tác động của việc không mask loss, vì không chạy thử biến thể đó.
- **E5 (GRPO)**: mới viết hàm reward IoU và 6 test (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng, box khổng lồ); test đã pass trên Kaggle. Chưa chạy GRPO nên không nhận phần điểm "có log" và "phân tích reward hacking".
