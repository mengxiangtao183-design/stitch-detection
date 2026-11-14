# 纭曞＋璁烘枃娴佺▼妗嗗浘鑽夌

本文档只保留论文画图相关的主流程；检测主流程使用清晰图像或 MPRNet 复原图像。
## 鍥?1锛氭€讳綋鎶€鏈矾绾?
```mermaid
flowchart LR
    A[鏁版嵁闆?DB / XJ / SD / RX] --> B{杈撳叆鍥惧儚鏉ユ簮}
    B -->|閫€鍖栧浘鍍?Imgs1| C[Deblurring / MPRNet 鍥惧儚澶嶅師]
    C --> D[澶嶅師鍥惧儚 clear 鎴?Imgs1_MPRNet]
    B -->|宸叉湁娓呮櫚鍥惧儚 clear| D

    D --> E[妫€娴嬭緭鍏ユ瀯寤?br/>RGB 璇诲彇<br/>Resize 320 x 320<br/>ImageNet 褰掍竴鍖朷
    F[GT 浜屽€兼帺鑶淽 --> G[鐩戠潱淇℃伅]
    H[GT 杈圭紭鍥?Edges] --> G

    E --> I[C3Net 鍒嗗壊妫€娴嬫ā鍨媇
    G --> J[璁粌鎹熷け / 璇勪及鎸囨爣]
    I --> K[杈撳嚭缁撴灉<br/>final_mask<br/>object_mask<br/>edge_mask]
    K --> L[缁撴灉鍙鍖?br/>浜屽€煎浘 / 鐑姏鍥?/ Overlay]
    K --> M[瀹氶噺璇勪环<br/>S_alpha / F_beta_w / MAE / E_phi / F_beta_m]
```

## 鍥?2锛氬浘鍍忎粠杈撳叆鍒拌緭鍑虹殑妫€娴嬫祦绋?
```mermaid
flowchart TD
    A[杈撳叆鍥惧儚 clear] --> B[鍥惧儚棰勫鐞?br/>RGB 杞?Tensor<br/>Resize 鍒?320 x 320<br/>ImageNet Normalize]
    B --> C[MobileNetV4-Hybrid-Medium Backbone]

    C --> C2[C2 鐗瑰緛<br/>80 x 80, 48 ch]
    C --> C3[C3 鐗瑰緛<br/>40 x 40, 80 ch]
    C --> C4[C4 鐗瑰緛<br/>20 x 20, 160 ch]
    C --> C5[C5 鐗瑰緛<br/>10 x 10, 960 ch]

    C2 --> D[1 x 1 Lateral Projection]
    C3 --> D
    C4 --> D
    C5 --> D

    D --> E[Mobile-FPN 鑷《鍚戜笅铻嶅悎<br/>P5 -> P4 -> P3 -> P2]

    E --> F[杈圭紭缁嗗寲鍒嗘敮<br/>P2 + 涓婇噰鏍?P3<br/>EdgeEnhanceBlock]
    E --> G[涓婁笅鏂囧畾浣嶅垎鏀?br/>P4 璇箟澧炲己<br/>ICG 鍥惧儚涓婁笅鏂囧紩瀵糫
    B --> G

    F --> H[杈圭紭鐗瑰緛]
    G --> I[瀹氫綅鐗瑰緛]

    H --> J[Attentive Fusion Head<br/>瀹氫綅鐗瑰緛鐢熸垚绌洪棿闂ㄦ帶<br/>绛涢€夎竟缂樼壒寰乚
    I --> J

    H --> K[High-Resolution Refine<br/>铻嶅悎杈圭紭 / 瀹氫綅 / RGB 鍥惧儚鐗瑰緛]
    I --> K
    B --> K
    J --> K

    K --> L[final_mask<br/>鏈€缁堝垎鍓插浘]
    I --> M[object_mask<br/>瀹氫綅杈呭姪鍥綸
    K --> N[edge_mask<br/>杈圭紭棰勬祴鍥綸
```

## 鍥?3锛欳3Net 璁粌娴佺▼

```mermaid
flowchart TD
    A[璁粌闆?br/>DB / XJ / SD] --> B[璇诲彇 train/clear]
    A --> C[璇诲彇 train/GT]
    A --> D[璇诲彇 train/Edges]

    B --> E[鍥惧儚棰勫鐞?br/>Resize 320 x 320<br/>Normalize]
    C --> F[GT 鎺╄啘<br/>淇濇寔鍘熷灏哄]
    D --> G[GT 杈圭紭<br/>淇濇寔鍘熷灏哄]

    E --> H[Student C3Net 鍓嶅悜浼犳挱]
    H --> I[棰勬祴杈撳嚭<br/>final_mask / object_mask / edge_mask]
    I --> J[鎻掑€煎埌 GT 鍘熷灏哄]

    F --> K[C3NetLoss]
    G --> K
    J --> K

    E --> L[鍙€?DINOv2 Teacher<br/>鐭ヨ瘑钂搁淇″彿]
    L --> M[KD Loss]
    I --> M

    K --> N[妫€娴嬫崯澶?br/>Final + Loc + Edge<br/>Topology + Boundary]
    M --> O[鎬绘崯澶盷
    N --> O

    O --> P[AMP 娣峰悎绮惧害鍙嶄紶]
    P --> Q[AdamW 浼樺寲<br/>Cosine LR Scheduler<br/>Gradient Clipping]
    Q --> R[淇濆瓨 Checkpoint<br/>璁粌鏃ュ織 / 鎸囨爣]
```

## 鍥?4锛氳瘎浼颁笌棰勬祴娴佺▼

```mermaid
flowchart TD
    A[娴嬭瘯闆?br/>DB / XJ / SD / RX] --> B[璇诲彇 test/clear]
    A --> C[璇诲彇 test/GT<br/>璇勪及妯″紡浣跨敤]

    B --> D[鍥惧儚棰勫鐞?br/>Resize 320 x 320<br/>Normalize]
    D --> E[鍔犺浇璁粌濂界殑 C3Net Checkpoint]
    E --> F[妯″瀷鎺ㄧ悊]
    F --> G[杈撳嚭 logits<br/>final_mask / object_mask / edge_mask]

    G --> H[Sigmoid 姒傜巼鍖朷
    H --> I[鎻掑€煎洖 GT 鎴栧師鍥惧昂瀵竇

    I --> J[淇濆瓨鍙鍖栫粨鏋?br/>Binary / Heatmap / Overlay]
    C --> K[瀹氶噺璇勪环]
    I --> K
    K --> L[鎸囨爣姹囨€?br/>S_alpha / F_beta_w / MAE / E_phi / F_beta_m]
    K --> M[鎸夎川閲忓垎绫?br/>Good / Medium / Bad]
```

## 鍥?5锛欴eblurring 澶嶅師鍓嶅鐞嗘祦绋?
```mermaid
flowchart LR
    A[閫€鍖栬緭鍏ュ浘鍍?Imgs1] --> B[Tensor 鍖?br/>鍙栧€艰寖鍥?0 鍒?1]
    B --> C[鍙€変寒搴﹁嚜閫傚簲鏍℃<br/>Light Adaptive Module]
    C --> D[MPRNet 澶氶樁娈靛鍘焆
    D --> E[Stage 1<br/>鍥涘潡灞€閮ㄥ浘鍍忓鍘焆
    E --> F[Stage 2<br/>涓婁笅鍗婂浘澶嶅師 + SAM]
    F --> G[Stage 3<br/>鍘熷浘灏哄害澶嶅師 + ORSNet]
    G --> H[Residual 杈撳嚭]
    H --> I[Restored = Input + Residual]
    I --> J[淇濆瓨澶嶅師鍥惧儚<br/>clear 鎴?Imgs1_MPRNet]
    J --> K[浣滀负 C3Net 妫€娴嬭緭鍏
```

## 璁烘枃涓缓璁繚鐣欑殑鏍稿績鍥?
濡傛灉绡囧箙鏈夐檺锛屽缓璁鏂囪嚦灏戜繚鐣欙細

1. 鍥?1 鎬讳綋鎶€鏈矾绾匡細璇存槑澶嶅師銆佹娴嬨€佽瘎浠蜂箣闂寸殑鍏崇郴銆?2. 鍥?2 C3Net 妫€娴嬫祦绋嬶細浣滀负妯″瀷缁撴瀯鍥俱€?3. 鍥?3 璁粌娴佺▼锛氳鏄庡杈撳嚭鐩戠潱銆佽竟缂樼洃鐫ｅ拰钂搁銆?4. 鍥?4 璇勪及涓庨娴嬫祦绋嬶細璇存槑瀹為獙缁撴灉濡備綍鐢熸垚銆?
鍥?5 鍙斁鍦ㄢ€滄暟鎹澶勭悊鈥濇垨鈥滃浘鍍忓鍘熸ā鍧椻€濈珷鑺傦紱濡傛灉璁烘枃閲嶇偣涓嶆槸澶嶅師缃戠粶锛屽彲浠ュ皢鍏剁畝鍖栦负涓€涓€滃浘鍍忓鍘熸ā鍧椻€濇柟妗嗐€?
