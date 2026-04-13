[References](https://huggingface.co/papers/trending?q=face+anti-spoofing)

# **State of the Art in Liveness Detection and Face Anti-Spoofing: The Paradigm Shift of 2024–2026**

The landscape of biometric security has undergone a profound transformation between 2024 and 2026, driven by the escalating sophistication of presentation attacks and the concurrent evolution of defensive machine learning architectures. Face Anti-Spoofing (FAS), or Presentation Attack Detection (PAD), has traditionally operated as a binary classification problem, where models were trained to distinguish between a "live" human subject and a "spoof" medium such as a photograph or an electronic screen.1 However, the emergence of high-fidelity 3D masks, hyper-realistic generative AI forgeries, and advanced digital injection attacks has rendered simple classification methods insufficient for robust security.3 This report provides an exhaustive synthesis of recent breakthroughs, centering on the transition toward forensic reasoning, the adaptation of vision foundation models, and the integration of multimodal sensory data to create unforgeable liveness signatures.

## **The Reasoning Era: Multimodal Large Language Models and Forensic Interpretability**

The most significant paradigm shift in the 2025–2026 period is the transition from feature-extraction-based classification to forensic reasoning tasks mediated by Multimodal Large Language Models (MLLMs). This "Reasoning Era" addresses the fundamental limitations of traditional "black-box" models, which provide a liveness probability score but fail to explain why a particular sample is considered a spoof.1 By reformulating FAS as an interpretable visual question-answering task, researchers have achieved higher generalization and established a foundation for trustworthy biometric authentication.

### **Interpretable Face Anti-Spoofing (I-FAS) and the VQA Paradigm**

The I-FAS framework represents a departure from standard binary supervision. It transforms the detection process into an Interpretable Visual Question Answering (VQA) paradigm.1 Instead of returning a score between 0.0 and 1.0, the model generates a detailed natural language explanation identifying specific artifacts, such as "interference moiré patterns detected on the screen edge" or "unnatural light absorption characteristics suggestive of a 2D print".1

A core component of the I-FAS architecture is the Spoof-aware Captioning and Filtering (SCF) strategy. SCF is designed to generate high-quality captions that specifically focus on forensic anomalies.1 These captions provide the model with a richer supervisory signal than simple labels. To manage the noise inherent in automated captioning, researchers introduced the Lopsided Language Model (L-LM) loss function.1 This loss function decouples the optimization of the model's judgment (the final decision) from its interpretation (the descriptive reasoning), ensuring that the reasoning process remains grounded in forensic reality while prioritizing classification accuracy.1

Furthermore, the I-FAS system employs a Globally Aware Connector (GAC) to align multi-level visual representations with the language model's latent space.1 This connector ensures that the model can perceive both local forensic clues—such as the micro-texture of a high-resolution print—and global contextual evidence, such as the illumination consistency between the face and its environment.1

### **Tool-Augmented Reasoning (TAR-FAS) and the Investigation Pipeline**

As spoofing techniques become more subtle, even advanced MLLMs may struggle to perceive fine-grained visual patterns through standard visual encoders alone. To bridge this gap, the Tool-Augmented Reasoning (TAR-FAS) framework was introduced in 2026\.5 TAR-FAS drives the model from intuitive observation to deep investigation by allowing it to adaptively invoke external visual tools during the reasoning process.5

The core of TAR-FAS is the Chain-of-Thought with Visual Tools (CoT-VT) paradigm.5 When the model encounters a challenging sample, it can decide to "call" a tool—such as a specialized frequency-domain analyzer or a micro-texture edge detector—to examine specific regions of the image.7 This investigative loop allows the model to refine its initial hypothesis based on concrete evidence provided by traditional computer vision algorithms.

| Tool Category | Specific Tool | Mechanism and Forensic Utility |
| :---- | :---- | :---- |
| **Spectral Analysis** | Fast Fourier Transform (FFT) | Uncovers abnormal high-frequency spectral distributions indicative of print or screen refresh cycles.9 |
| **Texture Analysis** | Local Binary Patterns (LBP) | Captures micro-texture irregularities and halftone dot patterns in printed media.9 |
| **Structural Analysis** | Laplacian Edge Detection | Identifies unnatural structural boundaries or depth discontinuities in replayed videos.9 |
| **Geometric Analysis** | Histogram of Oriented Gradients (HOG) | Exposes reflectance discontinuities and unnatural gradient flows in 3D masks.9 |
| **Local Inspection** | Zoom-In Tool | Enables high-resolution analysis of specific bounding boxes, such as pupils or skin pores.9 |

To train this sophisticated reasoning capability, the ToolFAS-16K dataset was constructed, containing over 16,000 images annotated with multi-turn tool-use reasoning trajectories.5 The model is optimized using Diverse-Tool Group Relative Policy Optimization (DT-GRPO), a reinforcement learning strategy that rewards the model for accurate classification while encouraging the use of a diverse set of investigative tools.5 This prevents the model from relying on a single type of clue and ensures a more holistic forensic assessment.

### **Path-Augmented Reinforcement Learning (PA-FAS)**

The challenge of multimodal reasoning in FAS lies in the scarcity of high-quality annotations that link different sensory inputs to a final verdict. The PA-FAS framework addresses this through path-augmented reinforcement learning.11 It utilizes a human-multimodal FAS reasoning tree, which categorizes forensic evidence across RGB, Depth, and Infrared (IR) modalities.11

A critical innovation in PA-FAS is the answer-shuffling mechanism used during the supervised fine-tuning (SFT) phase.11 By randomly swapping the final answer in the Chain-of-Thought with a label from a different sample, researchers "sever" the model's reliance on simple image-to-label mapping.11 This forces the model to focus on the logical consistency of its reasoning path, effectively mitigating "shortcut learning" where a model might correctly guess a label based on irrelevant features like background lighting.11 Following SFT, the model undergoes reinforcement learning using Group Relative Policy Optimization (GRPO), where it is rewarded for outputting correct labels and adhering to the logical structure of its internal forensic investigations.11

## **Generalization and Generative Defense: Battling Unseen Attacks**

A persistent obstacle in Face Anti-Spoofing is the domain generalization gap. Models trained on a specific set of attack types—such as printed photos or standard LCD screens—often fail when confronted with novel attack media, such as ultra-high-definition OLED displays or hybrid silicone masks.3 To combat this, researchers have turned to generative AI and vision foundation models to synthesize new data and extract more robust features.

### **Pattern Conversion Generative Adversarial Networks (PCGAN)**

The PCGAN framework focuses on the disentanglement of spoofing patterns from facial identity features.3 Standard data augmentation often applies global changes that do not reflect the physical reality of a presentation attack. In contrast, PCGAN utilizes a swapping auto-encoder architecture designed to explicitly separate "spoof artifacts" (the noise and patterns introduced by the re-capturing process) from "facial content" (the intrinsic identity and spatial structure of the face).13

This disentanglement allows the generation of synthetic, diverse FAS data. For example, a single facial identity can be combined with thousands of different synthesized spoof artifacts, ranging from halftone patterns in print to moiré-like interference in replay attacks.13 The effectiveness of these generated artifacts is confirmed through forensic visualization tools like Canny detection and Sobel filtering.13 This approach addresses the limitation of existing FAS datasets, which typically contain fewer than 100 identities, by allowing the model to learn liveness features across a virtually unlimited population of synthesized subjects.3

The PCGAN encoder is specially designed with reduced downsampling to ensure the preservation of fine-grained artifact patterns, which are often lost in standard deep learning bottlenecks.13 This is often paired with a Patch-based Multi-tasking Network (PMN), which jointly analyzes full-face images and randomly cropped patches.13 PMN ensures that the system can detect "partial attacks"—scenarios where only part of the face is spoofed, such as a cutout mask or a localized digital forgery.3

### **Vision Foundation Models: DINOv2 and the Register Mechanism**

The transition from specialized convolutional neural networks (CNNs) to Vision Foundation Models (VFMs) like DINOv2 has provided FAS researchers with significantly more robust feature extractors. However, standard Vision Transformers (ViTs) often suffer from the "spike phenomenon," where the model places excessive attention on non-essential patches, such as background elements, which can lead to overfitting on liveness-irrelevant signals.14

To solve this, researchers have optimized DINOv2 with "registers"—additional learnable tokens included in the input sequence that act as "sinks" for redundant or non-informative data.14 These registers suppress perturbations in the attention mechanism, allowing the model to focus its capacity on the minute, fine-grained details that differentiate live skin from synthetic materials.14

Experimental evaluations on the ICCV2025 FAS Workshop dataset, which includes a mix of physical and digital attacks (Print, Replay, Cutouts, FaceSwap), have demonstrated the superiority of this approach. By unfreezing only the last encoder block of the DINOv2 backbone and using Focal Loss to emphasize hard-to-classify examples, researchers achieved substantial improvements in accuracy and domain generalization.15

| Model Configuration | ACER (Lower is Better) | AUC (Higher is Better) | Accuracy (ACC) |
| :---- | :---- | :---- | :---- |
| **Workshop Baseline (AjianLiu)** | 0.2259 | 0.8989 | 0.6355 |
| **DINOv2 (All Blocks Trainable)** | 0.2605 | 0.8443 | \- |
| **DINOv2 (Last 4 Blocks Trainable)** | 0.2291 | 0.8841 | \- |
| **DINOv2 with Registers (Last Block Only)** | **0.1107** | **0.9480** | **0.9047** |

These results indicate that foundation models, when properly constrained and augmented with register mechanisms, provide the most effective path forward for generalizable biometric defense.14

### **Class-Free Prompt Learning (CFPL-FAS)**

The CFPL-FAS framework leverages large-scale vision-language models like CLIP to further enhance generalization.17 It addresses the core problem of liveness-irrelevant features—such as background lighting or sensor noise—interfering with the identification of spoofing clues.12 CFPL-FAS introduces two lightweight transformers, the Content Q-Former (CQF) and Style Q-Former (SQF).17

The CQF focuses on extracting prompts related to the content (where spoofing clues are often hidden), while the SQF isolates the style (the instance-specific liveness-irrelevant signals).18 By dynamically adjusting the classifier's weights based on these learned textual prompts, the model can "weaken" the impact of the style and "strengthen" the visual features associated with true liveness or spoofing patterns.12 This alignment of visual and textual semantics allows for a broader, more robust feature space that adapts to the characteristics of the specific device or environment during inference.

## **Expanding Beyond RGB: Emerging Multimodal Signatures**

As display and manufacturing technologies advance, the visual gap between a live face and a reproduction becomes increasingly narrow in the standard RGB spectrum. This has led to the adoption of "hardware-software co-design," utilizing specialized sensors to capture temporal, structural, and physiological signatures that are virtually impossible to forge using traditional presentation attack instruments.

### **Event-Based Vision and Temporal Ocular Dynamics**

Event cameras, which asynchronously report changes in brightness with microsecond resolution, have emerged as a high-performance alternative for liveness detection.20 These sensors are particularly effective at capturing fast temporal ocular dynamics, such as saccades and rapid eye blinks, that are impossible to faithfully replicate on standard 60Hz or 120Hz screens.20

The 2026 state-of-the-art involves an active challenge-response mechanism using event cameras.21 When a user is prompted to perform a fast eye movement, the event stream captures the high-frequency temporal pattern. Because replay attacks involve reconstructing event data into a frame-based representation for display, they inevitably introduce temporal resampling and spatial artifacts that appear as distinctive patterns in the event domain.20

Researchers have deployed Spiking Convolutional Neural Networks (SNNs) to process these event streams.20 SNNs are naturally suited for event data as they operate in a sparse, asynchronous manner that preserves the microsecond resolution of the signal.20 Experiments using an expanded RGBE-Gaze dataset have shown that event-based sensing can achieve up to 95.37% accuracy in discriminating between genuine eye movements and replayed video attacks.20

### **PCGattnNet: 3D Point Clouds and Graph Attention**

For smartphone-based access control, 3D point cloud data has become a critical modality for detecting flat surfaces and rigid masks.23 The Point Cloud Graph Attention Network (PCGattnNet) represents facial depth data captured from a smartphone's front-facing sensor as a dynamic graph.23 Unlike traditional 3D CNNs that use voxels, PCGattnNet can better capture the structural interactions between points, allowing it to detect the minute curvature differences between a human face and a high-fidelity mask.23

A key innovation in this space is the Adaptive Feature Learning (AFL) module, which replaces standard radius neighborhood sampling.25 By calculating local similarity through feature subtraction, the AFL module enhances the model's ability to capture prominent variations in facial regions like the forehead or cheeks, which are common areas where mask edges or flat surface signatures become apparent.10

### **BioMoTouch and Physiological Verification**

Beyond visual and structural signatures, the industry is exploring physiological liveness indicators. Remote photoplethysmography (rPPG) uses standard RGB cameras to detect subtle changes in skin color caused by the human pulse.6 Advanced CNN-RNN architectures are used to estimate these rPPG signals, effectively measuring the heartbeat of the person in front of the camera to verify their vitality.6

BioMoTouch extends this concept to mobile touch interaction.27 It recognizes that during a touch gesture, a user's finger morphology and skeletal structure induce distinctive physiological patterns in the capacitive touchscreen's response.27 By integrating capacitive signals with inertial measurement unit (IMU) data, BioMoTouch creates a unified representation of "touch behavior" that includes both the user's motor patterns and their internal physiological structure.26 In realistic usage conditions with 38 participants, BioMoTouch achieved a balanced accuracy of 99.71% and an equal error rate (EER) of 0.27%, demonstrating its potential for non-intrusive, continuous liveness verification.26

| Modality | Forensic Mechanism | Primary Defense Target |
| :---- | :---- | :---- |
| **Event-Based** | Microsecond temporal resolution of saccades/blinks | High-resolution replay attacks, digital deepfakes 20 |
| **3D Point Cloud** | Graph-based surface curvature and depth analysis | Printed photos, rigid 3D masks, tablet displays 23 |
| **rPPG** | Extraction of blood flow (heart rate) from RGB video | High-fidelity non-living masks, static prints 6 |
| **BioMoTouch** | Capacitive-IMU physiological contact dynamics | Robotic spoofing, artificial digits, identity theft 26 |
| **Audio-Visual** | AVSR / Lip-syncing and acoustic echo analysis | Voice cloning, synthetic audio-visual forgeries 23 |

## **Technical Comparison of Leading FAS Frameworks**

As we progress through 2026, the choice of a liveness detection framework depends heavily on the specific deployment scenario—whether it is a high-security banking environment, a massive border control system, or a lightweight mobile application.

### **Edge-Optimized Architectures: AttackNet V2.2 and CASO-PAD**

For mobile devices with limited computational power, researchers have developed highly efficient CNN architectures. AttackNet V2.2 utilizes addition-based residual learning and optimized activation functions to achieve an optimal balance between accuracy and speed.30 This model requires only 16.1 MB of memory, nearly half that of its predecessors, while doubling the training speed.30

CASO-PAD addresses the same challenge by introducing "involution"—a content-adaptive spatial operator—into the MobileNetV3 backbone.30 Traditional convolution uses fixed kernels across all spatial locations, but involution allows the kernel to be dynamically generated based on the specific content of each image patch.30 This allows CASO-PAD to better capture localized texture artifacts—such as the paper grain in a print attack—without the overhead of a large vision transformer.30

| Metric | AttackNet V2.2 | CASO-PAD | DINOv2 \+ Registers |
| :---- | :---- | :---- | :---- |
| **Memory Usage** | 16.1 MB 30 | 45 MB 31 | \~300+ MB 14 |
| **Primary Strength** | Training Efficiency 30 | Localized Texture 30 | Domain Generalization 15 |
| **Architecture** | Efficient Residual CNN 30 | MobileNetV3 \+ Involution 30 | Vision Transformer 15 |
| **Latency (CPU)** | 1.0–2.0 seconds 6 | \<100ms (Mobile) 30 | High (Server-side) 14 |

### **One-Class FAS: The SLIP Framework**

In many real-world applications, collecting a comprehensive dataset of all possible spoof attacks is impossible. One-Class FAS focuses on learning the "intrinsic liveness features" solely from genuine training images.32 The SLIP framework (Spoof-aware Language Image Pretraining) uses vision-language alignment to overcome the absence of spoof training data.33 By using "spoof prompts" to generate "spoof-like" image features in the latent space, SLIP allows the model to learn the boundary of what constitutes a "live" face without ever seeing a physical presentation attack during its training phase.32

## **Ethical and Forensic Trends in 2026**

The maturation of Face Anti-Spoofing has also brought a renewed focus on fairness, transparency, and the resilience of systems against adversarial subversion.

### **FairPAD: Mitigating Demographic Bias**

A critical issue identified in earlier FAS systems was the "bias gap," where detection algorithms exhibited higher False Rejection Rates (FRR) for specific ethnic or age groups.34 FairPAD was introduced to address this through Adversarial Attribute Disentanglement.34 By using adversarial supervision to "suppress" sensitive demographic information in the model's internal representations, FairPAD ensures that the liveness decision is based purely on forensic spoofing artifacts rather than the subject's demographic features.34 This is complemented by Demographic Distribution Alignment, which ensures that the model's feature extraction is consistent across different groups.34

To quantify the remaining inconsistencies in fairness, researchers introduced the Fairness Disagreement Index (FDI) in 2026\.37 This index measures the degree to which different fairness metrics (such as demographic parity or equality of opportunity) yield contradictory conclusions about a model's bias.37 This multi-metric approach ensures that "fairness reporting" is more than a performative exercise and reflects the actual reliability of the system in diverse real-world populations.36

### **Real-Time Forensics and the FaceLog Architecture**

For commercial web and mobile applications, the "gold standard" has become a two-layer authentication framework, exemplified by systems like FaceLog.39 The first layer is a lightweight liveness check based on the Eye Aspect Ratio (EAR) method.39 EAR detects natural eye blinks by calculating the ratio between the vertical and horizontal distances of eye landmarks:

![][image1]  
where ![][image2] are the coordinates of the detected eye landmarks.41 The EAR remains relatively constant when the eye is open and drops toward zero during a blink.41 This method provides a "first-pass" defense against static 2D images at ultra-low computational cost.39 Only after passing this first layer does the system trigger the more expensive ResNet-based facial verification and potentially the MLLM-based reasoning chains if an anomaly is detected.39

### **Vulnerabilities and the Cybersecurity Threat Model**

Despite these advancements, the field remains locked in a battle with hardware-level injection attacks.4 Security researchers have demonstrated that a universal methodology for bypassing liveness detection involves using "evil hardware" to inject fake video streams directly into the application's sensor interface, bypassing the physical camera entirely.4 Furthermore, "X-glasses" have been used to subvert attention-detection mechanisms (like pupil tracking) at an ultra-low cost.4

This highlights that the future of FAS must be a "system-level" defense. It is not enough for an algorithm to be accurate; the entire data pipeline—from the physical sensor to the encrypted transmission and the final reasoning engine—must be secured against manipulation.4 This "holistic" approach is why multimodal hardware-software co-design (such as event cameras or capacitive touch) is becoming the preferred strategy for high-security applications.20

## **Theoretical Foundations of Presentation Attack Detection**

The effectiveness of modern FAS systems is grounded in the physical laws of light and geometry. Most texture-based methods still rely on the principles of the Lambertian reflectance model.43 A live 3D face reflects light in complex patterns because the surface normal ![][image3] varies across the face's geometry, whereas a 2D photograph or screen has a constant surface normal.43

The intensity of a pixel ![][image4] can be modeled as:

![][image5]  
where ![][image6] is the reflectance coefficient, ![][image7] is the intensity of incoming light, and ![][image8] is the direction of the light source.43 By analyzing the 2D Fourier spectra of an image, models can calculate the High Frequency Descriptor (HFD), identifying that flat photos lack the high-frequency components and depth-induced shading found in live 3D subjects.2

The current research trend is to combine these "first-principles" of physics with the "semantic reasoning" of large language models.1 For instance, a model might use spectral analysis to find a moiré pattern and then use an MLLM to articulate: "The periodic high-frequency noise in the forehead region is consistent with an LCD screen refresh artifact, confirming a replay attack".9

## **Summary Outlook**

As we move into the late 2026 period, the "Gold Standard" for biometric security is defined by a multimodal, reasoning-based system. The era of the simple binary classifier has effectively ended, replaced by frameworks that emphasize interpretability (I-FAS, TAR-FAS), robustness through foundation models (DINOv2 with Registers), and unforgeable physical signatures (Event-based dynamics, BioMoTouch).

The key takeaway for 2026 is that the defensive gap between "live" and "spoof" is no longer being sought in the pixels alone, but in the coordination of physiological, temporal, and semantic evidence. While vision foundation models provide the backbone for feature extraction, the addition of temporal dynamics and human-readable forensic reasoning has created a defensive layer that is increasingly difficult for even the most sophisticated generative AI to penetrate. The focus on fairness (FairPAD) and system-level security (FaceLog) further ensures that these systems are not only robust against attacks but also transparent and equitable in their global deployment.

#### **Works cited**

1. \[2501.01720\] Interpretable Face Anti-Spoofing: Enhancing Generalization with Multimodal Large Language Models \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2501.01720](https://arxiv.org/abs/2501.01720)  
2. AN OVERVIEW OF FACE LIVENESS DETECTION \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/1405.2227](https://arxiv.org/pdf/1405.2227)  
3. \[2604.09018\] Domain-generalizable Face Anti-Spoofing with Patch-based Multi-tasking and Artifact Pattern Conversion \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2604.09018](https://arxiv.org/abs/2604.09018)  
4. Biometric Authentication Under Threat: Liveness Detection Hacking \- Black Hat, accessed May 1, 2026, [https://i.blackhat.com/USA-19/Wednesday/us-19-Chen-Biometric-Authentication-Under-Threat-Liveness-Detection-Hacking.pdf](https://i.blackhat.com/USA-19/Wednesday/us-19-Chen-Biometric-Authentication-Under-Threat-Liveness-Detection-Hacking.pdf)  
5. From Intuition to Investigation: A Tool-Augmented Reasoning ... \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2603.01038](https://arxiv.org/abs/2603.01038)  
6. Robust Face Liveness Detection for Biometric Authentication using Single Image \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2511.02645v1](https://arxiv.org/html/2511.02645v1)  
7. From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/2603.01038](https://arxiv.org/pdf/2603.01038)  
8. From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/401470040\_From\_Intuition\_to\_Investigation\_A\_Tool-Augmented\_Reasoning\_MLLM\_Framework\_for\_Generalizable\_Face\_Anti-Spoofing](https://www.researchgate.net/publication/401470040_From_Intuition_to_Investigation_A_Tool-Augmented_Reasoning_MLLM_Framework_for_Generalizable_Face_Anti-Spoofing)  
9. \[Papierüberprüfung\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing, accessed May 1, 2026, [https://www.themoonlight.io/de/review/from-intuition-to-investigation-a-tool-augmented-reasoning-mllm-framework-for-generalizable-face-anti-spoofing](https://www.themoonlight.io/de/review/from-intuition-to-investigation-a-tool-augmented-reasoning-mllm-framework-for-generalizable-face-anti-spoofing)  
10. Block diagram of the proposed LBP based anti-spoofing algorithm... \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/figure/Block-diagram-of-the-proposed-LBP-based-anti-spoofing-algorithm-display-do-not-possess\_fig2\_230775873](https://www.researchgate.net/figure/Block-diagram-of-the-proposed-LBP-based-anti-spoofing-algorithm-display-do-not-possess_fig2_230775873)  
11. Towards Interpretable and Generalizable Multimodal Face ... \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2511.17927](https://arxiv.org/abs/2511.17927)  
12. CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2403.14333v1](https://arxiv.org/html/2403.14333v1)  
13. Domain-generalizable Face Anti-Spoofing with Patch-based ... \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/2604.09018](https://arxiv.org/pdf/2604.09018)  
14. Optimizing DINOv2 with Registers for Face Anti-Spoofing \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2510.17201v1](https://arxiv.org/html/2510.17201v1)  
15. Optimizing DINOv2 with Registers for Face Anti-Spoofing \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2510.17201](https://arxiv.org/abs/2510.17201)  
16. Optimizing DINOv2 with Registers for Face Anti-Spoofing \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2510.17201v2](https://arxiv.org/html/2510.17201v2)  
17. CVPR Poster CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing, accessed May 1, 2026, [https://cvpr.thecvf.com/virtual/2024/poster/30953](https://cvpr.thecvf.com/virtual/2024/poster/30953)  
18. \[2403.14333\] CFPL-FAS: Class Free Prompt Learning for Generalizable Face Anti-spoofing, accessed May 1, 2026, [https://arxiv.org/abs/2403.14333](https://arxiv.org/abs/2403.14333)  
19. arXiv:2403.14333v1 \[cs.CV\] 21 Mar 2024, accessed May 1, 2026, [https://arxiv.org/pdf/2403.14333](https://arxiv.org/pdf/2403.14333)  
20. \[2604.26285\] Event-based Liveness Detection using Temporal Ocular Dynamics: An Exploratory Approach \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2604.26285](https://arxiv.org/abs/2604.26285)  
21. Event-based Liveness Detection using Temporal Ocular Dynamics: An Exploratory Approach \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2604.26285v1](https://arxiv.org/html/2604.26285v1)  
22. A 240 × 180 130 dB 3 µs Latency Global Shutter Spatiotemporal Vision Sensor, accessed May 1, 2026, [https://www.researchgate.net/publication/266149569\_A\_240\_180\_130\_dB\_3\_s\_Latency\_Global\_Shutter\_Spatiotemporal\_Vision\_Sensor](https://www.researchgate.net/publication/266149569_A_240_180_130_dB_3_s_Latency_Global_Shutter_Spatiotemporal_Vision_Sensor)  
23. (PDF) PCGattnNet: A 3D Point Cloud Dynamic Graph Attention for Generalizable Face Presentation Attack Detection \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/388437248\_PCGattnNet\_A\_3D\_Point\_Cloud\_Dynamic\_Graph\_Attention\_for\_Generalizable\_Face\_Presentation\_Attack\_Detection](https://www.researchgate.net/publication/388437248_PCGattnNet_A_3D_Point_Cloud_Dynamic_Graph_Attention_for_Generalizable_Face_Presentation_Attack_Detection)  
24. PCGattnNet: A 3-D Point Cloud Dynamic Graph Attention for Generalizable Face Presentation Attack Detection \- IEEE Xplore, accessed May 1, 2026, [https://ieeexplore.ieee.org/iel8/8423754/11180155/10854497.pdf](https://ieeexplore.ieee.org/iel8/8423754/11180155/10854497.pdf)  
25. A 3D Face Recognition Algorithm Directly Applied to Point Clouds \- MDPI, accessed May 1, 2026, [https://www.mdpi.com/2313-7673/10/2/70](https://www.mdpi.com/2313-7673/10/2/70)  
26. BioMoTouch: Touch-Based Behavioral Authentication via Biometric-Motion Interaction Modeling \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/403641848\_BioMoTouch\_Touch-Based\_Behavioral\_Authentication\_via\_Biometric-Motion\_Interaction\_Modeling](https://www.researchgate.net/publication/403641848_BioMoTouch_Touch-Based_Behavioral_Authentication_via_Biometric-Motion_Interaction_Modeling)  
27. BioMoTouch: Touch-Based Behavioral Authentication via Biometric-Motion Interaction Modeling \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2604.07071v1](https://arxiv.org/html/2604.07071v1)  
28. BioTouch: Reliable Re-Authentication via Finger Bio-Capacitance and Touching Behavior \- PMC, accessed May 1, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9105168/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9105168/)  
29. 26th Annual Conference of the International ... \- Proceedings.com, accessed May 1, 2026, [https://www.proceedings.com/content/084/084040webtoc.pdf](https://www.proceedings.com/content/084/084040webtoc.pdf)  
30. Face Spoofing Detection Based on Local Ternary Label Supervision in Fully Convolutional Networks | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/340419723\_Face\_Spoofing\_Detection\_Based\_on\_Local\_Ternary\_Label\_Supervision\_in\_Fully\_Convolutional\_Networks](https://www.researchgate.net/publication/340419723_Face_Spoofing_Detection_Based_on_Local_Ternary_Label_Supervision_in_Fully_Convolutional_Networks)  
31. (PDF) Optimizing CNN Architectures for Face Liveness Detection: Performance, Efficiency, and Generalization across Datasets \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/392557911\_Optimizing\_CNN\_Architectures\_for\_Face\_Liveness\_Detection\_Performance\_Efficiency\_and\_Generalization\_across\_Datasets](https://www.researchgate.net/publication/392557911_Optimizing_CNN_Architectures_for_Face_Liveness_Detection_Performance_Efficiency_and_Generalization_across_Datasets)  
32. SLIP: Spoof-Aware One-Class Face Anti-Spoofing with Language Image Pretraining \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2503.19982](https://arxiv.org/abs/2503.19982)  
33. SLIP: Spoof-Aware One-Class Face Anti-Spoofing with Language Image Pretraining \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2503.19982v1](https://arxiv.org/html/2503.19982v1)  
34. A face antispoofing database with diverse attacks | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/261046224\_A\_face\_antispoofing\_database\_with\_diverse\_attacks](https://www.researchgate.net/publication/261046224_A_face_antispoofing_database_with_diverse_attacks)  
35. Achieving Fairness Without Harm via Selective Demographic Experts \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/2511.06293](https://arxiv.org/pdf/2511.06293)  
36. Achieving Fairness Without Harm via Selective Demographic Experts \- arXiv, accessed May 1, 2026, [https://arxiv.org/html/2511.06293v1](https://arxiv.org/html/2511.06293v1)  
37. \[2604.15038\] When Fairness Metrics Disagree: Evaluating the Reliability of Demographic Fairness Assessment in Machine Learning \- arXiv, accessed May 1, 2026, [https://arxiv.org/abs/2604.15038](https://arxiv.org/abs/2604.15038)  
38. AI Fairness Beyond Complete Demographics: Current Achievements and Future Directions \- arXiv, accessed May 1, 2026, [https://arxiv.org/pdf/2511.13525](https://arxiv.org/pdf/2511.13525)  
39. Facelog: Login System with User Authentication Toolkit Utilizing Convolutional Neural Network Algorithm \- RSIS International, accessed May 1, 2026, [https://rsisinternational.org/journals/ijrsi/uploads/vol13-iss1-pg841-854-202601\_pdf.pdf](https://rsisinternational.org/journals/ijrsi/uploads/vol13-iss1-pg841-854-202601_pdf.pdf)  
40. Is artificial intelligence a new battleground for cybersecurity? | Request PDF \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/385612411\_Is\_artificial\_intelligence\_a\_new\_battleground\_for\_cybersecurity](https://www.researchgate.net/publication/385612411_Is_artificial_intelligence_a_new_battleground_for_cybersecurity)  
41. A Novel Approach to Liveness Detection: Real-time Eye Blink, Head Movement, and Lip Movement Analysis \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/389438167\_A\_Novel\_Approach\_to\_Liveness\_Detection\_Real-time\_Eye\_Blink\_Head\_Movement\_and\_Lip\_Movement\_Analysis](https://www.researchgate.net/publication/389438167_A_Novel_Approach_to_Liveness_Detection_Real-time_Eye_Blink_Head_Movement_and_Lip_Movement_Analysis)  
42. Real-Time Face Liveness Detection and Face Anti-spoofing Using Deep Learning \- Atlantis Press, accessed May 1, 2026, [https://www.atlantis-press.com/article/125989871.pdf](https://www.atlantis-press.com/article/125989871.pdf)  
43. (PDF) Face Liveness Detection from a Single Image with Sparse Low Rank Bilinear Discriminative Model \- ResearchGate, accessed May 1, 2026, [https://www.researchgate.net/publication/262405100\_Face\_Liveness\_Detection\_from\_a\_Single\_Image\_with\_Sparse\_Low\_Rank\_Bilinear\_Discriminative\_Model](https://www.researchgate.net/publication/262405100_Face_Liveness_Detection_from_a_Single_Image_with_Sparse_Low_Rank_Bilinear_Discriminative_Model)  
44. (PDF) Biometric Liveness Detection: Challenges and Research Opportunities, accessed May 1, 2026, [https://www.researchgate.net/publication/283555868\_Biometric\_Liveness\_Detection\_Challenges\_and\_Research\_Opportunities](https://www.researchgate.net/publication/283555868_Biometric_Liveness_Detection_Challenges_and_Research_Opportunities)  
45. An overview of face liveness detection \- SciSpace, accessed May 1, 2026, [https://scispace.com/pdf/an-overview-of-face-liveness-detection-3vtwjhjzbz.pdf](https://scispace.com/pdf/an-overview-of-face-liveness-detection-3vtwjhjzbz.pdf)
