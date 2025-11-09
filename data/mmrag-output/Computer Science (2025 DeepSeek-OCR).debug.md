# MM-RAG Debug Output

Total elements: 134

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| Figure (a) shows the compression ratio (number of text tokens in ground truth/number of vision tokens model used) testing on Fox [21] benchmark; Figure (b) shows performance comparisons on OmniDocBench [27]. DeepSeek-OCR can ac

VLM description: The figure (a) displays the compression ratio (number of text tokens in ground truth/number of vision tokens model used) testing on the Fox [21] benchmark. The figure (b) shows performance comparisons on the OmniDocBench [27]. DeepSeek-OCR can achieve high compression ratios, as indicated by the data trends in the figure.

---

## Element: page_1_para_1

**Type:** heading  
**Page:** 1  

**Content:**

# DeepSeek-OCR: Contexts Optical Compression

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

Haoran Wei, Yaofeng Sun, Yukun Li

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

We present DeepSeek- OCR as an initial investigation into the feasibility of compressing long contexts via optical 2D mapping. DeepSeek- OCR consists of two components: DeepEncoder and DeepSeek3B- MoE- A570M as the decoder. Specifically, DeepEncoder serves as the core engine, designed to maintain low activations under high- resolution input while achieving high compression ratios to ensure an optimal and manageable number of vision tokens. Experiments show that when the number of text tokens is within 10 times that of vision tokens (i.e., a compression ratio \(< 10\times\) ), the model can achieve decoding (OCR) precision of \(97\%\) . Even at a compression ratio of \(20\times\) , the OCR accuracy still remains at about \(60\%\) . This shows considerable promise for research areas such as historical long- context compression and memory forgetting mechanisms in LLMs. Beyond this, DeepSeek- OCR also demonstrates high practical value. On OmniDocBench, it surpasses GOT- OCR2.0 (256 tokens/page) using only 100 vision tokens, and outperforms MinerU2.0 (6000+ tokens per page on average) while utilizing fewer than 800 vision tokens. In production, DeepSeek- OCR can generate training data for LLMs/VLMs at a scale of \(200k+\) pages per day (a single A100- 40G). Codes and model weights are publicly accessible at http://github.com/deepseek- ai/DeepSeek- OCR.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

2.1 Typical Vision Encoders in VLMs 4

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

2.2 End-to-end OCR Models 4

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

3.2.1 Architecture of DeepEncoder 5

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

3.2.2 Multiple resolution support 6

---

## Element: page_1_para_5

**Type:** text  
**Page:** 1  

**Content:**

3.3 The MoE Decoder 7

---

## Element: page_1_para_6

**Type:** text  
**Page:** 1  

**Content:**

3.4.1 OCR 1.0 data 7

---

## Element: page_1_para_7

**Type:** text  
**Page:** 1  

**Content:**

3.4.2 OCR 2.0 data 8

---

## Element: page_1_para_8

**Type:** text  
**Page:** 1  

**Content:**

3.4.3 General vision data 9

---

## Element: page_1_para_9

**Type:** text  
**Page:** 1  

**Content:**

3.4.4 Text-only data 9

---

## Element: page_1_para_10

**Type:** text  
**Page:** 1  

**Content:**

3.5 Training Pipelines 9

---

## Element: page_1_para_11

**Type:** text  
**Page:** 1  

**Content:**

3.5.1 Training DeepEncoder 10

---

## Element: page_1_para_12

**Type:** text  
**Page:** 1  

**Content:**

3.5.2 Training DeepSeek- OCR 10

---

## Element: page_1_para_13

**Type:** text  
**Page:** 1  

**Content:**

4.1 Vision- text Compression Study 10

---

## Element: page_1_para_14

**Type:** text  
**Page:** 1  

**Content:**

4.2 OCR Practical Performance 12

---

## Element: page_1_para_15

**Type:** text  
**Page:** 1  

**Content:**

4.3 Qualitative Study 12

---

## Element: page_1_para_16

**Type:** text  
**Page:** 1  

**Content:**

4.3.1 Deep parsing 12

---

## Element: page_1_para_17

**Type:** text  
**Page:** 1  

**Content:**

4.3.2 Multilingual recognition 16

---

## Element: page_1_para_18

**Type:** text  
**Page:** 1  

**Content:**

4.3.3 General vision understanding 17

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

Current Large Language Models (LLMs) face significant computational challenges when processing long textual content due to quadratic scaling with sequence length. We explore a potential solution: leveraging visual modality as an efficient compression medium for textual information. A single image containing document text can represent rich information using substantially fewer tokens than the equivalent digital text, suggesting that optical compression through vision tokens could achieve much higher compression ratios.

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

This insight motivates us to reexamine vision- language models (VLMs) from an LLM- centric perspective, focusing on how vision encoders can enhance LLMs' efficiency in processing textual information rather than basic VQA [12, 16, 24, 32, 41] what humans excel at. OCR tasks, as an intermediate modality bridging vision and language, provide an ideal testbed for this vision- text compression paradigm, as they establish a natural compression- decompression mapping between visual and textual representations while offering quantitative evaluation metrics.

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

Accordingly, we present DeepSeek- OCR, a VLM designed as a preliminary proof- of- concept for efficient vision- text compression. Our work makes three primary contributions:

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

First, we provide comprehensive quantitative analysis of vision- text token compression ratios. Our method achieves \(96\% +\) OCR decoding precision at \(9 - 10\times\) text compression, \(\sim 90\%\) at \(10 - 12\times\) compression, and \(\sim 60\%\) at \(20\times\) compression on Fox [21] benchmarks featuring diverse document layouts (with actual accuracy being even higher when accounting for formatting differences between output and ground truth), as shown in Figure 1(a). The results demonstrate that compact language models can effectively learn to decode compressed visual representations, suggesting that larger LLMs could readily acquire similar capabilities through appropriate pretraining design.

---

## Element: page_1_para_5

**Type:** text  
**Page:** 1  

**Content:**

Second, we introduce DeepEncoder, a novel architecture that maintains low activation memory and minimal vision tokens even with high- resolution inputs. It serially connects window attention and global attention encoder components through a \(16\times\) convolutional compressor. This design ensures that the window attention component processes a large number of vision tokens, while the compressor reduces vision tokens before they enter the dense global attention component, achieving effective memory and token compression.

---

## Element: page_1_para_6

**Type:** text  
**Page:** 1  

**Content:**

Third, we develop DeepSeek- OCR based on DeepEncoder and DeepSeek3B- MoE [19, 20]. As shown in Figure 1(b), it achieves state- of- the- art performance within end- to- end models on OmniDocBench while using the fewest vision tokens. Additionally, we equip the model with capabilities for parsing charts, chemical formulas, simple geometric figures, and natural images to enhance its practical utility further. In production, DeepSeek- OCR can generate 33 million pages of data per day for LLMs or VLMs using 20 nodes (each with 8 A100- 40G GPUs).

---

## Element: page_1_para_7

**Type:** text  
**Page:** 1  

**Content:**

In summary, this work presents a preliminary exploration of using visual modality as an efficient compression medium for textual information processing in LLMs. Through DeepSeek- OCR, we demonstrate that vision- text compression can achieve significant token reduction (7- 20x) for different historical context stages, offering a promising direction for addressing long- context challenges in large language models. Our quantitative analysis provides empirical guidelines for VLM token allocation optimization, while the proposed DeepEncoder architecture showcases practical feasibility with real- world deployment capabilities. Although focused on OCR as a proof- of- concept, this paradigm opens new possibilities for rethinking how vision and language modalities can be synergistically combined to enhance computational efficiency in large- scale text processing and agent systems.

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| Typical vision encoders in popular VLMs. Here are three types of encoders commonly used in current open-source VLMs, all of which suffer from their respective deficiencies.

VLM description: The plot is a bar chart comparing three types of vision encoders in popular VLMs. The x-axis represents the types of encoders, and the y-axis represents the number of examples. The three types of encoders are:
1. **Type A** (top bar): This encoder is commonly used in current open-source VLMs and is known for its high accuracy but also for its high computational cost.
2. **Type B** (middle bar): This encoder is known for its high computational efficiency but is less accurate than Type A.
3. **Type C** (bottom bar): This encoder is known for its balance between accuracy and computational efficiency.

The chart shows that Type A is the most commonly used encoder, followed by Type B and then Type C.

---

## Element: page_1_para_1

**Type:** heading  
**Page:** 1  

**Content:**

### 2.1. Typical Vision Encoders in VLMs

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

Current open- source VLMs employ three main types of vision encoders, as illustrated in Figure 2. The first type is a dual- tower architecture represented by Vary [36], which utilizes parallel SAM [17] encoder to increase visual vocabulary parameters for high- resolution image processing. While offering controllable parameters and activation memory, this approach suffers from significant drawbacks: it requires dual image preprocessing that complicates deployment and makes encoder pipeline parallelism challenging during training. The second type is tile- based method exemplified by InternVL2.0 [8], which processes images by dividing them into small tiles for parallel computation, reducing activation memory under high- resolution settings. Although capable of handling extremely high resolutions, this approach has notable limitations due to its typically low native encoder resolution (below \(512 \times 512\) ), causing large images to be excessively fragmented and resulting in numerous vision tokens. The third type is adaptive resolution encoding represented by Qwen2- VL [35], which adopts the NaViT [10] paradigm to directly process full images through patch- based segmentation without tile parallelization. While this encoder can handle diverse resolutions flexibly, it faces substantial challenges with large images due to massive activation memory consumption that can cause GPU memory overflow, and sequence packing requires extremely long sequence lengths during training. Long vision tokens will slow down both prefill and generation phases of inference.

---

## Element: page_1_para_3

**Type:** heading  
**Page:** 1  

**Content:**

### 2.2. End-to-end OCR Models

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

OCR, particularly document parsing task, has been a highly active topic in the image- to- text domain. With the advancement of VLMs, a large number of end- to- end OCR models have emerged, fundamentally transforming the traditional pipeline architecture (which required separate detection and recognition expert models) by simplifying OCR systems. Nougat [6] first employs end- to- end framework for academic paper OCR on arXiv, demonstrating the potential of models in handling dense perception tasks. GOT- OCR2.0 [38] expands the scope of OCR2.0 to include more synthetic image parsing tasks and designs an OCR model with performance- efficiency trade- offs, further highlighting the potential of end- to- end OCR researches. Additionally, general vision models such as Qwen- VL series [35], InternVL series [8], and many their derivatives continuously enhance their document OCR capabilities to explore dense visual perception boundaries. However, a crucial research question that current models have not addressed is: for a document containing 1000 words, how many vision tokens are at least needed for decoding? This question holds significant importance for research in the principle that "a picture is worth a thousand words."

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| The architecture of DeepSeek-OCR. DeepSeek-OCR consists of a DeepEncoder and a DeepSeek-3B-MoE decoder. DeepEncoder is the core of DeepSeek-OCR, comprising three components: a SAM [17] for perception dominated by window attenti

VLM description: The plot is a bar chart with the title "The architecture of DeepSeek-OCR." The x-axis represents different components of the DeepSeek-OCR architecture, while the y-axis represents the number of components. The components are: DeepEncoder, DeepSeek-3B-MoE, and DeepSeek-3B-MoE. The y-axis ranges from 0 to 10, with intervals of 2. The bar chart shows that DeepEncoder has the highest number of components (10), followed by DeepSeek-3B-MoE (8), and then DeepSeek-3B-MoE (7).

---

## Element: page_1_para_1

**Type:** heading  
**Page:** 1  

**Content:**

### 3.1. Architecture

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

As shown in Figure 3, DeepSeek- OCR enjoys a unified end- to- end VLM architecture consisting of an encoder and a decoder. The encoder (namely DeepEncoder) is responsible for extracting image features and tokenizing as well as compressing visual representations. The decoder is used for generating the required result based on image tokens and prompts. DeepEncoder is approximately 380M in parameters, mainly composed of an 80M SAM- base [17] and a 300M CLIP- large [29] connected in series. The decoder adopts a 3B MoE [19, 20] architecture with 570M activated parameters. In the following paragraphs, we will delve into the model components, data engineering, and training skills.

---

## Element: page_1_para_3

**Type:** heading  
**Page:** 1  

**Content:**

### 3.2. DeepEncoder

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

To explore the feasibility of contexts optical compression, we need a vision encoder with the following features: 1. Capable of processing high resolutions; 2. Low activation at high resolutions; 3. Few vision tokens; 4. Support for multiple resolution inputs; 5. Moderate parameter count. However, as described in the Section 2.1, current open- source encoders cannot fully satisfy all these conditions. Therefore, we design a novel vision encoder ourselves, named DeepEncoder.

---

## Element: page_1_para_5

**Type:** heading  
**Page:** 1  

**Content:**

#### 3.2.1. Architecture of DeepEncoder

---

## Element: page_1_para_6

**Type:** text  
**Page:** 1  

**Content:**

DeepEncoder mainly consists of two components: a visual perception feature extraction component dominated by window attention, and a visual knowledge feature extraction component with dense global attention. To benefit from the pretraining gains of previous works, we use SAM- base (patch- size 16) and CLIP- large as the main architectures for the two components respectively. For CLIP, we remove the first patch embedding layer since its input is no longer images but output tokens from the previous pipeline. Between the two components, we borrow from Vary [36] and use a 2- layer convolutional module to perform \(16\times\) downsampling of vision tokens. Each convolutional layer has a kernel size of 3, stride of 2, padding of 1, and channels increase from 256 to 1024. Assuming we input a \(1024\times 1024\) image, the DeepEncoder will segment it into \(1024 / 16\times 1024 / 16 = 4096\) patch tokens. Since the first half of encoder is dominated by window attention and only 80M, the activation is acceptable. Before entering global attention,

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| To test model performance under different compression ratios (requiring different numbers of vision tokens) and enhance the practicality of DeepSeek-OCR, we configure it with multiple resolution modes.

VLM description: The plot is a bar chart with the title "DeepSeek-OCR: Contexts Optical Compression." The x-axis represents different compression ratios, ranging from 1:1 to 1:100. The y-axis represents the number of vision tokens required for each compression ratio. The data points are labeled with the authors' names and the DeepSeek-AI company. The chart shows a clear trend where the number of vision tokens required decreases as the compression ratio increases.

---

## Element: page_1_table_1

**Type:** table  
**Page:** 1  

**Content:**

| Multi resolution support of DeepEncoder. For both research and application purposes, we design DeepEncoder with diverse native resolution and dynamic resolution modes.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

the 4096 tokens go through the compression module and the token count becomes \(4096 / 16 = 256\) , thus making the overall activation memory controllable.

---

## Element: page_1_para_2

**Type:** heading  
**Page:** 1  

**Content:**

#### 3.2.2. Multiple resolution support

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

Suppose we have an image with 1000 optical characters and we want to test how many vision tokens are needed for decoding. This requires the model to support a variable number of vision tokens. That is to say the DeepEncoder needs to support multiple resolutions.

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

We meet the requirement aforementioned through dynamic interpolation of positional encodings, and design several resolution modes for simultaneous model training to achieve the capability of a single DeepSeek- OCR model supporting multiple resolutions. As shown in Figure 4, DeepEncoder mainly supports two major input modes: native resolution and dynamic resolution. Each of them contains multiple sub- modes.

---

## Element: page_1_para_5

**Type:** text  
**Page:** 1  

**Content:**

Native resolution supports four sub- modes: Tiny, Small, Base, and Large, with corresponding resolutions and token counts of \(512 \times 512\) (64), \(640 \times 640\) (100), \(1024 \times 1024\) (256), and \(1280 \times 1280\) (400) respectively. Since Tiny and Small models have relatively small resolutions, to avoid wasting vision tokens, images are processed by directly resizing the original shape. For Base and Large modes, in order to preserve the original image aspect ratio, images are padded to the corresponding size. After padding, the number of valid vision tokens is less than the actual number of vision tokens, with the calculation formula being: \[N_{valid} = \lceil N_{actual}\times [1 - ((max(w,h) - min(w,h)) / (max(w,h)))]\rceil \quad (1)\]

---

## Element: page_1_para_6

**Type:** text  
**Page:** 1  

**Content:**

where \(w\) and \(h\) represent the width and height of the original input image.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

Dynamic resolution can be composed of two native resolutions. For example, Gundam mode consists of \(n \times 640 \times 640\) tiles (local views) and a \(1024 \times 1024\) global view. The tiling method following InternVL2.0 [8]. Supporting dynamic resolution is mainly for application considerations, especially for ultra- high- resolution inputs (such as newspaper images). Tiling is a form of secondary window attention that can effectively reduce activation memory further. It's worth noting that due to our relatively large native resolutions, images won't be fragmented too much under dynamic resolution (the number of tiles is controlled within the range of 2 to 9). The vision token number output by the DeepEncoder under Gundam mode is: \(n \times 100 + 256\) , where \(n\) is the number of tiles. For images with both width and height smaller than 640, \(n\) is set to 0, i.e., Gundam mode will degrade to Base mode.

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

Gundam mode is trained together with the four native resolution modes to achieve the goal of one model supporting multiple resolutions. Note that Gundam- master mode ( \(1024 \times 1024\) local views \(+ 1280 \times 1280\) global view) is obtained through continued training on a trained DeepSeek- OCR model. This is mainly for load balancing, as Gundam- master's resolution is too large and training it together would slow down the overall training speed.

---

## Element: page_1_para_3

**Type:** heading  
**Page:** 1  

**Content:**

### 3.3. The MoE Decoder

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

Our decoder uses the DeepSeekMoE [19, 20], specifically DeepSeek- 3B- MoE. During inference, the model activates 6 out of 64 routed experts and 2 shared experts, with about 570M activated parameters. The 3B DeepSeekMoE is very suitable for domain- centric (OCR for us) VLM research, as it obtains the expressive capability of a 3B model while enjoying the inference efficiency of a 500M small model.

---

## Element: page_1_para_5

**Type:** text  
**Page:** 1  

**Content:**

The decoder reconstructs the original text representation from the compressed latent vision tokens of DeepEncoder as: \[f_{\mathrm{dec}}:\mathbb{R}^{n\times d_{\mathrm{latent}}}\to \mathbb{R}^{N\times d_{\mathrm{text}}}; \quad \hat{\mathbf{X}} = f_{\mathrm{dec}}(\mathbf{Z})\quad \mathrm{where} n\leq N \quad (2)\]

---

## Element: page_1_para_6

**Type:** text  
**Page:** 1  

**Content:**

where \(\mathbf{Z} \in \mathbb{R}^{n \times d_{\mathrm{latent}}}\) are the compressed latent(vision) tokens from DeepEncoder and \(\hat{\mathbf{X}} \in \mathbb{R}^{N \times d_{\mathrm{text}}}\) is the reconstructed text representation. The function \(f_{\mathrm{dec}}\) represents a non- linear mapping that can be effectively learned by compact language models through OCR- style training. It is reasonable to conjecture that LLMs, through specialized pretraining optimization, would demonstrate more natural integration of such capabilities.

---

## Element: page_1_para_7

**Type:** heading  
**Page:** 1  

**Content:**

### 3.4. Data Engine

---

## Element: page_1_para_8

**Type:** text  
**Page:** 1  

**Content:**

We construct complex and diverse training data for DeepSeek- OCR, including OCR 1.0 data, which mainly consists of traditional OCR tasks such as scene image OCR and document OCR; OCR 2.0 data, which mainly includes parsing tasks for complex artificial images, such as common charts, chemical formulas, and plane geometry parsing data; General vision data, which is mainly used to inject certain general image understanding capabilities into DeepSeek- OCR and preserve the general vision interface.

---

## Element: page_1_para_9

**Type:** heading  
**Page:** 1  

**Content:**

#### 3.4.1. OCR 1.0 data

---

## Element: page_1_para_10

**Type:** text  
**Page:** 1  

**Content:**

Document data is the top priority for DeepSeek- OCR. We collect 30M pages of diverse PDF data covering about 100 languages from the Internet, with Chinese and English accounting for approximately 25M and other languages accounting for 5M. For this data, we create two types of ground truth: coarse annotations and fine annotations. Coarse annotations are extracted

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| OCR 1.0 fine annotations display. We format the ground truth into an interleaved layout and text format, where each paragraph of text is preceded by the coordinates and label of it in the original image. All coordinates are nor

VLM description: The plot is a scientific figure from a research paper, depicting an OCR 1.0 fine annotations display. The ground truth is formatted into an interleaved layout and text format, with each paragraph of text preceded by the coordinates and label of it in the original image. The coordinates are not visible in the figure. The focus is on the plot type and structure, with no data trends, patterns, or key observations visible. The variable relationships shown are not clear from the figure.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

directly from the full dataset using fitz, aimed at teaching the model to recognize optical text, especially in minority languages. Fine annotations include 2M pages each for Chinese and English, labeled using advanced layout models (such as PP- DocLayout [33]) and OCR models (such as MinuerU [34] and GOT- OCR2.0 [38]) to construct detection and recognition interleaved data. For minority languages, in the detection part, we find that the layout model enjoys certain generalization capabilities. In the recognition part, we use fitz to create small patch data to train a GOT- OCR2.0, then use the trained model to label small patches after layout processing, employing a model flywheel to create 600K data samples. During the training of DeepSeek- OCR, coarse labels and fine labels are distinguished using different prompts. The ground truth for fine annotation image- text pairs can be seen in Figure 5. We also collect 3M Word data, constructing high- quality image- text pairs without layout by directly extracting content. This data mainly brings benefits to formulas and HTML- formatted tables. Additionally, we select some open- source data [28, 37] as supplements.

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

For natural scene OCR, our model mainly supports Chinese and English. The image data sources come from LAION [31] and Wukong [13], labeled using PaddleOCR [9], with 10M data samples each for Chinese and English. Like document OCR, natural scene OCR can also control whether to output detection boxes through prompts.

---

## Element: page_1_para_3

**Type:** heading  
**Page:** 1  

**Content:**

#### 3.4.2. OCR 2.0 data

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

Following GOT- OCR2.0 [38], we refer to chart, chemical formula, and plane geometry parsing data as OCR 2.0 data. For chart data, following OneChart [7], we use pyecharts and matplotlib

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| For charts, we do not use OneChart's [7] dictionary format, but instead use HTML table format as labels, which can save a certain amount of tokens. For plane geometry, we convert the ground truth to dictionary format, where the

VLM description: The plot is a bar chart with two horizontal axes. The x-axis represents the number of samples, ranging from 1 to 100, while the y-axis represents the number of correct predictions, ranging from 0 to 100. The chart shows a positive correlation between the number of samples and the number of correct predictions, with a trend line that suggests a positive relationship.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

to render 10M images, mainly including commonly used line, bar, pie, and composite charts. We define chart parsing as image- to- HTML- table conversion task, as shown in Figure 6(a). For chemical formulas, we utilize SMILES format from PubChem as the data source and render them into images using RDKit, constructing 5M image- text pairs. For plane geometry images, we follow Slow Perception [39] for generation. Specifically, we use perception- ruler size as 4 to model each line segment. To increase the diversity of rendered data, we introduce geometric translation- invariant data augmentation, where the same geometric image is translated in the original image, corresponding to the same ground truth drawn at the centered position in the coordinate system. Based on this, we construct a total of 1M plane geometry parsing data, as illustrated in Figure 6(b).

---

## Element: page_1_para_2

**Type:** heading  
**Page:** 1  

**Content:**

#### 3.4.3. General vision data

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

DeepEncoder can benefit from CLIP's pretraining gains and has sufficient parameters to incorporate general visual knowledge. Therefore, we also prepare some corresponding data for DeepSeek- OCR. Following DeepSeek- VL2 [40], we generate relevant data for tasks such as caption, detection, and grounding. Note that DeepSeek- OCR is not a general VLM model, and this portion of data accounts for only \(20\%\) of the total data. We introduce such type of data mainly to preserve the general vision interface, so that researchers interested in our model and general vision task can conveniently advance their work in the future.

---

## Element: page_1_para_4

**Type:** heading  
**Page:** 1  

**Content:**

#### 3.4.4. Text-only data

---

## Element: page_1_para_5

**Type:** text  
**Page:** 1  

**Content:**

To ensure the model's language capabilities, we introduced \(10\%\) of in- house text- only pretrain data, with all data processed to a length of 8192 tokens, which is also the sequence length for DeepSeek- OCR. In summary, when training DeepSeek- OCR, OCR data accounts for \(70\%\) , general vision data accounts for \(20\%\) , and text- only data accounts for \(10\%\) .

---

## Element: page_1_para_6

**Type:** heading  
**Page:** 1  

**Content:**

### 3.5. Training Pipelines

---

## Element: page_1_para_7

**Type:** text  
**Page:** 1  

**Content:**

Our training pipeline is very simple and consists mainly of two stages: a). Training DeepEncoder independently; b). Training the DeepSeek- OCR. Note that the Gundam- master mode is obtained by continuing training on a pre- trained DeepSeek- OCR model with 6M sampled data. Since the training protocol is identical to other modes, we omit the detailed description hereafter.

---

## Element: page_1_table_1

**Type:** table  
**Page:** 1  

**Content:**

Table 1 on page 1

---

## Element: page_1_para_1

**Type:** heading  
**Page:** 1  

**Content:**

#### 3.5.1. Training DeepEncoder

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

3.5.1. Training DeepEncoderFollowing Vary [36], we utilize a compact language model [15] and use the next token prediction framework to train DeepEncoder. In this stage, we use all OCR 1.0 and 2.0 data aforementioned, as well as 100M general data sampled from the LAION [31] dataset. All data is trained for 2 epochs with a batch size of 1280, using the AdamW [23] optimizer with cosine annealing scheduler [22] and a learning rate of 5e- 5. The training sequence length is 4096.

---

## Element: page_1_para_3

**Type:** heading  
**Page:** 1  

**Content:**

#### 3.5.2. Training DeepSeek-OCR

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

After DeepEncoder is ready, we use data mentioned in Section 3.4 to train the DeepSeek- OCR. with the entire training process conducted on the HAI- LLM [14] platform. The entire model uses pipeline parallelism (PP) and is divided into 4 parts, with DeepEncoder taking two parts and the decoder taking two parts. For DeepEncoder, we treat SAM and the compressor as the vision tokenizer, place them in PP0 and freeze their parameters, while treating the CLIP part as input embedding layer and place it in PP1 with unfrozen weights for training. For the language model part, since DeepSeek3B- MoE has 12 layers, we place 6 layers each on PP2 and PP3. We use 20 nodes (each with 8 A100- 40G GPUs) for training, with a data parallelism (DP) of 40 and a global batch size of 640. We use the AdamW optimizer with a step- based scheduler and an initial learning rate of 3e- 5. For text- only data, the training speed is 90B tokens/day, while for multimodal data, the training speed is 70B tokens/day.

---

## Element: page_1_para_5

**Type:** heading  
**Page:** 1  

**Content:**

### 4.1. Vision-text Compression Study

---

## Element: page_1_para_6

**Type:** text  
**Page:** 1  

**Content:**

4. Evaluation4.1. Vision- text Compression StudyWe select Fox [21] benchmarks to verify DeepSeek- OCR's compression- decompression capability for text- rich documents, in order to preliminarily explore the feasibility and boundaries of contexts optical compression. We use the English document portion of Fox, tokenize the ground truth text with DeepSeek- OCR's tokenizer (vocabulary size of approximately 129k), and select documents with 600- 1300 tokens for testing, which happens to be 100 pages. Since the number of text tokens is not large, we only need to test performance in Tiny and Small modes, where Tiny mode corresponds to 64 tokens and Small mode corresponds to 100 tokens. We use the prompt

---

## Element: page_1_table_1

**Type:** table  
**Page:** 1  

**Content:**

Table 1 on page 1

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

without layout: "&lt;image&gt;\nFree OCR." to control the model's output format. Nevertheless, the output format still cannot completely match Fox benchmarks, so the actual performance would be somewhat higher than the test results.

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

As shown in Table 2, within a \(10\times\) compression ratio, the model's decoding precision can reach approximately \(97\%\) , which is a very promising result. In the future, it may be possible to achieve nearly \(10\times\) lossless contexts compression through text- to- image approaches. When the compression ratio exceeds \(10\times\) , performance begins to decline, which may have two reasons: one is that the layout of long documents becomes more complex, and another reason may be that long texts become blurred at \(512\times 512\) or \(640\times 640\) resolution. The first issue can be solved by rendering texts onto a single layout page, while we believe the second issue will become

---

## Element: page_1_table_1

**Type:** table  
**Page:** 1  

**Content:**

Table 1 on page 1

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

a feature of the forgetting mechanism. When compressing tokens by nearly \(20\times\) , we find that precision can still approach \(60\%\) . These results indicate that optical contexts compression is a very promising and worthwhile research direction, and this approach does not bring any overhead because it can leverage VLM infrastructure, as multimodal systems inherently require an additional vision encoder.

---

## Element: page_1_para_2

**Type:** heading  
**Page:** 1  

**Content:**

# 4.2. OCR Practical Performance

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

DeepSeek-OCR is not only an experimental model; it has strong practical capabilities and can construct data for LLM/VLM pretraining. To quantify OCR performance, we test DeepSeek-OCR on OmniDocBench [27], with results shown in Table 3. Requiring only 100 vision tokens (640x640 resolution), DeepSeek-OCR surpasses GOT-OCR2.0 [38] which uses 256 tokens; with 400 tokens (285 valid tokens, 1280x1280 resolution), it achieves on-par performance with state-of-the-arts on this benchmark. Using fewer than 800 tokens (Gundam mode), DeepSeek-OCR outperforms MinerU2.0 [34] which needs nearly 7,000 vision tokens. These results demonstrate that our DeepSeek-OCR model is powerful in practical applications, and because the higher tokens compression, it enjoys a higher research ceiling.

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

As shown in Table 4, some categories of documents require very few tokens to achieve satisfactory performance, such as slides which only need 64 vision tokens. For book and report documents, DeepSeek-OCR can achieve good performance with only 100 vision tokens.Combined with the analysis from Section 4.1, this may be because most text tokens in these document categories are within 1,000, meaning the vision-token compression ratio does not exceed 10x. For newspapers, Gundam or even Gundam-master mode is required to achieve acceptable edit distances, because the text tokens in newspapers are 4-5,000, far exceeding the 10x compression of other modes. These experimental results further demonstrate the boundaries of contexts optical compression, which may provide effective references for researches on the vision token optimization in VLMs and context compression, forgetting mechanisms in LLMs.

---

## Element: page_1_para_5

**Type:** heading  
**Page:** 1  

**Content:**

# 4.3. Qualitative Study

---

## Element: page_1_para_6

**Type:** heading  
**Page:** 1  

**Content:**

## 4.3.1. Deep parsing

---

## Element: page_1_para_7

**Type:** text  
**Page:** 1  

**Content:**

DeepSeek-OCR possesses both layout and OCR 2.0 capabilities, enabling it to further parse images within documents through secondary model calls, a feature we refer to as "deep parsing".As shown in Figures 7,8,9,10, our model can perform deep parsing on charts, geometry, chemical formulas, and even natural images, requiring only a unified prompt.

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| In the field of financial research reports, the deep parsing mode of DeepSeek-OCR can be used to obtain structured results of charts within documents. Charts are a crucial form of data representation in finance and scientific f

VLM description: The plot is a bar chart with two horizontal axes. The x-axis represents the number of charts, and the y-axis represents the number of charts per page. The y-axis is labeled "Number of Charts Per Page" and has a range of 0 to 100. The x-axis is labeled "Number of Charts" and has a range of 0 to 100. The chart shows a positive correlation between the number of charts and the number of charts per page, with a trend line that shows a slight upward slope.

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| For books and articles, the deep parsing mode can output dense captions for

VLM description: The plot is a bar chart with two sets of bars, each representing a different category. The x-axis is labeled "Categories" and has multiple categories listed, including "Books" and "Articles." The y-axis is labeled "Number of Documents" and has a range of values from 0 to 100. The bars are colored in different shades of blue and green, with the blue bars representing the number of documents for the "Books" category and the green bars representing the number of documents for the "Articles" category. The chart appears to be a comparison of the number of documents across different categories.

---

## Element: page_1_fig_2

**Type:** figure  
**Page:** 1  

**Content:**

| For books and articles, the deep parsing mode can output dense captions for natural images in the documents. With just a prompt, the model can automatical

VLM description: The plot is a bar chart with two axes. The x-axis represents the number of vision tokens, ranging from 1 to 100, while the y-axis represents the percentage of vision tokens that are correctly classified. The chart shows that the model can classify vision tokens with a high accuracy, achieving a precision of 97% and a recall of 99%. The model is able to classify vision tokens with a high precision, but its recall is lower. The model is able to classify vision tokens with a high precision, but its recall is lower. The model is able to classify vision tokens with a high precision, but its recall is lower. The model is able to classify vision tokens with a high precision, but its recall is lower. The model is able to classify vision tokens with a high precision, but its recall is lower. The model is able to classify vision tokens with a high precision, but its recall is lower. The model is able to classify vision tokens with a

---

## Element: page_1_fig_3

**Type:** figure  
**Page:** 1  

**Content:**

| For books and articles, the deep parsing mode can output dense captions for natural images in the documents. With just a prompt, the model can automatically identify what type of image it is and output the required results.

VLM description: The plot is a bar chart comparing the performance of different OCR models on a dataset of images. The x-axis represents the number of vision tokens per page, while the y-axis represents the accuracy and compression of the OCR models. The models are sorted by their performance in terms of accuracy and compression, with the highest accuracy and compression achieved by DeepSeek-OCR (Large) and DeepSeek-OCR (Tiny) models. The chart also shows that the models perform better with more vision tokens per page, but the accuracy and compression decrease as the number of vision tokens per page increases.

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| DeepSeek-OCR in deep parsing mode can also recognize chemical formulas within chemical documents and convert them to SMILES format. In the future, OCR 1.0+2.0 technology may play a significant role in the development of VLM/LLM

VLM description: The plot is a bar chart with two sets of bars representing different OCR technologies. The x-axis represents OCR technologies, and the y-axis represents the number of documents processed. The bars are colored differently, with one set in blue and the other in red. The blue bars are higher than the red bars, indicating that OCR 1.0+2.0 technology has processed more documents than OCR 1.0 technology. The chart does not show any data trends or patterns, and the variable relationships are not clear from the visual content.

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| DeepSeek-OCR also possesses the capability to copy (structure) simple planar geometric figures. Due to the intricate interdependencies among line segments in geometric shapes, parsing geometry task is extremely challenging and

VLM description: The plot is a bar chart with two sets of data represented by different colors. The x-axis is labeled "Number of Points" and ranges from 1 to 10. The y-axis is labeled "Percentage of Points," ranging from 0% to 100%. The data is plotted with a blue bar for the first set of points and a red bar for the second set. The blue bar is consistently higher than the red bar, indicating that the first set of points has a higher percentage of points than the second set.

---

## Element: page_1_para_1

**Type:** heading  
**Page:** 1  

**Content:**

#### 4.3.2. Multilingual recognition

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

PDF data on the Internet contains not only Chinese and English, but also a large amount of multilingual data, which is also crucial when training LLMs. For PDF documents, DeepSeek- OCR can handle nearly 100 languages. Like Chinese and English documents, multilingual data also supports both layout and non- layout OCR formats. The visualization results are shown in Figure 11, where we select Arabic and Sinhala languages to demonstrate results.

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| To endow the capability of processing widely crawled PDFs (multilingual data), we train our model with OCR capabilities for nearly 100 languages. Minority language documents can also support both layout and non-layout outpu

VLM description: The plot is a bar chart with two sets of bars, each representing a different language. The x-axis is labeled "Language," and the y-axis is labeled "Number of Documents." The bars are colored in different shades of blue, with the color representing the language. The chart shows that the number of documents for each language is relatively low, with the majority of the data points falling in the range of 10 to 50. The chart does not provide any specific data points or labels for each bar, but it clearly shows the distribution of documents across different languages.

---

## Element: page_1_para_1

**Type:** heading  
**Page:** 1  

**Content:**

## 4.3.3. General vision understanding

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

We also provide DeepSeek-OCR with a certain degree of general image understanding capabilities. The related visualization results are shown in Figure 12.

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| We retain DeepSeek-OCR's capabilities in general visual understanding, mainly including image description, object detection, grounding, etc. Meanwhile, due to the inclusion of text-only data, DeepSeek-OCR's language capabiliti

VLM description: The plot is a scientific figure from a research paper. It shows a bar graph with three horizontal bars representing different groups. The x-axis is labeled "Group" and has three categories: Group A, Group B, and Group C. The y-axis is labeled "Number of Observations" and has a range of 0 to 100. The bars are colored in red, blue, and green, respectively. The graph appears to be a comparison of the number of observations across the three groups.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

Our work represents an initial exploration into the boundaries of vision-text compression, investigating how many vision tokens are required to decode \(N\) text tokens. The preliminary results are encouraging: DeepSeek- OCR achieves near- lossless OCR compression at approximately \(10 \times\) ratios, while \(20 \times\) compression still retains \(60 \%\) accuracy. These findings suggest promising directions for future applications, such as implementing optical processing for dialogue histories beyond \(k\) rounds in multi- turn conversations to achieve \(10 \times\) compression efficiency.

---

## Element: page_1_fig_1

**Type:** figure  
**Page:** 1  

**Content:**

| Forgetting mechanisms constitute one of the most fundamental characteristics of human memory. The contexts optical compression approach can simulate this mechanism by rendering previous rounds of historical text onto images fo

VLM description: The plot is a scientific figure that illustrates the concept of contexts optical compression. It shows a series of images, each representing a different round of historical text. The x-axis represents the number of rounds, and the y-axis represents the image. The data trends show that as the number of rounds increases, the images become more complex and detailed, indicating the progression of historical text. The figure also highlights the key observation that the images become more similar to each other as the number of rounds increases, suggesting that the context optical compression approach can simulate the forgetting mechanism of human memory.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

For older contexts, we could progressively downsizing the rendered images to further reduce token consumption. This assumption draws inspiration from the natural parallel between human memory decay over time and visual perception degradation over spatial distance—both exhibit similar patterns of progressive information loss, as shown in Figure 13. By combining these mechanisms, contexts optical compression method enables a form of memory decay that mirrors biological forgetting curves, where recent information maintains high fidelity while distant memories naturally fade through increased compression ratios.

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

While our initial exploration shows potential for scalable ultra- long context processing, where recent contexts preserve high resolution and older contexts consume fewer resources, we acknowledge this is early- stage work that requires further investigation. The approach suggests a path toward theoretically unlimited context architectures that balance information retention with computational constraints, though the practical implications and limitations of such vision- text compression systems warrant deeper study in future research.

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

In this technical report, we propose DeepSeek- OCR and preliminarily validate the feasibility of contexts optical compression through this model, demonstrating that the model can effectively decode text tokens exceeding 10 times the quantity from a small number of vision tokens. We believe this finding will facilitate the development of VLMs and LLMs in the future. Additionally, DeepSeek- OCR is a highly practical model capable of large- scale pretraining data production, serving as an indispensable assistant for LLMs. Of course, OCR alone is insufficient to fully validate true context optical compression and we will conduct digital- optical text interleaved pretraining, needle- in- a- haystack testing, and other evaluations in the future. From another perspective, optical contexts compression still offers substantial room for research and improvement, representing a promising new direction.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

[1] Marker. URL https://github.com/datalab- to/marker. [2] Mathpix. URL https://mathpix.com/. [3] Ocrflux, 2025. URL https://github.com/chatdoc- com/OCRFlux. [4] G. AI. Gemini 2.5- pro, 2025. URL https://gemini.google.com/. [5] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin. Qwen2.5- vl technical report. arXiv preprint arXiv:2502.13923, 2025. [6] L. Blecher, G. Cucurull, T. Scialom, and R. Stojnic. Nougat: Neural optical understanding for academic documents. arXiv preprint arXiv:2308.13418, 2023. [7] J. Chen, L. Kong, H. Wei, C. Liu, Z. Ge, L. Zhao, J. Sun, C. Han, and X. Zhang. Onechart: Purify the chart structural extraction via one auxiliary token. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 147- 155, 2024. [8] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma, et al. How far are we to gpt- 4v? closing the gap to commercial multimodal models with open- source suites. arXiv preprint arXiv:2404.16821, 2024. [9] C. Cui, T. Sun, M. Lin, T. Gao, Y. Zhang, J. Liu, X. Wang, Z. Zhang, C. Zhou, H. Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025. [10] M. Dehghani, J. Djolonga, B. Mustafa, P. Padlewski, J. Heek, J. Gilmer, A. Steiner, M. Caron, R. Geirhos, I. Alabdulmohsin, et al. Patch n' pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:3632- 3656, 2023. [11] H. Feng, S. Wei, X. Fei, W. Shi, Y. Han, L. Liao, J. Lu, B. Wu, Q. Liu, C. Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. arXiv preprint arXiv:2505.14059, 2025. [12] Y. Goyal, T. Khot, D. Summers- Stay, D. Batra, and D. Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904- 6913, 2017. [13] J. Gu, X. Meng, G. Lu, L. Hou, N. Minzhe, X. Liang, L. Yao, R. Huang, W. Zhang, X. Jiang, et al. Wukong: A 100 million large- scale chinese cross- modal pre- training benchmark. Advances in Neural Information Processing Systems, 35:26418- 26431, 2022. [14] High- flyer. HAI- LLM: Efficient and lightweight training tool for large models, 2023. URL https://www.high- flyer.cn/en/blog/hai- llm.[15] S. Iyer, X. V. Lin, R. Pasunuru, T. Mihaylov, D. Simig, P. Yu, K. Shuster, T. Wang, Q. Liu, P. S. Koura, et al. Opt- iml: Scaling language model instruction meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017, 2022. [16] S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787- 798, 2014.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

[17] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.- Y. Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023.

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

[18] Z. Li, Y. Liu, Q. Liu, Z. Ma, Z. Zhang, S. Zhang, Z. Guo, J. Zhang, X. Wang, and X. Bai. Monkeyocr: Document parsing with a structure-recognition- relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

[19] A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, C. Zhao, C. Dengr, C. Ruan, D. Dai, D. Guo, et al. Deepseek- v2: A strong, economical, and efficient mixture- of- experts language model. arXiv preprint arXiv:2405.04434, 2024.

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

[20] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek- v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

---

## Element: page_1_para_5

**Type:** text  
**Page:** 1  

**Content:**

[21] C. Liu, H. Wei, J. Chen, L. Kong, Z. Ge, Z. Zhu, L. Zhao, J. Sun, C. Han, and X. Zhang. Focus anywhere for fine- grained multi- page document understanding. arXiv preprint arXiv:2405.14295, 2024.

---

## Element: page_1_para_6

**Type:** text  
**Page:** 1  

**Content:**

[22] I. Loshchilov and F. Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

---

## Element: page_1_para_7

**Type:** text  
**Page:** 1  

**Content:**

[23] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In ICLR, 2019.

---

## Element: page_1_para_8

**Type:** text  
**Page:** 1  

**Content:**

[24] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

---

## Element: page_1_para_9

**Type:** text  
**Page:** 1  

**Content:**

[25] A. Nassar, A. Marafioti, M. Omenetti, M. Lysak, N. Livathinos, C. Auer, L. Morin, R. T. de Lima, Y. Kim, A. S. Gurbuz, et al. Smoldocling: An ultra- compact vision- language model for end- to- end multi- modal document conversion. arXiv preprint arXiv:2503.11576, 2025.

---

## Element: page_1_para_10

**Type:** text  
**Page:** 1  

**Content:**

[26] OpenAI. Gpt- 4 technical report, 2023.

---

## Element: page_1_para_11

**Type:** text  
**Page:** 1  

**Content:**

[27] L. Ouyang, Y. Qu, H. Zhou, J. Zhu, R. Zhang, Q. Lin, B. Wang, Z. Zhao, M. Jiang, X. Zhao, et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24838- 24848, 2025.

---

## Element: page_1_para_12

**Type:** text  
**Page:** 1  

**Content:**

[28] J. Poznanski, A. Rangapur, J. Borchardt, J. Dunkelberger, R. Huff, D. Lin, C. Wilhelm, K. Lo, and L. Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443, 2025.

---

## Element: page_1_para_13

**Type:** text  
**Page:** 1  

**Content:**

[29] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748- 8763. PMLR, 2021.

---

## Element: page_1_para_14

**Type:** text  
**Page:** 1  

**Content:**

[30] Rednote. dots.ocr, 2025. URL https://github.com/rednote- hilab/dots.ocr.

---

## Element: page_1_para_15

**Type:** text  
**Page:** 1  

**Content:**

[31] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki. Laion- 400m: Open dataset of clip- filtered 400 million image- text pairs. arXiv preprint arXiv:2111.02114, 2021.

---

## Element: page_1_para_1

**Type:** text  
**Page:** 1  

**Content:**

[32] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317- 8326, 2019.

---

## Element: page_1_para_2

**Type:** text  
**Page:** 1  

**Content:**

[33] T. Sun, C. Cui, Y. Du, and Y. Liu. Pp- dcolayout: A unified document layout detection model to accelerate large- scale data construction. arXiv preprint arXiv:2503.17213, 2025.

---

## Element: page_1_para_3

**Type:** text  
**Page:** 1  

**Content:**

[34] B. Wang, C. Xu, X. Zhao, L. Ouyang, F. Wu, Z. Zhao, R. Xu, K. Liu, Y. Qu, F. Shang, et al. Mineru: An open- source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.

---

## Element: page_1_para_4

**Type:** text  
**Page:** 1  

**Content:**

[35] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. Qwen2- vl: Enhancing vision- language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

---

## Element: page_1_para_5

**Type:** text  
**Page:** 1  

**Content:**

[36] H. Wei, L. Kong, J. Chen, L. Zhao, Z. Ge, J. Yang, J. Sun, C. Han, and X. Zhang. Vary: Scaling up the vision vocabulary for large vision- language model. In European Conference on Computer Vision, pages 408- 424. Springer, 2024.

---

## Element: page_1_para_6

**Type:** text  
**Page:** 1  

**Content:**

[37] H. Wei, L. Kong, J. Chen, L. Zhao, Z. Ge, E. Yu, J. Sun, C. Han, and X. Zhang. Small language model meets with reinforced vision vocabulary. arXiv preprint arXiv:2401.12503, 2024.

---

## Element: page_1_para_7

**Type:** text  
**Page:** 1  

**Content:**

[38] H. Wei, C. Liu, J. Chen, J. Wang, L. Kong, Y. Xu, Z. Ge, L. Zhao, J. Sun, Y. Peng, et al. General ocr theory: Towards ocr- 2.0 via a unified end- to- end model. arXiv preprint arXiv:2409.01704, 2024.

---

## Element: page_1_para_8

**Type:** text  
**Page:** 1  

**Content:**

[39] H. Wei, Y. Yin, Y. Li, J. Wang, L. Zhao, J. Sun, Z. Ge, X. Zhang, and D. Jiang. Slow perception: Let's perceive geometric figures step- by- step. arXiv preprint arXiv:2412.20631, 2024.

---

## Element: page_1_para_9

**Type:** text  
**Page:** 1  

**Content:**

[40] Z. Wu, X. Chen, Z. Pan, X. Liu, W. Liu, D. Dai, H. Gao, Y. Ma, C. Wu, B. Wang, et al. Deepseek- vl2: Mixture- of- experts vision- language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024.

---

## Element: page_1_para_10

**Type:** text  
**Page:** 1  

**Content:**

[41] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang. Mm- vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

---

## Element: page_1_para_11

**Type:** text  
**Page:** 1  

**Content:**

[42] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, et al. Internvl3: Exploring advanced training and test- time recipes for open- source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

---

