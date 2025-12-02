# Complete Word-by-Word Speech Script
## Real-time Voice Phishing Detection & Speaker Biometrics
**Duration: 20 minutes**

---

## **SLIDE 1: Title Page** *(1 minute)*

Good morning everyone, and thank you for being here today. I'm excited to present my Bachelor's Thesis Project on "Real-time Voice Phishing Detection and Speaker Biometrics."

This research addresses one of the most pressing cybersecurity challenges of our time. As we've all witnessed, cybercriminals are constantly evolving their tactics. They're moving beyond traditional email phishing attacks to something far more sophisticated and dangerous - real-time voice phishing, or what we call "vishing."

My work combines cutting-edge natural language processing with deep learning-based speaker biometrics to create a comprehensive defense system. The goal is simple but ambitious: protect users from sophisticated voice attacks while maintaining their privacy and providing lightning-fast response times.

---

## **SLIDE 2: Contents** *(30 seconds)*

Let me quickly walk you through today's agenda. We'll start by understanding why voice phishing is such a critical threat right now. Then I'll show you our system architecture and dive deep into two main components: our phishing detection model using BiLSTM networks, and our novel CNN-based speaker biometrics system. We'll look at experimental results, discuss backend integration, and explore future research directions. I've reserved time at the end for your questions.

---

## **SLIDE 3: Motivation - Rising Voice Phishing** *(1.5 minutes)*

So why does this problem demand our immediate attention? The answer lies in a fundamental shift happening in cybercrime right now.

For years, we've been fighting email and SMS phishing. And honestly, we've gotten pretty good at it. Spam filters catch most malicious emails. Users are more aware of suspicious links. But here's the problem - cybercriminals aren't standing still.

They're pivoting to voice channels, and this shift is strategic. Think about it - when someone calls you claiming to be from your bank, there's an immediate sense of urgency and legitimacy that's much harder to achieve through a text message. The caller can adapt their approach in real-time based on how you respond. They can sound official, use technical jargon, and create pressure that makes even tech-savvy people vulnerable.

The technical challenge this creates is enormous. Unlike email phishing, which we can analyze at our leisure using static content analysis, voice phishing happens in real-time. We need systems that can process audio, understand context, and make security decisions within seconds while the call is still happening.

Our goal is ambitious but absolutely necessary: build a low-latency pipeline that can detect malicious intent in live conversations and protect users from these increasingly sophisticated attacks.

---

## **SLIDE 4: Evidence - Incident Statistics** *(1 minute)*

Let me show you just how serious this threat has become. These numbers come from major cybersecurity firms tracking incident reports across multiple industries.

Look at this trend: we've gone from approximately eight hundred seventy-seven thousand incidents to over nine hundred thirty-two thousand attacks in just one year. That's a twenty-eight percent increase. Current projections suggest we're approaching nearly one million voice phishing incidents annually.

And these aren't just robocalls asking about your car warranty. These are sophisticated, targeted attacks where criminals impersonate banks, government agencies, or even your own IT department. The financial losses and personal data breaches from these attacks are staggering.

This rising trend isn't just a statistic - it's a clear signal that we need immediate investment in both engineering solutions and fundamental research into advanced detection mechanisms.

---

## **SLIDE 5: System Goal and High-level Workflow** *(1.5 minutes)*

Given this threat landscape, our system is designed around three core objectives. First, we need to detect voice phishing in live calls with minimal latency. We're targeting sub-second response times because every second counts when someone is being manipulated. Second, we must maintain extremely high precision to minimize false positives, because incorrectly flagging legitimate calls will erode user trust and make the system unusable. Third, we absolutely must preserve user privacy throughout the entire process.

Here's how our pipeline works: client applications chunk incoming audio and transmit it to our system. The audio immediately undergoes speech-to-text conversion using OpenAI's Whisper model, which we chose for its exceptional robustness across different accents, audio qualities, and background noise conditions. The resulting transcript feeds into our phishing detection model for linguistic content analysis.

But here's the key innovation - simultaneously, we're extracting speaker biometric features for trusted caller identification. This parallel processing is crucial because it allows us to potentially bypass computationally expensive analysis for callers we already know and trust.

Our design priorities reflect real-world deployment constraints. Privacy means audio is processed ephemerally and never stored permanently on our servers. Efficiency means trusted callers can skip heavy computational analysis, improving their experience. And interpretability means our attention mechanism can show users exactly which words or phrases triggered security alerts, so they understand why the system made its decision.

---

## **SLIDE 6: System Diagram** *(1 minute)*

This architecture diagram shows you our complete system flow. As you can see, audio enters through two parallel processing paths: the transcript path for analyzing what's being said, and the biometric path for identifying who's saying it.

The speech-to-text component uses Whisper for robust transcription. That transcript feeds into our BiLSTM-based phishing detector, which I'll explain in detail shortly. At the same time, we're extracting MFCC features from the raw audio and generating speaker embeddings using our CNN model.

The decision engine is really the brain of our system. It combines signals from both the phishing detector and the speaker verification system. If a caller is identified as trusted with high confidence, we can bypass or significantly reduce the sensitivity of phishing detection. For unknown callers, we apply full analysis with standard security thresholds.

This dual-path approach gives us the best of both worlds - improved performance and better user experience, while never compromising on security.

---

## **SLIDE 7: Why 1D-CNN + BiLSTM + Attention?** *(1.5 minutes)*

Now let's dive into our phishing detection model. We chose a hybrid architecture that combines three complementary deep learning techniques, and each one addresses specific challenges in voice phishing detection.

First, our one-D CNN layers extract local n-gram-like features. Think of these as suspicious phrases and linguistic patterns that phishers commonly use. Phrases like "verify your account immediately" or "urgent security update required" or "we've detected suspicious activity." The convolutional layers learn to recognize these local patterns regardless of where they appear in the conversation.

Second, our BiLSTM networks capture long-range context and dependencies. This is absolutely crucial because phishing attempts often involve setup phrases early in the call that only become suspicious when combined with later requests. For example, "I'm calling from your bank's security department" might seem innocent at first, but becomes highly suspicious when followed twenty seconds later by "I need you to confirm your PIN number." The bidirectional nature allows our model to consider both past and future context when evaluating any given word or phrase.

Third, our attention mechanism focuses the model on the most relevant words and phrases. This serves two critical purposes: it improves model performance by highlighting the most important features, and it provides interpretability. We can show users exactly which parts of the conversation triggered the security alert.

We use FastText embeddings as our foundation because they handle rare and out-of-vocabulary words exceptionally well. This is essential since phishers often use uncommon terminology or technical jargon to sound official and authoritative.

---

## **SLIDE 8: Detailed Architecture** *(1.5 minutes)*

Let me walk you through the specific architectural details of our phishing detection model. We start with three-hundred-dimensional FastText embeddings, which provide rich semantic representations of words while effectively handling vocabulary variations.

The one-D CNN layer uses kernel size three with sixty-four filters, followed by ReLU activation and max pooling with size two. This extracts local linguistic features while reducing dimensionality for computational efficiency.

The BiLSTM is really the heart of our model. We use sixty-four forward hidden units and thirty-two backward hidden units. This asymmetry reflects the fact that in conversational analysis, future context is often less informative than past context. We apply zero-point-one dropout for regularization to prevent overfitting on our training data.

The attention mechanism computes weights over all BiLSTM outputs, creating a weighted context vector that emphasizes the most relevant parts of the input sequence. This context vector then feeds into our final dense classifier for binary phishing detection.

The entire architecture is carefully designed to balance model capacity with computational efficiency, enabling real-time inference while maintaining high accuracy. Every layer serves a specific purpose in the detection pipeline.

---

## **SLIDE 9: BiLSTM Code Implementation** *(1.5 minutes)*

This PyTorch implementation shows our PhishingDetector class in detail. Let me walk you through the key components and explain why each part is necessary.

In the constructor, we define our embedding layer, one-D convolution, bidirectional LSTM, attention mechanism, and final classifier. Notice the careful dimension management - the BiLSTM outputs have dimension hidden_dim times two due to the bidirectional nature.

The forward pass demonstrates the complex data flow through our network. We start with token embeddings, then transpose for the CNN layer which expects channels-first format. After convolution and ReLU activation, we transpose back for the LSTM which expects batch-first format.

The attention mechanism is implemented as a linear layer that computes scalar weights for each LSTM output. We apply softmax to normalize these weights, ensuring they sum to one, then compute a weighted sum to create our final context vector.

This implementation handles all the complex tensor manipulations required when combining CNN and RNN layers, ensuring proper data flow throughout the network. Getting these tensor shapes right was actually one of the trickier parts of the implementation.

---

## **SLIDE 10: Training Strategies** *(1 minute)*

Training this model presented several methodological challenges that required careful consideration. Our initial approach using standard seventy-fifteen-fifteen train-validation-test splits produced frustratingly unstable results, particularly on smaller datasets where individual samples could dramatically impact performance metrics.

We solved this with five-fold stratified cross-validation, which provided much more reliable performance estimates and reduced variance by approximately eighteen percent. Stratification ensures balanced class distribution across all folds, which is absolutely crucial for binary classification tasks like phishing detection.

We employed multiple regularization techniques: dropout layers within the model architecture, early stopping with patience of ten epochs to prevent overfitting, and careful learning rate scheduling using step decay to ensure stable convergence.

One critical lesson we learned is that data processing consistency is absolutely essential. Tokenization and embedding alignment must be identical between training and inference, or model performance degrades significantly.

---

## **SLIDE 11: Training Code Implementation** *(1 minute)*

This implementation demonstrates our cross-validation training setup. We use StratifiedKFold from scikit-learn to ensure balanced class distribution across all folds, which prevents any single fold from being dominated by one class.

For each fold, we initialize a completely fresh model instance, configure the Adam optimizer with learning rate zero-point-zero-zero-one, and set up step learning rate scheduling. The training loop implements early stopping based on validation loss - if validation loss doesn't improve for ten consecutive epochs, we terminate training to prevent overfitting.

This approach provides robust performance estimates and helps us identify the optimal model configuration across different data splits. The cross-validation results give us much more confidence in our model's ability to generalize to unseen data, which is critical for real-world deployment.

---

## **SLIDE 12: Experimental Results** *(1 minute)*

Our experimental results demonstrate promising performance across different evaluation scenarios. Published literature using similar architectures reports accuracy around ninety-nine-point-three-two percent and F1-score of ninety-nine-point-three-one percent on larger, well-curated datasets.

Our implementation achieved an F1-score of zero-point-nine-one on our internal dataset of eighty samples. While this is lower than literature results, it's important to understand that performance was highly sensitive to data splits on smaller datasets, which actually validates our decision to use cross-validation.

The cross-validation approach significantly improved recall and reduced variance across different data partitions. This demonstrates the real value of robust evaluation methodologies, particularly when working with limited training data - which is unfortunately a common challenge in cybersecurity applications where labeled phishing examples are naturally scarce.

---

## **SLIDE 13: CNN Speaker Biometrics - Technical Architecture** *(1.5 minutes)*

Now let's dive into our speaker biometrics system - this is really where our implementation makes its most significant contributions to the field. The entire system is designed for real-time operation with minimal computational overhead.

Our input consists of raw audio waveforms sampled at sixteen kilohertz in three-second windows. This duration represents a careful balance - shorter windows don't contain sufficient information for reliable speaker identification, while longer windows introduce unacceptable latency for real-time applications.

Feature extraction produces thirty-nine-dimensional MFCC features: thirteen base coefficients plus delta and delta-delta features. MFCCs are well-established in the speaker recognition field for their robustness to noise and computational efficiency.

The CNN architecture consists of three convolutional blocks with batch normalization, specifically designed to learn hierarchical representations of individual speaker characteristics. The final output is a one-hundred-twenty-eight-dimensional speaker embedding vector - compact enough for efficient storage and comparison, yet rich enough to capture the unique vocal characteristics that distinguish one speaker from another.

Speaker verification uses cosine similarity between embeddings, providing fast, reliable identification with a simple distance metric that's both computationally efficient and theoretically well-founded.

---

## **SLIDE 14: MFCC Feature Extraction Pipeline** *(1.5 minutes)*

Our feature extraction pipeline is comprehensive and designed for robustness across different recording conditions. We begin with audio preprocessing including normalization and silence removal to ensure consistent input characteristics regardless of the recording environment.

Windowing uses twenty-five-millisecond Hamming windows with ten-millisecond hop length. These parameters are industry standard in speech processing and provide an excellent time-frequency resolution trade-off for speaker recognition tasks.

MFCC computation extracts thirteen base coefficients representing the spectral envelope characteristics that are most relevant for speaker identification. These coefficients capture the shape of the spectral envelope while being relatively invariant to background noise and channel effects.

Delta features - which are first derivatives - represent the velocity of spectral changes over time. Delta-delta features - second derivatives - represent acceleration of these changes. These temporal derivatives capture the dynamic characteristics of speech that are absolutely crucial for distinguishing between different speakers.

Mean-variance normalization per utterance ensures consistent feature scaling across different recording conditions and speakers, which is essential for reliable CNN training and inference.

For three-second audio at sixteen kilohertz, we typically obtain three hundred time frames, resulting in a three-hundred-by-thirty-nine feature matrix that captures both spectral and temporal characteristics of the speaker's unique voice.

---

## **SLIDE 15: MFCC Feature Extraction Code** *(1 minute)*

This implementation uses librosa for robust and efficient MFCC extraction. The function takes raw audio and sampling rate as inputs, with configurable MFCC coefficient count for flexibility.

We extract base MFCCs using librosa's highly optimized implementation, then compute delta and delta-delta features using the delta function with appropriate order parameters. The key insight here is proper feature stacking - we vertically stack all three feature types to create our comprehensive thirty-nine-dimensional feature vector.

Transposition to time-first format and per-utterance normalization ensure perfect compatibility with our CNN architecture and consistent feature scaling. This preprocessing step is absolutely critical for model performance and must be identical between training and inference phases.

---

## **SLIDE 16: CNN Architecture Design** *(1.5 minutes)*

Our CNN architecture follows a systematic design philosophy that's specifically optimized for speaker recognition tasks. The input layer accepts batches of time-frequency feature matrices with shape batch-size, time-frames, thirty-nine.

We use three convolutional blocks with progressively increasing filter counts: sixty-four, one-hundred-twenty-eight, and two-hundred-fifty-six filters respectively. This hierarchical approach allows the network to learn increasingly complex and abstract representations of speaker characteristics, from basic spectral patterns to high-level vocal signatures.

Each block includes two-D convolution with carefully chosen kernel sizes, batch normalization for training stability and faster convergence, ReLU activation for non-linearity, and max pooling for dimensionality reduction and translation invariance.

Global average pooling eliminates spatial dimensions while preserving channel information, making our network robust to variable input lengths. This is absolutely crucial for real-world deployment where audio segments may vary slightly in duration due to processing constraints.

The dense layers progressively reduce dimensionality from five-hundred-twelve to two-hundred-fifty-six to our final one-hundred-twenty-eight-dimensional embedding. Dropout between dense layers with probability zero-point-three prevents overfitting while maintaining sufficient model capacity for speaker discrimination.

---

## **SLIDE 17: CNN Speaker Model Implementation** *(1 minute)*

This PyTorch implementation shows our CNNSpeakerModel class structure in complete detail. The constructor defines all layers systematically: three convolutional blocks, each with convolution, batch normalization, and pooling layers arranged for optimal information flow.

Global average pooling is implemented using AdaptiveAvgPool2d, which automatically handles variable input sizes by computing the mean across spatial dimensions. This makes our model robust to slight variations in input length.

The dense layers create our embedding pathway with appropriate dropout for regularization. This modular design makes the architecture easy to understand, modify, and debug during development, which was crucial for iterative improvement of our system.

---

## **SLIDE 18: CNN Forward Pass Implementation** *(1 minute)*

The forward pass demonstrates the careful tensor manipulation required throughout our network. We add a channel dimension for two-D convolution compatibility, then systematically process through all convolutional blocks.

Each block applies convolution, batch normalization, ReLU activation, and max pooling in sequence. Global average pooling reduces spatial dimensions, and we flatten the result for the dense layers.

The final L2 normalization step is absolutely crucial - it ensures all embeddings lie on the unit sphere, making cosine similarity both computationally efficient and theoretically sound. This normalization step is essential for reliable speaker verification performance in production environments.

---

## **SLIDE 19: Training Strategy for Speaker CNN** *(1.5 minutes)*

Our training strategy uses triplet loss with margin zero-point-two, which directly optimizes the similarity metric we use for verification. This is much more effective than traditional classification approaches because triplet loss learns to minimize distance between same-speaker embeddings while maximizing distance between different speakers.

Each triplet consists of an anchor sample, a positive sample from the same speaker, and a negative sample from a different speaker. This approach directly optimizes our embedding space for speaker discrimination tasks.

Data augmentation is crucial for building robustness into our system. We apply time shifting within plus-or-minus zero-point-one seconds, noise addition with signal-to-noise ratio between fifteen and twenty-five decibels, and speed perturbation between zero-point-nine-x and one-point-one-x. These augmentations simulate real-world variations in recording conditions that our system will encounter in deployment.

Balanced batch sampling ensures each training batch contains multiple speakers with multiple utterances, enabling effective triplet mining during training. We use four speakers with four utterances each per batch, which provides sufficient diversity for robust learning.

The Adam optimizer with learning rate zero-point-zero-zero-one and weight decay one-e-minus-four provides stable training with appropriate regularization to prevent overfitting.

---

## **SLIDE 20: Triplet Loss Implementation** *(1 minute)*

This implementation computes triplet loss efficiently using PyTorch's pairwise distance function. We calculate Euclidean distances between anchor-positive pairs and anchor-negative pairs, then apply our margin-based loss function.

The ReLU function ensures we only penalize triplets where the positive distance exceeds the negative distance by less than our margin. This encourages the model to create clear separation between different speakers while keeping same-speaker embeddings close together in the embedding space.

The training loop shows how we generate embeddings for all three sample types and compute the loss. This direct optimization of our verification metric leads to significantly better real-world performance compared to traditional classification-based approaches.

---

## **SLIDE 21: Speaker Verification Pipeline** *(1.5 minutes)*

Our verification system operates in two distinct phases that are designed for practical deployment scenarios.

During the enrollment phase, users provide three to five utterances totaling fifteen to thirty seconds of speech. We extract MFCC features from each utterance, generate embeddings using our trained CNN, and store the averaged embedding as the user's biometric template. Averaging multiple embeddings provides robustness against individual recording variations and improves overall system reliability.

During the verification phase, we process incoming audio chunks in three-second windows, extract features using identical preprocessing steps, generate embeddings with the same CNN model, and compute cosine similarity with the stored template.

Threshold selection typically ranges from zero-point-six to zero-point-eight depending on security requirements. Higher thresholds provide better security against impersonation attacks but may increase false rejections of legitimate users, so this becomes a tunable parameter based on organizational risk tolerance.

Real-time optimization uses a sliding window approach for continuous verification throughout longer calls, providing ongoing confidence assessment rather than just single-point verification at the beginning of a call.

---

## **SLIDE 22: Speaker Verification Service** *(1 minute)*

This service class encapsulates our complete verification workflow in a production-ready interface. The constructor loads our trained model and sets verification parameters that can be adjusted based on deployment requirements.

The enroll-speaker method handles the entire enrollment process, processing multiple audio samples and averaging their embeddings for robust template creation. We include comprehensive error handling to ensure graceful degradation if individual samples fail processing due to audio quality issues.

The verify-speaker method performs real-time verification, handling feature extraction, embedding generation, and similarity computation in a single, efficient call. It returns both a boolean decision and the raw similarity score, allowing calling applications to implement their own confidence thresholds if needed.

This design enables easy integration into larger systems while maintaining clean separation of concerns and robust error handling.

---

## **SLIDE 23: Performance Metrics & Evaluation** *(1.5 minutes)*

Our evaluation used a custom dataset with fifty speakers and twenty utterances each, providing sufficient diversity for robust performance assessment across different vocal characteristics and recording conditions.

Our key metrics demonstrate competitive performance with commercial systems. Equal Error Rate of two-point-three percent is comparable to state-of-the-art speaker verification systems. False Acceptance Rate of zero-point-eight percent at one percent False Rejection Rate shows an excellent security-usability balance for practical deployment.

Processing performance easily meets real-time requirements: embedding extraction takes just forty-five milliseconds on CPU and twelve milliseconds on GPU. This enables deployment in various computational environments, from edge devices to cloud infrastructure.

Robustness testing validates our system's practical deployment viability. Ninety-four-point-two percent accuracy across different recording sessions demonstrates consistency over time. Eighty-nine-point-seven percent accuracy in noisy conditions at fifteen decibel signal-to-noise ratio shows resilience to real-world audio challenges. Ninety-one-point-five percent accuracy across different microphones proves the system works with various hardware configurations.

Memory footprint is minimal at just one-hundred-twenty-eight bytes per enrolled user, enabling scalable deployment even with very large user bases without significant storage concerns.

---

## **SLIDE 24: Integration with Phishing Detection** *(1 minute)*

Our integration strategy maximizes both security and efficiency through intelligent system coordination that leverages the strengths of both components.

Parallel processing runs speaker verification concurrently with speech-to-text conversion, minimizing overall system latency. Early bypass allows trusted speakers to skip computationally expensive phishing analysis, dramatically improving user experience for legitimate callers while maintaining security.

Risk scoring uses speaker confidence to intelligently modulate phishing detection thresholds. High speaker confidence can reduce sensitivity for known users, while unknown callers always receive full analysis with standard security thresholds.

Fallback strategies ensure security is never compromised. Unknown speakers always receive complete pipeline analysis, and any system failures default to maximum security posture. Privacy controls require explicit user consent for biometric enrollment, and users can opt out while maintaining phishing protection.

Performance impact analysis shows fifteen percent reduction in overall system latency for known callers, demonstrating clear efficiency benefits without any security trade-offs.

---

## **SLIDE 25: Backend Architecture** *(1 minute)*

Our backend follows a modular service architecture that enables independent scaling and maintenance of different system components. The main server handles API orchestration and request routing, while specialized services manage specific functionality domains.

The STT service uses Whisper for robust speech recognition across various audio conditions and languages. The ML model service handles phishing detection inference with proper model loading, caching, and version management. The speaker biometric service manages enrollment, verification, and secure template storage.

Audio preprocessing service handles format conversion, noise reduction, and feature extraction with optimized pipelines. This separation of concerns enables optimization of each component independently and greatly facilitates testing, debugging, and maintenance in production environments.

---

## **SLIDE 26: Runtime Decision Logic** *(1 minute)*

Our runtime system processes audio chunks through carefully orchestrated parallel pipelines designed for minimal latency and maximum reliability. Speech-to-text and phishing detection run concurrently with speaker verification to minimize total processing time.

The decision engine implements sophisticated, configurable logic that combines rule-based heuristics, machine learning predictions, and biometric confidence scores. Thresholds can be dynamically adjusted based on organizational risk tolerance, user preferences, and real-time threat intelligence.

Privacy preservation is built into every step - we process audio ephemerally and never store raw audio data, while persisting only mathematical embeddings with explicit user consent. This approach carefully balances security needs with privacy requirements and regulatory compliance.

---

## **SLIDE 27: Top Priorities** *(1 minute)*

Our immediate development priorities focus on production readiness and system robustness for real-world deployment.

Tokenizer alignment verification ensures consistent inference performance by confirming that vocabulary processing and embedding matrix preprocessing exactly matches our training procedures. Even small mismatches here can significantly degrade model performance.

Anti-spoofing detection implementation will add production-grade security against voice synthesis and replay attacks, which is absolutely essential for real-world deployment where attackers may use recorded audio or synthetic voices.

Comprehensive unit testing for CNN model input-output and MFCC feature extraction will ensure system reliability and facilitate continuous integration and deployment pipelines.

CNN inference optimization through quantization and pruning will enable edge deployment while maintaining accuracy, which is crucial for privacy-sensitive applications where users prefer local processing.

---

## **SLIDE 28: Future Work** *(1 minute)*

Our longer-term research directions focus on advancing the state-of-the-art in voice security systems and expanding system capabilities.

Attention-based speaker models like x-vectors and ECAPA-TDNN represent the current frontier in speaker recognition research and could significantly improve our system's accuracy and robustness across diverse populations and recording conditions.

Edge deployment with privacy-preserving speech-to-text would enable completely local processing, eliminating any privacy concerns about cloud-based audio processing while maintaining full functionality.

Semi-supervised continual learning from production false positives would allow our system to adapt and improve based on real-world deployment experience, becoming more accurate over time.

Multi-modal fusion combining voice biometrics with behavioral patterns could provide additional security layers and improved user experience through more comprehensive identity verification.

---

## **SLIDE 29: References** *(30 seconds)*

Our work builds on established research foundations in both phishing detection and speaker recognition. Key references include the BiLSTM and attention mechanisms from Boussougou and Park's work, x-vector embeddings from Snyder and colleagues, and ECAPA-TDNN architectures from Desplanques and team.

All project materials, implementation code, and experimental artifacts are available in our repository for complete reproducibility and to enable further research by the community.

---

## **SLIDE 30: Thank You** *(30 seconds)*

In conclusion, we've successfully developed a comprehensive system that combines advanced natural language processing techniques for phishing detection with robust CNN-based speaker biometrics. Our approach carefully balances security, privacy, and performance while providing interpretable results that users can understand and trust.

This system demonstrates significant potential for real-time voice security applications and opens several promising avenues for future research in multi-modal biometric systems and privacy-preserving security technologies.

Thank you very much for your attention and engagement throughout this presentation. I'm excited to answer any questions you might have about the implementation details, experimental results, or future directions of this research. Please, go ahead with your questions.

---

**[END OF PRESENTATION - READY FOR Q&A]**