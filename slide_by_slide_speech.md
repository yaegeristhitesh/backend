# Slide-by-Slide Speech Script: Real-time Voice Phishing Detection & Speaker Biometrics

**Total Duration: 20 minutes**

---

## **SLIDE 1: Title Page** *(1 minute)*

Good morning everyone. I'm excited to present my Bachelor's Thesis Project on "Real-time Voice Phishing Detection and Speaker Biometrics."

This work addresses a critical cybersecurity challenge that's becoming increasingly urgent in our digital age. As cybercriminals evolve their tactics, moving beyond traditional email phishing to sophisticated real-time voice attacks, we need intelligent systems that can protect users while maintaining their privacy and providing fast response times.

My research combines advanced natural language processing with deep learning-based speaker biometrics to create a comprehensive defense system against voice phishing attacks.

---

## **SLIDE 2: Contents** *(30 seconds)*

Today's presentation is structured into several key sections. I'll begin by establishing the motivation and scope of the rising voice phishing threat. Then I'll walk you through our system goals and architecture, dive deep into our phishing detection model using BiLSTM networks, explain our training methodology, present experimental results, and spend significant time on our novel CNN-based speaker biometrics implementation. Finally, I'll discuss backend integration and future research directions.

---

## **SLIDE 3: Motivation - Rising Voice Phishing** *(1.5 minutes)*

Let me start with why this problem demands immediate attention. We're witnessing a fundamental shift in cybercrime tactics. While email and SMS phishing have become increasingly detectable through spam filters and user awareness, attackers are pivoting to voice channels - what cybersecurity experts call "vishing" or voice phishing.

This shift is strategic. Voice calls create a sense of urgency and legitimacy that's harder to achieve through text-based communications. Attackers can adapt their approach in real-time based on victim responses, making these attacks particularly dangerous.

The technical challenge is significant: unlike email phishing, which can be analyzed at leisure using static content analysis, voice phishing happens in real-time. We need systems that can process audio, understand context, and make decisions within seconds while the call is still active.

Our goal is ambitious but necessary: build a low-latency pipeline that can detect malicious intent in live conversations and reduce user exposure to these sophisticated attacks.

---

## **SLIDE 4: Evidence - Incident Statistics** *(1 minute)*

The data validates our concern. Industry reports from major cybersecurity firms show alarming trends. Looking at incident statistics from recent years, we see a clear upward trajectory.

For example, documented cases have grown from approximately 877,536 incidents to over 932,923 attacks - that represents a 28% increase in just one year. Current projections suggest we're approaching nearly one million voice phishing incidents annually.

These aren't just robocalls or simple scams. These are sophisticated, targeted attacks where criminals impersonate banks, government agencies, or trusted contacts. The financial and personal data losses from these attacks justify immediate investment in both engineering solutions and fundamental research into advanced detection mechanisms.

---

## **SLIDE 5: System Goal and High-level Workflow** *(1.5 minutes)*

Our system architecture is designed around three core objectives. First, detect voice phishing in live calls with minimal latency - we're targeting sub-second response times. Second, maintain high precision to minimize false positives, because incorrectly flagging legitimate calls erodes user trust. Third, preserve user privacy throughout the entire process.

The pipeline operates as follows: client applications chunk incoming audio and transmit it to our system. The audio undergoes speech-to-text conversion using OpenAI's Whisper model, chosen for its robustness across different accents and audio qualities. The resulting transcript feeds into our phishing detection model for content analysis.

Simultaneously - and this is crucial for efficiency - we extract speaker biometric features for trusted caller identification. This parallel processing allows us to potentially bypass heavy analysis for known, trusted contacts.

Our design priorities reflect real-world deployment constraints: privacy means audio is processed ephemerally and never stored permanently; efficiency means trusted callers can skip computationally expensive analysis; and interpretability means our attention mechanism can show users exactly which words or phrases triggered security alerts.

---

## **SLIDE 6: System Diagram** *(1 minute)*

This architecture diagram illustrates our complete system flow. Audio enters through two parallel processing paths: the transcript path for linguistic content analysis and the biometric path for speaker identification.

The speech-to-text component uses Whisper for robust transcription. The transcript feeds into our BiLSTM-based phishing detector, which I'll detail shortly. Concurrently, we extract MFCC features from the raw audio and generate speaker embeddings using our CNN model.

The decision engine is the system's brain - it combines signals from both the phishing detector and speaker verification system. If a caller is identified as trusted with high confidence, we can bypass or reduce the sensitivity of phishing detection. For unknown callers, we apply full analysis with standard thresholds.

This dual-path approach significantly improves both performance and user experience while maintaining security.

---

## **SLIDE 7: Why 1D-CNN + BiLSTM + Attention?** *(1.5 minutes)*

Now let's examine our phishing detection model architecture. We chose a hybrid approach combining three complementary deep learning techniques, each addressing specific challenges in voice phishing detection.

First, 1D-CNN layers extract local n-gram-like features. These are essentially suspicious phrases and linguistic patterns that phishers commonly use - phrases like "verify your account immediately" or "urgent security update required." The convolutional layers learn to recognize these local patterns regardless of their position in the conversation.

Second, BiLSTM networks capture long-range context and dependencies. This is crucial because phishing attempts often involve setup phrases early in the call that only become suspicious when combined with later requests. The bidirectional nature allows the model to consider both past and future context when evaluating any given word or phrase.

Third, our attention mechanism focuses the model on the most relevant words and phrases. This serves dual purposes: it improves model performance by highlighting important features, and it provides interpretability - we can show users exactly which parts of the conversation triggered the alert.

We use FastText embeddings as our foundation because they handle rare and out-of-vocabulary words effectively. This is essential since phishers often use uncommon terminology or technical jargon to sound official and authoritative.

---

## **SLIDE 8: Detailed Architecture (BiLSTM-focused)** *(1.5 minutes)*

Here are the specific architectural details of our phishing detection model. We start with 300-dimensional FastText embeddings, which provide rich semantic representations of words while handling vocabulary variations effectively.

The 1D-CNN layer uses kernel size 3 with 64 filters, followed by ReLU activation and max pooling with size 2. This extracts local features while reducing dimensionality.

The BiLSTM is the model's core component. We use 64 forward hidden units and 32 backward hidden units - this asymmetry reflects that future context is often less informative than past context in conversational analysis. We apply 0.1 dropout for regularization to prevent overfitting.

The attention mechanism computes weights over all BiLSTM outputs, creating a weighted context vector that emphasizes the most relevant parts of the input sequence. This context vector feeds into our final dense classifier for binary phishing detection.

The entire architecture is designed to balance model capacity with computational efficiency, enabling real-time inference while maintaining high accuracy.

---

## **SLIDE 9: BiLSTM Code Implementation** *(1.5 minutes)*

This PyTorch implementation shows our PhishingDetector class in detail. Let me walk through the key components.

In the constructor, we define our embedding layer, 1D convolution, bidirectional LSTM, attention mechanism, and final classifier. Notice the careful dimension management - the BiLSTM outputs have dimension hidden_dim times 2 due to bidirectionality.

The forward pass demonstrates the data flow. We start with token embeddings, then transpose for the CNN layer which expects channels-first format. After convolution and ReLU activation, we transpose back for the LSTM which expects batch-first format.

The attention mechanism is implemented as a linear layer that computes scalar weights for each LSTM output. We apply softmax to normalize these weights, then compute a weighted sum to create our final context vector.

This implementation handles the complex tensor manipulations required when combining CNN and RNN layers, ensuring proper data flow throughout the network.

---

## **SLIDE 10: Training Strategies Used** *(1 minute)*

Training presented several methodological challenges that required careful consideration. Our initial approach using standard 70-15-15 train-validation-test splits produced unstable results, particularly on smaller datasets where individual samples could significantly impact performance metrics.

We addressed this with 5-fold stratified cross-validation, which provided more reliable performance estimates and reduced variance by approximately 18%. Stratification ensures balanced class distribution across all folds, which is crucial for binary classification tasks like phishing detection.

We employed multiple regularization techniques: dropout layers within the model architecture, early stopping with patience of 10 epochs to prevent overfitting, and careful learning rate scheduling using step decay to ensure stable convergence.

Data processing consistency proved critical - tokenization and embedding alignment must be identical between training and inference to maintain model performance.

---

## **SLIDE 11: Training Code Implementation** *(1 minute)*

This implementation demonstrates our cross-validation training setup. We use StratifiedKFold from scikit-learn to ensure balanced class distribution across all folds.

For each fold, we initialize a fresh model instance, configure the Adam optimizer with learning rate 0.001, and set up step learning rate scheduling. The training loop implements early stopping based on validation loss - if validation loss doesn't improve for 10 consecutive epochs, we terminate training to prevent overfitting.

This approach provides robust performance estimates and helps identify the optimal model configuration across different data splits. The cross-validation results give us confidence in our model's generalization capability.

---

## **SLIDE 12: Key Experimental Results** *(1 minute)*

Our experimental results demonstrate promising performance across different evaluation scenarios. Literature reports from similar architectures show accuracy around 99.32% and F1-score of 99.31% on larger, well-curated datasets.

Our implementation achieved an F1-score of 0.91 on our internal dataset of 80 samples. While this is lower than literature results, it's important to note that performance was highly sensitive to data splits on smaller datasets, which validates our decision to use cross-validation.

The cross-validation approach significantly improved recall and reduced variance across different data partitions. This demonstrates the value of robust evaluation methodologies, particularly when working with limited training data - a common challenge in cybersecurity applications where labeled phishing examples are scarce.

---

## **SLIDE 13: CNN Speaker Biometrics - Technical Architecture** *(1.5 minutes)*

Now let's dive into our speaker biometrics system - this is where our implementation makes significant contributions to the field. The system is designed for real-time operation with minimal computational overhead.

Our input consists of raw audio waveforms sampled at 16kHz in 3-second windows. This duration balances speaker identification accuracy with system responsiveness - shorter windows lack sufficient information, while longer windows introduce unacceptable latency.

Feature extraction produces 39-dimensional MFCC features: 13 base coefficients plus delta and delta-delta features. MFCCs are well-established in speaker recognition for their robustness and computational efficiency.

The CNN architecture consists of three convolutional blocks with batch normalization, designed to learn hierarchical representations of speaker characteristics. The output is a 128-dimensional speaker embedding vector - compact enough for efficient storage and comparison, yet rich enough to capture individual speaker characteristics.

Speaker verification uses cosine similarity between embeddings, providing fast, reliable identification with a simple distance metric that's both computationally efficient and theoretically well-founded.

---

## **SLIDE 14: MFCC Feature Extraction Pipeline** *(1.5 minutes)*

Our feature extraction pipeline is comprehensive and robust. We begin with audio preprocessing including normalization and silence removal to ensure consistent input characteristics.

Windowing uses 25ms Hamming windows with 10ms hop length - these parameters are standard in speech processing and provide good time-frequency resolution trade-offs.

MFCC computation extracts 13 base coefficients representing the spectral envelope characteristics that are most relevant for speaker identification. These coefficients capture the shape of the spectral envelope while being relatively invariant to noise and channel effects.

Delta features - first derivatives - represent the velocity of spectral changes over time. Delta-delta features - second derivatives - represent acceleration. These temporal derivatives capture dynamic characteristics of speech that are crucial for speaker discrimination.

Mean-variance normalization per utterance ensures consistent feature scaling across different recording conditions and speakers.

For 3-second audio at 16kHz, we typically obtain 300 time frames, resulting in a 300×39 feature matrix that captures both spectral and temporal characteristics of the speaker's voice.

---

## **SLIDE 15: MFCC Feature Extraction Code** *(1 minute)*

This implementation uses librosa for robust and efficient MFCC extraction. The function takes raw audio and sampling rate as inputs, with configurable MFCC coefficient count.

We extract base MFCCs using librosa's optimized implementation, then compute delta and delta-delta features using the delta function with appropriate order parameters. The key insight is proper feature stacking - we vertically stack all three feature types to create our 39-dimensional feature vector.

Transposition to time-first format and per-utterance normalization ensure compatibility with our CNN architecture and consistent feature scaling. This preprocessing is critical for model performance and must be identical between training and inference.

---

## **SLIDE 16: CNN Architecture Design** *(1.5 minutes)*

Our CNN architecture follows a systematic design philosophy optimized for speaker recognition tasks. The input layer accepts batches of time-frequency feature matrices with shape (batch_size, time_frames, 39).

We use three convolutional blocks with progressively increasing filter counts: 64, 128, and 256 filters respectively. This hierarchical approach allows the network to learn increasingly complex and abstract representations of speaker characteristics.

Each block includes 2D convolution with appropriate kernel sizes, batch normalization for training stability and faster convergence, ReLU activation for non-linearity, and max pooling for dimensionality reduction and translation invariance.

Global average pooling eliminates spatial dimensions while preserving channel information, making the network robust to variable input lengths. This is crucial for real-world deployment where audio segments may vary slightly in duration.

The dense layers progressively reduce dimensionality from 512 to 256 to our final 128-dimensional embedding. Dropout between dense layers with probability 0.3 prevents overfitting while maintaining model capacity.

---

## **SLIDE 17: CNN Speaker Model Implementation** *(1 minute)*

This PyTorch implementation shows our CNNSpeakerModel class structure. The constructor defines all layers systematically: three convolutional blocks, each with convolution, batch normalization, and pooling layers.

Global average pooling is implemented using AdaptiveAvgPool2d, which automatically handles variable input sizes by computing the mean across spatial dimensions.

The dense layers create our embedding pathway with appropriate dropout for regularization. This modular design makes the architecture easy to understand, modify, and debug during development.

---

## **SLIDE 18: CNN Forward Pass Implementation** *(1 minute)*

The forward pass demonstrates careful tensor manipulation throughout the network. We add a channel dimension for 2D convolution compatibility, then systematically process through all convolutional blocks.

Each block applies convolution, batch normalization, ReLU activation, and max pooling in sequence. Global average pooling reduces spatial dimensions, and we flatten for the dense layers.

The final L2 normalization is crucial - it ensures all embeddings lie on the unit sphere, making cosine similarity both computationally efficient and theoretically sound. This normalization step is essential for reliable speaker verification performance.

---

## **SLIDE 19: Training Strategy for Speaker CNN** *(1.5 minutes)*

Our training strategy uses triplet loss with margin 0.2, which directly optimizes the similarity metric we use for verification. Triplet loss learns to minimize distance between same-speaker embeddings while maximizing distance between different speakers.

Each triplet consists of an anchor sample, a positive sample from the same speaker, and a negative sample from a different speaker. This approach directly optimizes the embedding space for speaker discrimination.

Data augmentation is crucial for robustness. We apply time shifting within ±0.1 seconds, noise addition with SNR between 15-25dB, and speed perturbation between 0.9x and 1.1x. These augmentations simulate real-world variations in recording conditions.

Balanced batch sampling ensures each training batch contains multiple speakers with multiple utterances, enabling effective triplet mining during training. We use 4 speakers with 4 utterances each per batch.

The Adam optimizer with learning rate 0.001 and weight decay 1e-4 provides stable training with appropriate regularization.

---

## **SLIDE 20: Triplet Loss Implementation** *(1 minute)*

This implementation computes triplet loss efficiently using PyTorch's pairwise distance function. We calculate distances between anchor-positive and anchor-negative pairs, then apply the margin-based loss function.

The ReLU ensures we only penalize triplets where the positive distance exceeds the negative distance by less than the margin. This encourages the model to create clear separation between different speakers while keeping same-speaker embeddings close.

The training loop shows how we generate embeddings for all three sample types and compute the loss. This direct optimization of the verification metric leads to better real-world performance compared to classification-based approaches.

---

## **SLIDE 21: Speaker Verification Pipeline** *(1.5 minutes)*

Our verification system operates in two distinct phases designed for practical deployment scenarios.

During enrollment, users provide 3-5 utterances totaling 15-30 seconds of speech. We extract MFCC features from each utterance, generate embeddings using our trained CNN, and store the averaged embedding as the user's template. Averaging multiple embeddings provides robustness against individual recording variations.

During verification, we process incoming audio chunks in 3-second windows, extract features using identical preprocessing, generate embeddings with the same CNN model, and compute cosine similarity with the stored template.

Threshold selection typically ranges from 0.6 to 0.8 depending on security requirements - higher thresholds provide better security but may increase false rejections of legitimate users.

Real-time optimization uses a sliding window approach for continuous verification throughout longer calls, providing ongoing confidence assessment rather than single-point verification.

---

## **SLIDE 22: Speaker Verification Service** *(1 minute)*

This service class encapsulates the complete verification workflow in a production-ready interface. The constructor loads the trained model and sets verification parameters.

The enroll_speaker method handles the enrollment process, processing multiple audio samples and averaging their embeddings for robust template creation. Error handling ensures graceful degradation if individual samples fail processing.

The verify_speaker method performs real-time verification, handling feature extraction, embedding generation, and similarity computation in a single call. It returns both a boolean decision and the similarity score for confidence assessment.

This design enables easy integration into larger systems while maintaining clean separation of concerns.

---

## **SLIDE 23: Performance Metrics & Evaluation** *(1.5 minutes)*

Our evaluation used a custom dataset with 50 speakers and 20 utterances each, providing sufficient diversity for robust performance assessment.

Key metrics demonstrate competitive performance: Equal Error Rate of 2.3% is comparable to commercial speaker verification systems. False Acceptance Rate of 0.8% at 1% False Rejection Rate shows good security-usability balance.

Processing performance meets real-time requirements: embedding extraction takes just 45ms on CPU and 12ms on GPU, enabling deployment in various computational environments.

Robustness testing validates practical deployment viability: 94.2% accuracy across different recording sessions, 89.7% accuracy in noisy conditions at 15dB SNR, and 91.5% accuracy across different microphones demonstrate the system's resilience to real-world variations.

Memory footprint is minimal at 128 bytes per enrolled user, enabling scalable deployment even with large user bases.

---

## **SLIDE 24: Integration with Phishing Detection** *(1 minute)*

The integration strategy maximizes both security and efficiency through intelligent system coordination.

Parallel processing runs speaker verification concurrently with speech-to-text conversion, minimizing overall latency. Early bypass allows trusted speakers to skip computationally expensive phishing analysis, improving user experience for legitimate callers.

Risk scoring uses speaker confidence to modulate phishing detection thresholds - high speaker confidence can reduce sensitivity, while unknown callers receive full analysis with standard thresholds.

Fallback strategies ensure security is never compromised - unknown speakers always receive complete pipeline analysis, and system failures default to maximum security posture.

Privacy controls require explicit user consent for biometric enrollment, and users can opt out while maintaining phishing protection. Performance impact shows 15% reduction in overall system latency for known callers, demonstrating clear efficiency benefits.

---

## **SLIDE 25: Backend Architecture** *(1 minute)*

Our backend follows a modular service architecture enabling independent scaling and maintenance. The main server handles API orchestration and request routing, while specialized services manage specific functionality.

STT service uses Whisper for robust speech recognition across various audio conditions. ML model service handles phishing detection inference with proper model loading and caching. Speaker biometric service manages enrollment, verification, and template storage.

Audio preprocessing service handles format conversion, noise reduction, and feature extraction. This separation enables optimization of each component independently and facilitates testing and debugging.

---

## **SLIDE 26: Runtime Decision Logic** *(1 minute)*

The runtime system processes audio chunks through carefully orchestrated parallel pipelines. STT and phishing detection run concurrently with speaker verification to minimize latency.

The decision engine implements configurable logic combining rule-based heuristics, ML predictions, and biometric confidence scores. Thresholds can be adjusted based on organizational risk tolerance and user preferences.

Privacy preservation processes audio ephemerally while persisting only mathematical embeddings with explicit user consent. This approach balances security needs with privacy requirements.

---

## **SLIDE 27: Top Priorities** *(1 minute)*

Our immediate development priorities focus on production readiness and system robustness.

Tokenizer alignment verification ensures consistent inference performance by confirming that vocabulary and embedding matrix preprocessing matches training procedures exactly.

Anti-spoofing detection implementation will add production-grade security against voice synthesis and replay attacks, essential for real-world deployment.

Comprehensive unit testing for CNN model I/O and MFCC feature extraction will ensure system reliability and facilitate continuous integration.

CNN inference optimization through quantization and pruning will enable edge deployment while maintaining accuracy, crucial for privacy-sensitive applications.

---

## **SLIDE 28: Future Work** *(1 minute)*

Longer-term research directions focus on advancing the state-of-the-art in voice security systems.

Attention-based speaker models like x-vectors and ECAPA-TDNN represent the current frontier in speaker recognition and could improve our system's accuracy and robustness.

Edge deployment with privacy-preserving STT would enable completely local processing, eliminating privacy concerns about cloud-based audio processing.

Semi-supervised continual learning from production false positives would allow the system to adapt and improve based on real-world deployment experience.

Multi-modal fusion combining voice biometrics with behavioral patterns could provide additional security layers and improved user experience.

---

## **SLIDE 29: References** *(30 seconds)*

Our work builds on established research in both phishing detection and speaker recognition. Key references include the BiLSTM and attention mechanisms from Boussougou and Park, x-vector embeddings from Snyder et al., and ECAPA-TDNN architectures from Desplanques et al.

All project materials and implementation artifacts are available in our repository for reproducibility and further research.

---

## **SLIDE 30: Thank You** *(30 seconds)*

In conclusion, we've developed a comprehensive system that successfully combines advanced NLP techniques for phishing detection with robust CNN-based speaker biometrics. Our approach balances security, privacy, and performance while providing interpretable results that users can understand and trust.

The system demonstrates significant potential for real-time voice security applications and opens several promising avenues for future research in multi-modal biometric systems and privacy-preserving security technologies.

Thank you for your attention. I'm happy to answer any questions about the implementation, results, or future directions of this research.

---

## **Q&A Preparation**

**Expected Questions:**

1. **"How do you handle false positives in phishing detection?"**
   - Attention mechanism provides interpretability
   - Configurable thresholds based on risk tolerance
   - Speaker biometrics provide additional confidence signals

2. **"What about privacy concerns with biometric storage?"**
   - Only mathematical embeddings stored, not raw audio
   - 128 bytes per user - minimal footprint
   - Explicit user consent required
   - Ephemeral audio processing by design

3. **"How does the system scale with more enrolled users?"**
   - O(1) embedding comparison per user
   - Constant CNN inference time
   - Parallel processing architecture enables horizontal scaling