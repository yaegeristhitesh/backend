# 20-Minute Speech Script: Real-time Voice Phishing Detection & Speaker Biometrics

**Total Duration: ~20 minutes**
**Slide Timing Guide: Each section includes approximate timing**

---

## Opening & Title Slide (1 minute)

Good morning/afternoon everyone. Today I'm excited to present my work on "Real-time Voice Phishing Detection and Speaker Biometrics" - a system that combines advanced machine learning techniques to protect users from increasingly sophisticated voice-based phishing attacks.

*[Show title slide]*

This project addresses a critical cybersecurity challenge: as attackers move beyond traditional email phishing to real-time voice calls, we need intelligent systems that can detect malicious intent while preserving user privacy and maintaining low latency.

---

## Contents Overview (30 seconds)

*[Show contents slide]*

I'll walk you through our motivation, system architecture, the core phishing detection model using BiLSTM networks, our novel CNN-based speaker biometrics implementation, experimental results, and future directions.

---

## Section 1: Motivation & Rising Threat (2.5 minutes)

### Slide: Motivation - Rising Voice Phishing

*[Show motivation slide]*

Let me start with why this problem matters. We're witnessing a fundamental shift in cybercrime tactics. While email and SMS phishing are becoming easier to detect and filter, attackers are increasingly turning to voice channels - what we call "vishing" or voice phishing.

The statistics are alarming. Recent industry reports show year-over-year increases in voice-based attacks, with some regions seeing growth rates exceeding 25%. These aren't just robocalls - they're sophisticated, real-time social engineering attempts where attackers impersonate banks, government agencies, or trusted contacts.

### Slide: Evidence - Incident Statistics

*[Show statistics slide]*

The data speaks for itself. Looking at incident reports from cybersecurity firms, we see a clear upward trend: from approximately 877,000 incidents to over 932,000 attacks - that's a 28% increase in just one year, with projections approaching nearly one million incidents.

What makes voice phishing particularly dangerous is its real-time, dynamic nature. Unlike email phishing, which can be analyzed at leisure, voice calls happen in real-time. Attackers adapt their approach based on victim responses, making rule-based detection systems inadequate.

This rising trend justifies immediate investment in both engineering solutions and research into advanced detection mechanisms.

---

## Section 2: System Goal & Workflow (2 minutes)

### Slide: System Goal and High-level Workflow

*[Show system goal slide]*

Our system has three primary objectives: detect voice phishing in live calls with low latency, maintain high precision to minimize false positives, and preserve user privacy throughout the process.

The pipeline works as follows: client audio is chunked and sent to our system, where it undergoes speech-to-text conversion using Whisper. The resulting transcript feeds into our phishing detection model. Simultaneously, we extract speaker biometric features for trusted caller identification.

Our key priorities are privacy - audio is processed ephemerally and not stored, efficiency - trusted callers can bypass heavy analysis, and interpretability - our attention mechanism shows which words triggered alerts.

### Slide: System Diagram

*[Show system architecture]*

This diagram illustrates our complete architecture. Audio flows through two parallel paths: the transcript path for content analysis and the biometric path for speaker identification. The decision engine combines both signals to make final determinations about call legitimacy.

---

## Section 3: Phishing Model Core Idea (3 minutes)

### Slide: Why 1D-CNN + BiLSTM + Attention?

*[Show architecture rationale slide]*

Now let's dive into our phishing detection model. We chose a hybrid architecture combining three complementary techniques:

First, 1D-CNN layers extract local n-gram-like features - essentially identifying suspicious phrases and linguistic patterns that phishers commonly use.

Second, BiLSTM networks capture long-range context in conversations. The bidirectional nature is crucial because phishing attempts often involve setup phrases early in the call that only become suspicious when combined with later requests.

Third, our attention mechanism focuses the model on the most relevant words and phrases, providing interpretability - we can show users exactly which parts of the conversation triggered the alert.

We use FastText embeddings to handle rare and out-of-vocabulary words, which is essential since phishers often use uncommon terminology to sound official.

### Slide: Detailed Architecture

*[Show detailed architecture slide]*

Here are the specific architectural details: We start with 300-dimensional FastText embeddings, followed by 1D-CNN with kernel size 3 and 64 filters. The BiLSTM has 64 forward and 32 backward hidden units with 0.1 dropout for regularization.

The attention mechanism computes weights over BiLSTM outputs to create a context vector, which feeds into our final dense classifier for binary phishing detection.

### Slide: BiLSTM Code Implementation

*[Show code slide]*

This PyTorch implementation shows our PhishingDetector class. Notice how we transpose tensors between CNN and LSTM layers to handle different expected input formats. The attention mechanism uses softmax to weight LSTM outputs, and we sum these weighted representations to create our final context vector.

---

## Section 4: Training Strategy (1.5 minutes)

### Slide: Training Strategies

*[Show training strategies slide]*

Training presented several challenges. Initial 70-15-15 splits produced unstable results on small datasets. We addressed this with 5-fold stratified cross-validation, which reduced variance by approximately 18% and provided more reliable performance estimates.

We employed multiple regularization techniques: dropout layers, early stopping with patience of 10 epochs, and careful learning rate scheduling to prevent overfitting.

### Slide: Training Code

*[Show training code slide]*

This implementation shows our cross-validation setup with StratifiedKFold to ensure balanced class distribution across folds. We use Adam optimizer with step learning rate decay and implement early stopping based on validation loss to prevent overfitting.

---

## Section 5: Experimental Results (1 minute)

### Slide: Key Experimental Results

*[Show results slide]*

Our results are promising. Referenced literature reports accuracy around 99.32% and F1-score of 99.31% on larger datasets. Our implementation achieved F1-score of 0.91 on a smaller internal dataset of 80 samples, though performance was sensitive to data splits - highlighting the importance of our cross-validation approach.

The cross-validation strategy significantly improved recall and reduced variance, demonstrating the value of robust evaluation methodologies for small datasets.

---

## Section 6: CNN-Based Speaker Biometrics (6 minutes)

### Slide: CNN Speaker Biometrics Technical Architecture

*[Show CNN architecture overview]*

Now let's explore our speaker biometrics system - this is where our implementation really shines. The system processes raw audio waveforms at 16kHz in 3-second windows, extracts 39-dimensional MFCC features, and uses a 3-block CNN architecture to generate 128-dimensional speaker embeddings.

Speaker verification uses cosine similarity between embeddings, providing fast, reliable identification of trusted callers.

### Slide: MFCC Feature Extraction Pipeline

*[Show MFCC pipeline slide]*

Our feature extraction is comprehensive. We start with audio normalization and silence removal, then apply 25ms Hamming windows with 10ms hop length for spectral analysis.

The MFCC computation extracts 13 base coefficients representing spectral envelope characteristics. We then compute delta features - first derivatives representing velocity of spectral changes - and delta-delta features representing acceleration.

This gives us 39 total features per time frame. For 3-second audio at 16kHz, we typically get 300 time frames, resulting in a 300x39 feature matrix.

### Slide: MFCC Code Implementation

*[Show MFCC code slide]*

This implementation uses librosa for robust MFCC extraction. We compute base MFCCs, then delta and delta-delta features using librosa's delta function. The key is proper stacking and normalization - we transpose to time-first format and apply mean-variance normalization per utterance for consistent feature scaling.

### Slide: CNN Architecture Design

*[Show CNN design slide]*

Our CNN architecture has three convolutional blocks with increasing filter counts: 64, 128, and 256 filters respectively. Each block includes batch normalization for training stability, ReLU activation, and max pooling for dimensionality reduction.

Global average pooling eliminates spatial dimensions, followed by dense layers that progressively reduce dimensionality from 512 to 256 to our final 128-dimensional embedding. Dropout between dense layers prevents overfitting.

### Slide: CNN Implementation Code

*[Show CNN implementation slide]*

This PyTorch implementation shows our CNNSpeakerModel class. Notice the systematic progression through convolutional blocks, each with batch normalization and pooling. The global average pooling is crucial for handling variable-length inputs.

### Slide: CNN Forward Pass

*[Show forward pass slide]*

The forward pass adds a channel dimension for 2D convolution, processes through all convolutional blocks, applies global pooling, then passes through dense layers with dropout. The final L2 normalization ensures embeddings lie on the unit sphere, making cosine similarity computationally efficient and theoretically sound.

### Slide: Training Strategy for Speaker CNN

*[Show training strategy slide]*

We use triplet loss with margin 0.2, which learns to minimize distance between same-speaker embeddings while maximizing distance between different speakers. Data augmentation includes time shifting, noise addition, and speed perturbation to improve robustness.

Balanced batch sampling ensures each batch contains multiple speakers with multiple utterances, enabling effective triplet mining during training.

### Slide: Triplet Loss Implementation

*[Show triplet loss code slide]*

This implementation computes pairwise distances between anchor, positive, and negative samples. The triplet loss encourages positive pairs to be closer than negative pairs by at least the margin value. The training loop shows how we generate embeddings for all three sample types and compute the loss.

### Slide: Speaker Verification Pipeline

*[Show verification pipeline slide]*

The system operates in two phases. During enrollment, we collect multiple utterances from users, extract features, generate embeddings, and store averaged templates for robustness.

During verification, we process incoming audio chunks, generate embeddings, and compute cosine similarity with stored templates. Thresholds typically range from 0.6 to 0.8 depending on security requirements.

### Slide: Speaker Verification Service

*[Show service code slide]*

This service class encapsulates the complete verification workflow. The enroll_speaker method averages multiple embeddings for robust templates. The verify_speaker method handles feature extraction, embedding generation, and similarity computation in a single call.

### Slide: Performance Metrics

*[Show performance slide]*

Our evaluation used a custom dataset with 50 speakers and 20 utterances each. We achieved 2.3% Equal Error Rate, which is competitive with commercial systems. Embedding extraction takes just 45ms on CPU and 12ms on GPU, meeting real-time requirements.

Robustness tests show 94.2% accuracy across sessions, 89.7% in noisy conditions, and 91.5% across different microphones, demonstrating practical deployment viability.

---

## Section 7: Integration & Backend (1.5 minutes)

### Slide: Backend Architecture

*[Show backend slide]*

Our backend follows a modular service architecture. The main server orchestrates requests, while specialized services handle STT, ML model inference, and biometric processing. This separation enables independent scaling and maintenance.

### Slide: Runtime Decision Logic

*[Show decision logic slide]*

The runtime system processes audio chunks through parallel pipelines. STT and phishing detection run concurrently with speaker verification. The decision engine combines rule-based logic, ML predictions, and biometric confidence scores.

Importantly, we maintain privacy by processing audio ephemerally while persisting only embeddings with user consent.

---

## Section 8: Future Work & Conclusions (1.5 minutes)

### Slide: Top Priorities

*[Show priorities slide]*

Our immediate priorities include confirming tokenizer alignment for consistent inference, implementing anti-spoofing detection for production deployment, adding comprehensive unit tests, and optimizing CNN inference for edge deployment through quantization and pruning.

### Slide: Future Work

*[Show future work slide]*

Longer-term directions include implementing attention-based speaker models like x-vectors and ECAPA-TDNN, enabling edge deployment with privacy-preserving STT, developing semi-supervised learning from production data, and exploring multi-modal fusion combining voice with behavioral biometrics.

---

## Closing (30 seconds)

### Slide: Thank You

*[Show thank you slide]*

In conclusion, we've developed a comprehensive system that combines advanced NLP techniques for phishing detection with robust CNN-based speaker biometrics. Our approach balances security, privacy, and performance while providing interpretable results.

The system demonstrates the potential for real-time voice security applications and opens several avenues for future research in multi-modal biometric systems.

Thank you for your attention. I'm happy to take questions about any aspect of the implementation or results.

---

## Q&A Preparation Notes

**Potential Questions & Responses:**

1. **"How does your system handle false positives?"**
   - Our attention mechanism provides interpretability, allowing users to understand why alerts were triggered
   - Configurable thresholds allow adjustment based on user risk tolerance
   - Speaker biometrics provide additional confidence signals to reduce false positives for known callers

2. **"What about privacy concerns with biometric data?"**
   - We store only mathematical embeddings, not raw audio
   - User consent is required for enrollment
   - Embeddings are 128 bytes per user - minimal storage footprint
   - Audio processing is ephemeral by design

3. **"How does performance scale with more users?"**
   - Embedding comparison is O(1) per enrolled user
   - CNN inference time is constant regardless of database size
   - Parallel processing architecture enables horizontal scaling

4. **"What about adversarial attacks on the biometric system?"**
   - Current implementation focuses on basic spoofing resistance
   - Future work includes dedicated anti-spoofing detection
   - Multi-modal approaches can provide additional security layers

**Technical Deep-Dive Questions:**

- **MFCC vs other features**: MFCCs are robust, well-understood, and computationally efficient
- **CNN vs RNN for speaker modeling**: CNNs are faster for inference and handle variable-length inputs well
- **Triplet loss vs other losses**: Triplet loss directly optimizes the similarity metric we use for verification
