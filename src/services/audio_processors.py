import tempfile
import os
import aiofiles
from typing import Dict, Any
import numpy as np
import librosa
from pydub import AudioSegment
import logging
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        self.sample_rate = 16000
        self.max_duration = 300  # 5 minutes
        
    async def process_upload(self, audio_file: UploadFile, request_id: str) -> Dict[str, Any]:
        """Process uploaded audio file and extract features"""
        logger.info(f"[{request_id}] Processing audio: {audio_file.filename}")
        
        # Validate file format
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if file_ext not in self.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported: {self.supported_formats}"
            )
        
        # Create temporary file
        temp_path = os.path.join(self.temp_dir, f"{request_id}{file_ext}")
        
        try:
            # Save uploaded file
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
            
            # Convert to WAV if needed
            if file_ext != '.wav':
                wav_path = await self._convert_to_wav(temp_path)
                os.remove(temp_path)
                temp_path = wav_path
            
            # Load audio with librosa
            audio_array, sample_rate = librosa.load(
                temp_path,
                sr=self.sample_rate,
                duration=self.max_duration
            )
            
            # Extract features
            features = await self._extract_audio_features(audio_array, sample_rate)
            
            # Calculate audio statistics
            duration = len(audio_array) / sample_rate
            amplitude = np.abs(audio_array).mean()
            
            # Cleanup
            os.remove(temp_path)
            
            return {
                "filename": audio_file.filename,
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "amplitude": amplitude,
                "features": features,
                "original_format": file_ext[1:],
                "size_bytes": len(content),
                "audio_array": audio_array  # For STT processing
            }
            
        except Exception as e:
            # Cleanup on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"[{request_id}] Audio processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio processing error: {e}")
    
    async def _convert_to_wav(self, input_path: str) -> str:
        """Convert any audio format to WAV"""
        output_path = os.path.splitext(input_path)[0] + '.wav'
        
        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(1)  # Convert to mono
            audio.export(output_path, format="wav")
            return output_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise
    
    async def _extract_audio_features(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract audio features including MFCCs"""
        features = {}
        
        # Extract MFCC features for voice biometrics
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_array, 
                sr=sample_rate, 
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            features["mfcc"] = mfcc.tolist()
            features["mfcc_mean"] = np.mean(mfcc, axis=1).tolist()
            features["mfcc_std"] = np.std(mfcc, axis=1).tolist()
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            features["mfcc"] = []
        
        # Additional audio features
        try:
            features["spectral_centroid"] = float(librosa.feature.spectral_centroid(
                y=audio_array, sr=sample_rate
            ).mean())
            
            features["zero_crossing_rate"] = float(librosa.feature.zero_crossing_rate(
                audio_array
            ).mean())
            
            features["rms_energy"] = float(librosa.feature.rms(
                y=audio_array
            ).mean())
            
            # Voice activity detection
            features["speech_ratio"] = self._detect_speech_ratio(audio_array, sample_rate)
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            features.update({
                "spectral_centroid": 0.0,
                "zero_crossing_rate": 0.0,
                "rms_energy": 0.0,
                "speech_ratio": 0.0
            })
        
        return features
    
    def _detect_speech_ratio(self, audio_array: np.ndarray, sample_rate: int) -> float:
        """Calculate ratio of speech vs silence"""
        try:
            # Simple energy-based VAD
            frame_length = int(0.025 * sample_rate)  # 25ms
            hop_length = int(0.010 * sample_rate)    # 10ms
            
            energy = librosa.feature.rms(
                y=audio_array,
                frame_length=frame_length,
                hop_length=hop_length
            ).flatten()
            
            # Threshold for speech detection
            threshold = energy.mean() * 0.1
            speech_frames = (energy > threshold).sum()
            total_frames = len(energy)
            
            return speech_frames / total_frames if total_frames > 0 else 0.0
        except:
            return 0.0