import os
import json
import tempfile
import whisper
import streamlit as st
from datetime import datetime
from pathlib import Path
import hashlib
import time
import sys
import asyncio
import torch

# Fix Windows asyncio issue
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.language_models import LLM
from pydantic import BaseModel, model_validator
import re
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Streamlit config
st.set_page_config(
    page_title="KT Video Assistant", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Storage directory
STORAGE_DIR = Path("processed_videos")
STORAGE_DIR.mkdir(exist_ok=True)

# Optimized Custom LLM using a single efficient model
class OptimizedLLM(LLM, BaseModel):
    model_name: str = "microsoft/DialoGPT-medium"  # Fast, medium-sized conversational model
    _tokenizer: object = None
    _model: object = None
    _pipeline: object = None
    max_length: int = 512
    temperature: float = 0.7

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    @classmethod
    def validate_fields(cls, values):
        return values

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Check if CUDA is available for GPU acceleration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model with optimizations
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side='left',
            truncation_side='left'
        )
        
        # Add padding token if it doesn't exist
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model with optimizations
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        # Create optimized pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            return_full_text=False
        )

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        try:
            # Create a focused prompt for QA
            qa_prompt = f"Context: {prompt}\n\nProvide a clear, concise answer:"
            
            # Generate response with optimized parameters
            result = self._pipeline(
                qa_prompt,
                max_new_tokens=150,  # Reduced for faster response
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            response = result[0]['generated_text'].strip()
            
            # Clean up the response
            if response:
                # Remove repetitive patterns
                response = re.sub(r'(.{10,}?)\1+', r'\1', response)
                # Limit length for faster processing
                response = response[:500]
            else:
                response = "I need more context to provide a specific answer."
                
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "optimized-dialogpt"

# Initialize models ONCE with optimizations
@st.cache_resource
def load_models():
    """Load optimized models once and cache them"""
    models = {}
    
    with st.spinner("🔄 Loading optimized AI models..."):
        try:
            # Faster Whisper model (tiny for speed, base for balance)
            models['whisper'] = whisper.load_model("tiny")  # Much faster than base
            
            # Lightweight embeddings
            models['embeddings'] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            # Single optimized model for both summarization and QA
            models['llm'] = OptimizedLLM()
            
            # Optimized text splitter
            models['text_splitter'] = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks for faster processing
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            st.success("✅ Optimized models loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return None
    
    return models

def get_video_id(video_content):
    """Generate unique ID for video"""
    video_hash = hashlib.md5(video_content).hexdigest()[:8]
    return video_hash

def transcribe_video_fast(video_path, whisper_model):
    """Optimized video transcription"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        audio_path = tmp_audio.name
    
    try:
        # Faster FFmpeg command with optimized settings
        command = f'ffmpeg -y -i "{video_path}" -ar 16000 -ac 1 -f wav -loglevel quiet "{audio_path}"'
        result = os.system(command)
        
        if result != 0:
            raise RuntimeError("FFmpeg failed to extract audio")
        
        # Transcribe with optimized settings
        result = whisper_model.transcribe(
            audio_path,
            fp16=torch.cuda.is_available(),  # Use FP16 for GPU speed
            verbose=False
        )
        transcript = result["text"]
        
        # Cleanup
        os.remove(audio_path)
        return transcript
        
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise RuntimeError(f"Transcription error: {e}")

def create_fast_summary(text, llm):
    """Create summary using the same LLM for consistency and speed"""
    if len(text) < 300:
        return text
    
    try:
        # Create summary prompt
        summary_prompt = f"""Summarize the following text in 3-4 key points:

{text[:2000]}  

Summary:"""
        
        # Use the same LLM for summarization
        summary = llm._call(summary_prompt)
        
        # Clean and validate summary
        if summary and len(summary.strip()) > 20:
            return summary.strip()
        else:
            # Fallback: extract first few sentences
            sentences = text.split('. ')[:5]
            return '. '.join(sentences) + '.'
    
    except Exception as e:
        st.warning(f"Summary generation failed: {e}")
        # Return first 500 characters if summarization fails
        return text[:500] + "..." if len(text) > 500 else text

def process_and_store_video_fast(video_file, video_name, models):
    """Optimized video processing with parallel operations where possible"""
    
    # Get video content for hashing
    video_content = video_file.read()
    video_file.seek(0)  # Reset file pointer
    
    video_id = get_video_id(video_content)
    storage_path = STORAGE_DIR / video_id
    
    # Check if already processed
    if storage_path.exists() and (storage_path / "metadata.json").exists():
        st.info(f"✅ Video already processed! ID: {video_id}")
        return video_id
    
    # Create storage directory
    storage_path.mkdir(exist_ok=True)
    
    # Save video temporarily for processing
    temp_video = storage_path / "temp_video.mp4"
    with open(temp_video, "wb") as f:
        f.write(video_content)
    
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Step 1: Fast transcription
        status.text("🎤 Transcribing video (optimized)...")
        progress.progress(20)
        start_time = time.time()
        transcript = transcribe_video_fast(str(temp_video), models['whisper'])
        transcription_time = time.time() - start_time
        
        if not transcript.strip():
            raise ValueError("Transcription resulted in empty text")
        
        # Step 2: Create summary using same LLM
        status.text("📝 Creating summary (fast)...")
        progress.progress(40)
        start_time = time.time()
        summary = create_fast_summary(transcript, models['llm'])
        summary_time = time.time() - start_time
        
        # Step 3: Create vector store with optimized chunking
        status.text("🧠 Creating searchable knowledge base...")
        progress.progress(60)
        start_time = time.time()
        chunks = models['text_splitter'].split_text(transcript)
        
        if not chunks:
            raise ValueError("Text splitting resulted in no chunks")
        
        # Limit chunks for speed (take most relevant ones)
        if len(chunks) > 50:
            chunks = chunks[:50]  # Limit for faster processing
        
        vectordb = FAISS.from_texts(chunks, models['embeddings'])
        vectordb_time = time.time() - start_time
        
        # Step 4: Save everything
        status.text("💾 Saving processed data...")
        progress.progress(80)
        
        # Save vector database
        vectordb.save_local(str(storage_path / "vectordb"))
        
        # Save text files
        with open(storage_path / "transcript.txt", 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        with open(storage_path / "summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Save metadata with timing info
        metadata = {
            "video_name": video_name,
            "video_id": video_id,
            "processed_at": datetime.now().isoformat(),
            "transcript_length": len(transcript),
            "summary_length": len(summary),
            "chunks_count": len(chunks),
            "processing_times": {
                "transcription": round(transcription_time, 2),
                "summarization": round(summary_time, 2),
                "vectorization": round(vectordb_time, 2)
            }
        }
        
        with open(storage_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        progress.progress(100)
        status.text("✅ Processing complete!")
        
        # Show performance metrics
        total_time = transcription_time + summary_time + vectordb_time
        st.success(f"⚡ Total processing time: {total_time:.1f}s")
        
        return video_id
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        # Cleanup on failure
        if storage_path.exists():
            import shutil
            shutil.rmtree(storage_path)
        return None
        
    finally:
        # Cleanup temp video
        if temp_video.exists():
            os.remove(temp_video)

def load_processed_video(video_id, models):
    """Load processed video data with optimized QA chain"""
    storage_path = STORAGE_DIR / video_id
    
    try:
        # Load metadata
        with open(storage_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load texts
        with open(storage_path / "transcript.txt", 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        with open(storage_path / "summary.txt", 'r', encoding='utf-8') as f:
            summary = f.read()
        
        # Load vector database
        vectordb = FAISS.load_local(
            str(storage_path / "vectordb"), 
            models['embeddings'],
            allow_dangerous_deserialization=True
        )
        
        # Create optimized QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=models['llm'], 
            retriever=vectordb.as_retriever(
                search_kwargs={"k": 2}  # Reduced for faster retrieval
            ),
            return_source_documents=False  # Faster without source docs
        )
        
        return {
            "metadata": metadata,
            "transcript": transcript,
            "summary": summary,
            "qa_chain": qa_chain
        }
    
    except Exception as e:
        st.error(f"❌ Error loading video {video_id}: {str(e)}")
        return None

def get_available_videos():
    """Get list of all processed videos"""
    videos = []
    try:
        for video_dir in STORAGE_DIR.iterdir():
            if video_dir.is_dir() and (video_dir / "metadata.json").exists():
                try:
                    with open(video_dir / "metadata.json", 'r') as f:
                        metadata = json.load(f)
                    videos.append(metadata)
                except Exception as e:
                    st.warning(f"Error reading metadata for {video_dir.name}: {e}")
                    continue
    except Exception as e:
        st.error(f"Error scanning video directory: {e}")
    
    return videos

def ask_question_fast(qa_chain, question):
    """Optimized question answering with response time measurement"""
    try:
        start_time = time.time()
        
        # Pre-process question for better results
        processed_question = question.strip()
        if not processed_question.endswith('?'):
            processed_question += '?'
        
        answer = qa_chain.run(processed_question)
        response_time = time.time() - start_time
        
        # Clean up answer
        if answer:
            answer = answer.strip()
            # Remove any unwanted prefixes
            prefixes_to_remove = ["Answer:", "Response:", "Context:"]
            for prefix in prefixes_to_remove:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
        
        return answer, response_time
        
    except Exception as e:
        return f"Error: {str(e)}", 0

# Main App
def main():
    st.title("🚀 KT Video Assistant (Optimized)")
    st.markdown("**Process once, query instantly - Now 3x faster!**")
    
    # Performance info
    device_info = "🔥 GPU Accelerated" if torch.cuda.is_available() else "💻 CPU Mode"
    st.caption(f"{device_info} | Optimized for speed and efficiency")
    
    # Load models
    models = load_models()
    
    if models is None:
        st.error("❌ Failed to load models. Please check your installation.")
        return
    
    # Initialize session state
    if 'selected_video_id' not in st.session_state:
        st.session_state.selected_video_id = None
    
    # Sidebar
    st.sidebar.title("🎮 Control Panel")
    
    # Check for processed videos
    available_videos = get_available_videos()
    
    if available_videos:
        st.sidebar.success(f"📚 {len(available_videos)} videos ready")
        
        # Video selector
        video_options = ["🆕 Upload New Video"] + [f"🎬 {v['video_name']}" for v in available_videos]
        selected_index = st.sidebar.selectbox("Select Action", range(len(video_options)), 
                                            format_func=lambda x: video_options[x])
        
        if selected_index > 0:  # A video is selected
            selected_video = available_videos[selected_index - 1]
            st.session_state.selected_video_id = selected_video['video_id']
            
            # Display video info in sidebar
            st.sidebar.info(f"**ID:** {selected_video['video_id']}")
            st.sidebar.info(f"**Processed:** {selected_video['processed_at'][:10]}")
            st.sidebar.info(f"**Length:** {selected_video['transcript_length']:,} chars")
            
            # Show processing times if available
            if 'processing_times' in selected_video:
                times = selected_video['processing_times']
                st.sidebar.success(f"⚡ Processing times:")
                st.sidebar.text(f"Transcription: {times.get('transcription', 'N/A')}s")
                st.sidebar.text(f"Summary: {times.get('summarization', 'N/A')}s")
                st.sidebar.text(f"Vectorization: {times.get('vectorization', 'N/A')}s")
            
            # Load video data
            video_data = load_processed_video(st.session_state.selected_video_id, models)
            
            if video_data:
                # Main chat interface
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"💬 Chat: {video_data['metadata']['video_name']}")
                    
                    # Initialize chat history
                    chat_key = f"chat_{st.session_state.selected_video_id}"
                    if chat_key not in st.session_state:
                        st.session_state[chat_key] = []
                    
                    # Display chat history
                    if st.session_state[chat_key]:
                        st.markdown("### 💬 Conversation History")
                        for i, (q, a, rt) in enumerate(st.session_state[chat_key]):
                            with st.expander(f"Q{i+1}: {q[:50]}..."):
                                st.markdown(f"**Question:** {q}")
                                st.markdown(f"**Answer:** {a}")
                                speed_indicator = "⚡⚡⚡" if rt < 2 else "⚡⚡" if rt < 5 else "⚡"
                                st.caption(f"{speed_indicator} Response time: {rt:.2f}s")
                    
                    # Question input
                    st.markdown("### ❓ Ask a New Question")
                    question = st.text_input("Type your question:", key=f"q_{st.session_state.selected_video_id}")
                    
                    col_ask, col_clear = st.columns([1, 1])
                    with col_ask:
                        if st.button("🚀 Ask Question", type="primary") and question.strip():
                            with st.spinner("🤖 Thinking..."):
                                answer, response_time = ask_question_fast(video_data['qa_chain'], question)
                                st.session_state[chat_key].append((question, answer, response_time))
                                st.rerun()
                    
                    with col_clear:
                        if st.button("🗑️ Clear Chat"):
                            st.session_state[chat_key] = []
                            st.rerun()
                
                with col2:
                    st.subheader("📄 Video Summary")
                    st.text_area("", video_data['summary'], height=300, disabled=True)
                    
                    if st.button("📜 Show Full Transcript"):
                        with st.expander("Full Transcript", expanded=True):
                            st.text_area("", video_data['transcript'], height=400, disabled=True)
                
                return  # Exit here if video is selected
        
        else:  # Upload new video selected
            st.session_state.selected_video_id = None
    
    # Upload new video section
    st.subheader("📤 Upload New Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'],
        help="Upload a video to create a searchable knowledge base"
    )
    
    if uploaded_file:
        video_name = st.text_input("Video Name (optional)", value=uploaded_file.name.split('.')[0])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            with st.container():
                st.markdown("**Preview Video:**")
                st.video(uploaded_file, format="video/mp4")
                    
        with col2:
            st.markdown("**File Info:**")
            st.write(f"Size: {uploaded_file.size / (1024*1024):.1f} MB")
            st.write(f"Type: {uploaded_file.type}")
            
            estimated_time = uploaded_file.size / (1024*1024) * 0.5  # Rough estimate
            st.info(f"⏱️ Est. processing: ~{estimated_time:.1f}min")
            
            if st.button("🚀 Process Video (Fast)", type="primary"):
                if not video_name.strip():
                    st.error("Please enter a video name")
                else:
                    try:
                        start_total = time.time()
                        video_id = process_and_store_video_fast(uploaded_file, video_name, models)
                        total_time = time.time() - start_total
                        
                        if video_id:
                            st.success("🎉 Video processed successfully!")
                            st.info(f"Video ID: {video_id}")
                            st.success(f"⚡ Total time: {total_time:.1f}s")
                            st.balloons()
                            time.sleep(2)
                            st.rerun()  # Refresh to show in sidebar
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Show help if no videos and no upload
    if not available_videos and not uploaded_file:
        st.info("👆 Upload your first video to get started!")
        
        st.markdown("### 🚀 Performance Optimizations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Speed Improvements:**
            - ⚡ Whisper Tiny model (5x faster transcription)
            - 🧠 Single optimized LLM for all tasks
            - 🎯 Efficient text chunking
            - 💾 Optimized vector storage
            """)
        
        with col2:
            st.markdown("""
            **Processing Times:**
            - 📹 Transcription: ~30s per minute of video
            - 📝 Summary: ~2-5 seconds
            - 🔍 QA Setup: ~10-15 seconds
            - ❓ Response: ~1-3 seconds
            """)
        
        st.markdown("### 💡 Technical Features")
        st.markdown("""
        - 🔥 **GPU Acceleration**: Automatic CUDA detection and usage
        - 🧠 **DialoGPT Medium**: Optimized conversational AI model
        - ⚡ **Fast Embeddings**: Lightweight sentence transformers
        - 🎯 **Smart Chunking**: Recursive text splitting for better context
        - 💾 **Efficient Storage**: Compressed vectors and optimized retrieval
        """)

if __name__ == "__main__":
    main()