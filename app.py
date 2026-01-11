import streamlit as st
import cv2, tempfile, os, numpy as np, pandas as pd
from PIL import Image
from ultralytics import YOLO
from deep_translator import GoogleTranslator


st.set_page_config(page_title="Automated Follicle Detection and Maturity Analysis", layout="wide")

# Enhanced CSS with glowing buttons and better spacing
st.markdown("""
<style>
    /* Main background */
    body {
        background: #000000;
        color: #ffffff;
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Logo glow effects */
    .logo-glow {
        filter: drop-shadow(0 0 20px rgba(0, 217, 255, 0.6)) drop-shadow(0 0 40px rgba(0, 217, 255, 0.4));
        transition: all 0.3s ease;
    }
    
    .logo-glow:hover {
        filter: drop-shadow(0 0 30px rgba(0, 217, 255, 0.8)) drop-shadow(0 0 60px rgba(0, 217, 255, 0.5));
        transform: scale(1.05);
    }
    
    /* Glowing button effect */
    .stButton > button {
        background: linear-gradient(135deg, #00FF88 0%, #00D9FF 100%) !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 30px rgba(0, 255, 136, 0.6) !important;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1A1A1A 0%, #00D9FF 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(0, 217, 255, 0.5) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 25px rgba(0, 217, 255, 0.5) !important;
    }
    
    /* Frame navigation buttons */
    div[data-testid="column"] button {
        width: 100% !important;
    }
    
    /* Suggestion boxes */
    .suggestion-box {
        background: linear-gradient(135deg, #1A1A1A 0%, #2A2A2A 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid rgba(0, 217, 255, 0.3);
        margin: 1rem 0;
    }
    
    .suggestion-box:hover {
        border-color: #00D9FF;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
    }
    
    .suggestion-title {
        color: #00FF88;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Disclaimer box */
    .disclaimer-box {
        background: linear-gradient(135deg, #2A1A1A 0%, #3A2A2A 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid rgba(255, 171, 0, 0.5);
        border-left: 6px solid #FFD700;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Clinical Suggestions Database
SUGGESTIONS = {
    "mature": {
        "doctor": [
            "Consider scheduling oocyte retrieval as follicles have reached clinically mature size.",
            "Correlate AI findings with hormone levels (E2, LH) and clinical ultrasound before final decision."
        ],
        "patient": [
            "Follicles appear mature, you may be approaching egg retrieval—please follow your doctor's instructions closely.",
            "Continue prescribed medications and attend follow-up scans or blood tests as scheduled."
        ]
    },
    "immature": {
        "doctor": [
            "Continue follicular stimulation and reassess growth in the next monitoring cycle.",
            "Review dosage and protocol if follicular growth remains slow or uneven."
        ],
        "patient": [
            "Follicles are still developing, additional time or medication may be needed.",
            "Do not worry—regular monitoring helps optimize timing for better outcomes."
        ]
    }
}

DISCLAIMER_EN = (
    "This analysis is generated using an AI-based software tool and is intended for decision support only. "
    "It does not replace clinical judgment, manual ultrasound evaluation, or medical diagnosis. "
    "Final treatment decisions must always be made by a qualified healthcare professional."
)

# Initialize translator
@st.cache_resource
def get_translator():
    return Translator()

def translate_text(text, lang_code):
    """Translate text to target language"""
    if lang_code == 'en':
        return text
    try:
        translator = get_translator()
        return translator.translate(text, dest=lang_code).text
    except:
        return text  # Return original if translation fails

def get_case_from_csv(df):
    """Determine if follicles are mature or immature based on CSV data"""
    if len(df) == 0:
        return "immature"
    avg_mature = df["mature_count"].mean()
    return "mature" if avg_mature >= 1 else "immature"

def generate_suggestions(case_type, lang_code):
    """Generate translated suggestions based on case type and language"""
    suggestions = {
        "doctor": [translate_text(s, lang_code) for s in SUGGESTIONS[case_type]["doctor"]],
        "patient": [translate_text(s, lang_code) for s in SUGGESTIONS[case_type]["patient"]],
        "disclaimer": translate_text(DISCLAIMER_EN, lang_code)
    }
    return suggestions

# Title with AFD logo (glowing)
st.markdown('<div style="margin-top: 20px; margin-bottom: 30px;">', unsafe_allow_html=True)
afd_logo_path = 'afd_logo.png'
if os.path.exists(afd_logo_path):
    col_logo, col_title = st.columns([0.8, 11.2])
    with col_logo:
        st.markdown('<div class="logo-glow">', unsafe_allow_html=True)
        st.image(afd_logo_path, width=100)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_title:
        st.markdown("""
        <div style="margin-top: 10px">
            <h1 style="color: #00D9FF; margin-bottom: 5px;">Automated Follicle Detection and Maturity Analysis</h1>
            <div style="color: #B5B5B5; font-size: 1.1rem;">Using YOLOv8 Segmentation for IVF Monitoring</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <h1 style="color: #00D9FF;">Automated Follicle Detection and Maturity Analysis</h1>
    <div style="color: #B5B5B5; font-size: 1.1rem;">Using YOLOv8 Segmentation for IVF Monitoring</div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with AI SPRY logo (glowing)
with st.sidebar:
    # Add AI SPRY logo at top of sidebar with glow
    aispry_logo_path = 'aispry_logo.png'
    if os.path.exists(aispry_logo_path):
        st.markdown('<div class="logo-glow" style="text-align: center; padding: 10px;">', unsafe_allow_html=True)
        st.image(aispry_logo_path, width=180)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    st.header('👤 Patient & Run Details')
    patient_name = st.text_input('Patient Name', '')
    patient_age = st.number_input('Patient Age', min_value=18, max_value=80, value=30)
    st.markdown('**Model:** YOLOv8 (Segmentation)')
    
    st.markdown('---')
    st.markdown('### 🔧 Detection Parameters')
    conf_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.50, 0.05, 
                               help="Higher values = fewer but more confident detections")
    iou_threshold = st.slider('IoU Threshold (NMS)', 0.0, 1.0, 0.30, 0.05,
                              help="Lower values = more aggressive overlap removal")
    min_area = st.number_input('Min Follicle Area (pixels)', min_value=50, max_value=5000, value=200,
                               help="Minimum area to consider as a valid follicle")
    
    st.markdown('---')
    st.markdown('### 🌐 Language Selection')
    language_map = {
        'English': 'en',
        'Hindi (हिंदी)': 'hi',
        'Telugu (తెలుగు)': 'te',
        'Tamil (தமிழ்)': 'ta',
        'Malayalam (മലയാളം)': 'ml'
    }
    selected_language = st.selectbox('Select Language', list(language_map.keys()))
    lang_code = language_map[selected_language]

st.markdown('### 📤 Upload Video for Analysis')
st.markdown('<div style="color: #B5B5B5; margin-bottom: 15px;">Upload ultrasound video for follicle detection (1 FPS processing)</div>', unsafe_allow_html=True)
uploaded = st.file_uploader('Upload video', type=['mp4','mov','avi'], label_visibility="collapsed")

# Helper functions from original code
def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def masks_overlap_significantly(mask1, mask2, threshold=0.3):
    """Check if masks overlap by more than threshold"""
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = mask1.sum()
    area2 = mask2.sum()
    if area1 == 0 or area2 == 0:
        return False
    overlap_ratio1 = intersection / area1
    overlap_ratio2 = intersection / area2
    return max(overlap_ratio1, overlap_ratio2) > threshold

def apply_nms_to_masks(masks_info, iou_thresh=0.30):
    """Apply Non-Maximum Suppression with stricter overlap removal"""
    if len(masks_info) <= 1:
        return masks_info
    
    sorted_masks = sorted(masks_info, key=lambda x: x['conf'], reverse=True)
    keep = []
    
    while sorted_masks:
        current = sorted_masks.pop(0)
        keep.append(current)
        
        filtered = []
        for other in sorted_masks:
            if not masks_overlap_significantly(current['mask'], other['mask'], iou_thresh):
                filtered.append(other)
        sorted_masks = filtered
    
    return keep

def merge_close_masks(masks_info, distance_threshold=50):
    """Merge masks that are very close together (likely same follicle)"""
    if len(masks_info) <= 1:
        return masks_info
    
    merged = []
    used = set()
    
    for i, mask_i in enumerate(masks_info):
        if i in used:
            continue
            
        ys_i, xs_i = np.where(mask_i['mask'] == 1)
        if len(xs_i) == 0:
            continue
        cx_i, cy_i = np.mean(xs_i), np.mean(ys_i)
        
        masks_to_merge = [mask_i]
        used.add(i)
        
        for j, mask_j in enumerate(masks_info):
            if j <= i or j in used:
                continue
                
            ys_j, xs_j = np.where(mask_j['mask'] == 1)
            if len(xs_j) == 0:
                continue
            cx_j, cy_j = np.mean(xs_j), np.mean(ys_j)
            
            dist = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            
            if dist < distance_threshold:
                masks_to_merge.append(mask_j)
                used.add(j)
        
        if len(masks_to_merge) > 1:
            combined_mask = np.zeros_like(masks_to_merge[0]['mask'])
            max_conf = max(m['conf'] for m in masks_to_merge)
            for m in masks_to_merge:
                combined_mask = np.logical_or(combined_mask, m['mask']).astype(np.uint8)
            merged.append({'mask': combined_mask, 'conf': max_conf})
        else:
            merged.append(mask_i)
    
    return merged

# Analysis button
if st.button('🔬 Analyze Video', use_container_width=True):
    if uploaded is None:
        st.warning('⚠️ Please upload a video file')
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp.write(uploaded.read())
        tmp.flush()
        video_path = tmp.name
        
        st.info('📊 Loading model...')
        if not os.path.exists('best.pt'):
            st.error('❌ best.pt not found in app folder. Place weights as best.pt')
            st.stop()
        
        model = YOLO('best.pt')
        st.info('🎬 Processing video (detections at ~1 FPS)...')
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps/1.0)))
        
        out_file = os.path.join(os.getcwd(), 'analyzed_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        
        records = []
        last_masks = []
        analyzed_frames = []
        PIXEL_TO_MM = 0.10
        idx = 0
        detection_frame_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            
            if total > 0:
                progress_bar.progress(min(idx / total, 1.0))
                status_text.text(f"Processing frame {idx}/{total}")
            
            run = (idx % step == 1) or idx == 1
            masks_info = last_masks
            
            if run:
                detection_frame_count += 1
                res = model(frame, conf=conf_threshold, iou=0.3, verbose=False)[0]
                masks_info = []
                
                if getattr(res, 'masks', None) is not None:
                    if hasattr(res.masks, 'xy') and res.masks.xy:
                        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
                        for i, poly in enumerate(res.masks.xy):
                            try:
                                pts = np.array(poly, dtype=np.int32)
                                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                                cv2.fillPoly(mask, [pts], 1)
                                area = cv2.countNonZero(mask)
                                
                                if area < min_area:
                                    continue
                                    
                                conf = float(confs[i]) if i < len(confs) else 0.0
                                if conf >= conf_threshold:
                                    masks_info.append({'mask': mask, 'conf': conf, 'area': area})
                            except:
                                pass
                                
                    elif hasattr(res.masks, 'data') and len(res.masks.data):
                        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
                        for i in range(len(res.masks.data)):
                            try:
                                m = res.masks.data[i].cpu().numpy()
                                if m.shape[:2] != (frame.shape[0], frame.shape[1]):
                                    m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
                                mask_bin = (m > 0.5).astype(np.uint8)
                                area = cv2.countNonZero(mask_bin)
                                
                                if area < min_area:
                                    continue
                                    
                                conf = float(confs[i]) if i < len(confs) else 0.0
                                if conf >= conf_threshold:
                                    masks_info.append({'mask': mask_bin, 'conf': conf, 'area': area})
                            except:
                                pass
                
                masks_info = merge_close_masks(masks_info, distance_threshold=50)
                masks_info = apply_nms_to_masks(masks_info, iou_threshold)
                last_masks = masks_info

            overlay = np.zeros_like(frame)
            follicle_sizes = []
            count = 0
            
            for mi in masks_info:
                conf = mi.get('conf', 0.0)
                mask = mi.get('mask')
                count += 1
                
                if conf > 0.7:
                    color = (0, 220, 0)
                else:
                    color = (0, 180, 120)
                    
                overlay[mask == 1] = color
                area = int(cv2.countNonZero(mask))
                
                if area > 0:
                    diam_px = 2.0 * np.sqrt(area / np.pi)
                    diam_mm = diam_px * PIXEL_TO_MM
                    follicle_sizes.append(diam_mm)
                    ys, xs = np.where(mask == 1)
                    if len(xs) > 0:
                        cx, cy = int(np.mean(xs)), int(np.mean(ys))
                        cv2.putText(frame, f"{conf:.2f}", (cx-20, cy-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, f"{diam_mm:.1f}mm", (cx-25, cy+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            blended = cv2.addWeighted(frame, 0.7, overlay, 0.6, 0)
            
            cv2.rectangle(blended, (5, 5), (250, 45), (0, 0, 0), -1)
            cv2.putText(blended, f"Follicles: {count}", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            out.write(blended)
            
            if run:
                analyzed_frames.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

            if run:
                mature = int(sum(1 for s in follicle_sizes if s >= 18.0))
                immature = int(sum(1 for s in follicle_sizes if s < 18.0))
                highest = round(max(follicle_sizes), 2) if follicle_sizes else 0.0
                avg_size = round(np.mean(follicle_sizes), 2) if follicle_sizes else 0.0
                eggs_ready = mature
                records.append({
                    'detection_frame': detection_frame_count, 
                    'video_frame_id': idx, 
                    'num_follicles': count, 
                    'mature_count': mature,
                    'immature_count': immature, 
                    'avg_follicle_mm': avg_size,
                    'highest_follicle_mm': highest,
                    'eggs_ready_for_retrieval': eggs_ready
                })

        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        df = pd.DataFrame(records)
        
        # Store results in session state
        st.session_state['analysis_complete'] = True
        st.session_state['results_df'] = df
        st.session_state['video_file'] = out_file
        st.session_state['analyzed_frames'] = analyzed_frames
        st.session_state['current_frame_idx'] = 0
        st.session_state['lang_code'] = lang_code

# Display results if analysis is complete
if st.session_state.get('analysis_complete', False):
    df = st.session_state['results_df']
    out_file = st.session_state['video_file']
    analyzed_frames = st.session_state.get('analyzed_frames', [])
    lang_code = st.session_state.get('lang_code', 'en')
    
    st.success('✅ Analysis Complete!')
    
    # Summary statistics
    st.markdown('### 📊 Summary Statistics')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Avg Follicles", f"{df['num_follicles'].mean():.1f}")
    with col2:
        st.metric("Max Follicles", f"{df['num_follicles'].max()}")
    with col3:
        st.metric("Avg Mature", f"{df['mature_count'].mean():.1f}")
    with col4:
        st.metric("Avg Size (mm)", f"{df['avg_follicle_mm'].mean():.2f}")
    with col5:
        st.metric("Max Diameter (mm)", f"{df['highest_follicle_mm'].max():.2f}")
    
    # Clinical Interpretation & Suggestions
    st.markdown('---')
    st.markdown('### 🧠 AI Clinical Interpretation')
    
    case_type = get_case_from_csv(df)
    suggestions = generate_suggestions(case_type, lang_code)
    
    # Display case type
    if case_type == "mature":
        st.markdown('<div style="background: linear-gradient(135deg, #00FF88 0%, #00D9A3 100%); padding: 1rem; border-radius: 10px; text-align: center; color: #000000; font-size: 1.2rem; font-weight: bold;">🟢 Mature Follicles Detected (≥ 18 mm)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 1rem; border-radius: 10px; text-align: center; color: #000000; font-size: 1.2rem; font-weight: bold;">🟡 Immature Follicles Only (< 18 mm)</div>', unsafe_allow_html=True)
    
    # Doctor and Patient Suggestions
    col_doc, col_pat = st.columns(2)
    
    with col_doc:
        st.markdown("""
        <div class="suggestion-box">
            <div class="suggestion-title">👨‍⚕️ Doctor Suggestions</div>
        """, unsafe_allow_html=True)
        for i, suggestion in enumerate(suggestions['doctor'], 1):
            st.markdown(f"{i}. {suggestion}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_pat:
        st.markdown("""
        <div class="suggestion-box">
            <div class="suggestion-title">👩 Patient Guidance</div>
        """, unsafe_allow_html=True)
        for i, suggestion in enumerate(suggestions['patient'], 1):
            st.markdown(f"{i}. {suggestion}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer-box">
        <div style="color: #FFD700; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">
            ⚠️ Mandatory Disclaimer
        </div>
        <div style="color: #FFFFFF; line-height: 1.6;">
            {suggestions['disclaimer']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Results
    st.markdown('---')
    st.markdown('### 📋 Detailed Results')
    st.dataframe(df, use_container_width=True)
    st.download_button('⬇️ Download CSV Report', 
                       data=df.to_csv(index=False).encode('utf-8'),
                       file_name='follicle_analysis_results.csv', 
                       mime='text/csv', 
                       key='csv_download',
                       use_container_width=True)
    
    # Frame Navigator (Single Panel)
    if analyzed_frames:
        st.markdown('---')
        st.markdown('### 🖼️ Analyzed Frames Navigator')
        st.markdown(f'<div style="color: #B5B5B5; margin-bottom: 15px;">Navigate through {len(analyzed_frames)} analyzed frames (1 per second)</div>', unsafe_allow_html=True)
        
        # Initialize frame index
        if 'current_frame_idx' not in st.session_state:
            st.session_state['current_frame_idx'] = 0
        
        # Navigation controls
        col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([1, 1, 2, 1, 1])
        
        with col_nav1:
            if st.button('⏮️ First', use_container_width=True):
                st.session_state['current_frame_idx'] = 0
        
        with col_nav2:
            if st.button('◀️ Previous', use_container_width=True):
                if st.session_state['current_frame_idx'] > 0:
                    st.session_state['current_frame_idx'] -= 1
        
        with col_nav3:
            st.markdown(f"<div style='text-align: center; padding: 10px; background: #1A1A1A; border-radius: 8px; color: #00D9FF; font-weight: bold;'>Frame {st.session_state['current_frame_idx'] + 1} of {len(analyzed_frames)}</div>", unsafe_allow_html=True)
        
        with col_nav4:
            if st.button('Next ▶️', use_container_width=True):
                if st.session_state['current_frame_idx'] < len(analyzed_frames) - 1:
                    st.session_state['current_frame_idx'] += 1
        
        with col_nav5:
            if st.button('Last ⏭️', use_container_width=True):
                st.session_state['current_frame_idx'] = len(analyzed_frames) - 1
        
        # Display current frame
        current_idx = st.session_state['current_frame_idx']
        st.image(analyzed_frames[current_idx], 
                caption=f"Detection Frame {current_idx + 1} (Second {current_idx + 1})", 
                use_column_width=True)
        
        # Frame-specific data
        if current_idx < len(df):
            frame_data = df.iloc[current_idx]
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                st.metric("Follicles Detected", int(frame_data['num_follicles']))
            with col_f2:
                st.metric("Mature Count", int(frame_data['mature_count']))
            with col_f3:
                st.metric("Avg Size (mm)", f"{frame_data['avg_follicle_mm']:.2f}")
            with col_f4:
                st.metric("Max Size (mm)", f"{frame_data['highest_follicle_mm']:.2f}")
    
    # Download analyzed video
    st.markdown('---')
    st.markdown('### 📥 Download Analyzed Video')
    if os.path.exists(out_file):
        with open(out_file, 'rb') as vf:
            video_bytes = vf.read()
        
        st.download_button('⬇️ Download Analyzed Video', 
                         data=video_bytes, 
                         file_name='analyzed_follicle_video.mp4', 
                         mime='video/mp4', 
                         key='video_download',
                         use_container_width=True)
    else:
        st.error('❌ Analyzed video file not found. Please run the analysis again.')
    
    st.markdown('---')
    st.info('💡 Tip: Adjust the detection parameters in the sidebar if the count seems incorrect.')
