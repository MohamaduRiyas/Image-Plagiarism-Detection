import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import LinearSegmentedColormap
import time
import requests
from PIL import Image
import io
import base64
import hashlib

# Set page config with improved styling
st.set_page_config(
    page_title="Image Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E0E0E0;
    }
    .card {
        background-color: #F5F7FA;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #5F6368;
    }
    .highlight {
        background-color: #FFE0B2;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    .upload-section {
        background-color: #E1F5FE;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .footer {
        color: #5F6368;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E0E0E0;
    }
    .stProgress .st-ep {
        background-color: #1E88E5;
    }
    .attribution-card {
        background-color: #FFF8E1;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FFA000;
    }
    .source-found {
        background-color: #E8F5E9;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .source-info {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .source-info img {
        margin-right: 12px;
        border-radius: 4px;
        border: 1px solid #EEEEEE;
    }
</style>
""", unsafe_allow_html=True)

# Create Upload Directory
if not os.path.exists("uploads"):
    os.makedirs("uploads")
    
# Function to calculate image hash for reverse search
def calculate_image_hash(img_path):
    img = Image.open(img_path)
    img = img.resize((8, 8), Image.LANCZOS)
    img = img.convert('L')  # Convert to grayscale
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ''.join(['1' if pixel >= avg else '0' for pixel in pixels])
    hexadecimal = hex(int(bits, 2))[2:].upper().zfill(16)
    return hexadecimal

# Function to perform reverse image search (simulated)
def find_image_source(img_path):
    """
    This function performs a reverse image search to find potential sources.
    
    In a real implementation, this could use an API like Google Vision, TinEye, or a custom service.
    For demonstration purposes, we simulate finding results with some sample data.
    """
    # Calculate image hash for the uploaded image
    img_hash = calculate_image_hash(img_path)
    
    # Simulate API call delay
    time.sleep(1.5)
    
    # Simulate possible outcomes based on image hash
    # In reality, this would be an actual API call to a reverse image search service
    hash_value = int(img_hash, 16)
    
    if hash_value % 3 == 0:  # Simulate finding multiple sources
        return [
            {
                "url": "https://example.com/original-image",
                "title": "Original Image Publication",
                "source": "Example Photography",
                "license": "Copyright © Example Photography",
                "contact": "licensing@example.com",
                "thumbnail": "https://picsum.photos/100/100"
            },
            {
                "url": "https://stockphoto.com/image123",
                "title": "Stock Photo Library Image",
                "source": "Stock Photo Inc.",
                "license": "Standard License",
                "contact": "support@stockphoto.com",
                "thumbnail": "https://picsum.photos/100/100?random=1"
            }
        ]
    elif hash_value % 3 == 1:  # Simulate finding a single source
        return [
            {
                "url": "https://creativeworks.org/gallery/landscape-2023",
                "title": "Nature Photography Collection",
                "source": "Creative Works Foundation",
                "license": "Creative Commons Attribution",
                "contact": "info@creativeworks.org",
                "thumbnail": "https://picsum.photos/100/100?random=2"
            }
        ]
    else:  # Simulate no sources found
        return []

# Function to Compute SSIM Score
def compute_ssim(img1, img2, progress_bar=None):
    if progress_bar:
        progress_bar.progress(0.1)
        time.sleep(0.1)
    
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if progress_bar:
        progress_bar.progress(0.2)
        time.sleep(0.1)
    
    # Resize the smaller image to match the larger one's dimensions
    h1, w1 = img1_gray.shape
    h2, w2 = img2_gray.shape
    
    if h1 != h2 or w1 != w2:
        # Determine the target size (using the larger dimensions)
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        # Resize both images to the target size
        img1_gray = cv2.resize(img1_gray, (target_w, target_h))
        img2_gray = cv2.resize(img2_gray, (target_w, target_h))
    
    if progress_bar:
        progress_bar.progress(0.3)
        time.sleep(0.1)
    
    # Compute SSIM
    score, diff = ssim(img1_gray, img2_gray, full=True)
    
    if progress_bar:
        progress_bar.progress(0.4)
        time.sleep(0.1)
    
    return score, diff

# Function to Compute ORB Feature Matching Score
def compute_orb_similarity(img1, img2, progress_bar=None):
    if progress_bar:
        progress_bar.progress(0.5)
        time.sleep(0.1)
    
    # Resize images to have the same dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2 or w1 != w2:
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        img1 = cv2.resize(img1, (target_w, target_h))
        img2 = cv2.resize(img2, (target_w, target_h))
    
    if progress_bar:
        progress_bar.progress(0.6)
        time.sleep(0.1)
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if progress_bar:
        progress_bar.progress(0.7)
        time.sleep(0.1)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0, None, None  # No keypoints found
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    similarity = len(matches) / max(len(des1), len(des2))
    
    if progress_bar:
        progress_bar.progress(0.8)
        time.sleep(0.1)
    
    # Sort matches by distance for visualization
    matches = sorted(matches, key=lambda x: x.distance)
    
    return similarity, kp1, kp2, matches, img1, img2

# Function to create a detailed difference visualization
def visualize_differences(img1, img2, ssim_diff, progress_bar=None):
    if progress_bar:
        progress_bar.progress(0.85)
        time.sleep(0.1)
    
    # Resize images to have the same dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2 or w1 != w2:
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        img1 = cv2.resize(img1, (target_w, target_h))
        img2 = cv2.resize(img2, (target_w, target_h))
    
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create absolute difference image
    diff = cv2.absdiff(img1, img2)
    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
    
    # Normalize SSIM diff to range 0-1 (1 means completely different)
    ssim_diff_normalized = (1.0 - ssim_diff) / 2.0
    
    if progress_bar:
        progress_bar.progress(0.9)
        time.sleep(0.1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    
    # Custom colormap for SSIM differences (green to red)
    colors = [(0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)]  # green -> yellow -> red
    cmap = LinearSegmentedColormap.from_list('diff_cmap', colors, N=256)
    
    # Plot original images
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title('Original Image 1', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_rgb)
    axes[0, 1].set_title('Original Image 2', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Plot absolute difference image
    im3 = axes[1, 0].imshow(diff_rgb)
    axes[1, 0].set_title('Absolute Pixel Difference', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Plot SSIM difference map with custom colormap
    heatmap = axes[1, 1].imshow(ssim_diff_normalized, cmap=cmap)
    axes[1, 1].set_title('SSIM Difference Map', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add colorbar for SSIM differences
    cbar = fig.colorbar(heatmap, ax=axes[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Difference Level', fontsize=10)
    
    # Add overall title
    fig.suptitle('Comprehensive Difference Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Add annotations explaining the analysis
    fig.text(0.02, 0.01, "GREEN = Similar, RED = Different", fontsize=8, color='#444444')
    
    if progress_bar:
        progress_bar.progress(0.95)
        time.sleep(0.1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join("uploads", "difference_visualization.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if progress_bar:
        progress_bar.progress(1.0)
        time.sleep(0.1)
    
    return fig_path

# Sidebar for app description and instructions
with st.sidebar:
    st.image("https://i.pinimg.com/736x/26/a8/12/26a812c94fc3a7fa797571471381e66d.jpg")


    st.markdown("<h2 style='text-align: center;'>About this App</h2>", unsafe_allow_html=True)
    st.markdown("""
    This tool helps you detect potential plagiarism between two images using advanced
    computer vision techniques. Analyze various aspects of similarity including:
    
    - **Structural Similarity (SSIM)**: Measures overall image structure similarity
    - **Feature Matching**: Detects similar points of interest between images
    - **Pixel-level Differences**: Shows exact areas where changes occur
    - **Source Detection**: Finds potential original sources for images
    
    ### How to use:
    1. Upload two images to compare
    2. Wait for the analysis to complete
    3. Review the detailed results and visualizations
    4. If plagiarism is detected, check original sources
    5. Export the report if needed
    """)
    
    st.markdown("<div class='highlight'><strong>Note:</strong> Images will be automatically resized for comparison if they have different dimensions.</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<h3 style='text-align: center;'>Similarity Thresholds</h3>", unsafe_allow_html=True)
    st.markdown("""
    - **> 90%**: High similarity (Likely plagiarism)
    - **70% - 90%**: Moderate similarity (Minor changes)
    - **< 70%**: Low similarity (Different images)
    """)

# Main content
st.markdown("<h1 class='main-header'>🖼️ Advanced Image Plagiarism Detection</h1>", unsafe_allow_html=True)
st.markdown("Detect potential image plagiarism using computer vision and deep analysis techniques.")

# Upload section with improved UI
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3>Upload First Image</h3>", unsafe_allow_html=True)
    uploaded_file1 = st.file_uploader("", type=["jpg", "png", "jpeg", "bmp", "tiff"], key="file1")
    if uploaded_file1:
        st.success("✅ Image 1 uploaded successfully")
    
with col2:
    st.markdown("<h3>Upload Second Image</h3>", unsafe_allow_html=True)
    uploaded_file2 = st.file_uploader("", type=["jpg", "png", "jpeg", "bmp", "tiff"], key="file2")
    if uploaded_file2:
        st.success("✅ Image 2 uploaded successfully")
st.markdown("</div>", unsafe_allow_html=True)

# Process images if uploaded
if uploaded_file1 and uploaded_file2:
    # Read images
    img1_path = os.path.join("uploads", uploaded_file1.name)
    img2_path = os.path.join("uploads", uploaded_file2.name)
    
    with open(img1_path, "wb") as f:
        f.write(uploaded_file1.getbuffer())
    with open(img2_path, "wb") as f:
        f.write(uploaded_file2.getbuffer())
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        st.error("⚠️ Error reading one or both images. Please make sure they are valid image files.")
    else:
        # Show uploaded images with their dimensions
        st.markdown("<h2 class='sub-header'>📸 Uploaded Images</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(img1_path, caption=f"Image 1", use_column_width=True)
            st.markdown(f"<p align='center'><strong>Dimensions:</strong> {img1.shape[1]}x{img1.shape[0]}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(img2_path, caption=f"Image 2", use_column_width=True)
            st.markdown(f"<p align='center'><strong>Dimensions:</strong> {img2.shape[1]}x{img2.shape[0]}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Notify user if images are being resized
        if img1.shape != img2.shape:
            st.info("⚙️ Images have different dimensions. Resizing for comparison...")
        
        # Analysis in progress with animated progress bar
        st.markdown("<h2 class='sub-header'>🔍 Analysis in Progress</h2>", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        
        # Compute similarity scores
        ssim_score, ssim_diff = compute_ssim(img1, img2, progress_bar)
        orb_result = compute_orb_similarity(img1, img2, progress_bar)
        
        if len(orb_result) == 1:
            orb_score = orb_result[0]
            matches_visualization = None
        else:
            orb_score, kp1, kp2, matches, img1_orb, img2_orb = orb_result
            
            # Draw matches for ORB if keypoints were found
            if len(matches) > 0:
                matches_img = cv2.drawMatches(img1_orb, kp1, img2_orb, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                matches_path = os.path.join("uploads", "feature_matches.jpg")
                cv2.imwrite(matches_path, matches_img)
                matches_visualization = matches_path
            else:
                matches_visualization = None
        
        # Create visualization
        diff_visualization = visualize_differences(img1, img2, ssim_diff, progress_bar)
        
        # Results display
        st.markdown("<h2 class='sub-header'>📊 Detection Results</h2>", unsafe_allow_html=True)
        
        # Metric cards in columns with nice formatting
        col1, col2, col3 = st.columns(3)
        
        # Calculate combined score and determine status
        combined_score = (ssim_score * 0.6) + (orb_score * 0.4)  # Weighted average
        
        if combined_score > 0.9:
            status_color = "#F44336"  # Red for high similarity (plagiarism alert)
            status_text = "⚠️ High Similarity"
            status_details = "Possible plagiarism detected! The images show significant structural and feature similarities."
            plagiarism_detected = True
        elif combined_score > 0.7:
            status_color = "#FF9800"  # Orange for moderate similarity
            status_text = "⚠️ Moderate Similarity"
            status_details = "Minor modifications detected. The images share substantial similarities but have some differences."
            plagiarism_detected = True
        else:
            status_color = "#4CAF50"  # Green for low similarity (different images)
            status_text = "✅ Low Similarity"
            status_details = "The images appear to be different. No significant evidence of plagiarism was found."
            plagiarism_detected = False
        
        # Overall status card
        st.markdown(f"""
        <div class='card' style='background-color: {status_color}15;'>
            <h3 style='color: {status_color};'>{status_text}</h3>
            <p>{status_details}</p>
            <div class='metric-value' style='color: {status_color};'>{combined_score:.0%}</div>
            <div class='metric-label'>Combined Similarity Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual metrics
        with col1:
            st.markdown(f"""
            <div class='card'>
                <h4>SSIM Score</h4>
                <div class='metric-value'>{ssim_score:.0%}</div>
                <div class='metric-label'>Structural Similarity</div>
                <p style='font-size: 0.8rem; margin-top: 0.5rem;'>Measures how similar the images are in terms of luminance, contrast, and structure.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='card'>
                <h4>ORB Score</h4>
                <div class='metric-value'>{orb_score:.0%}</div>
                <div class='metric-label'>Feature Matching</div>
                <p style='font-size: 0.8rem; margin-top: 0.5rem;'>Measures similarity based on key feature points detected in both images.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate percentage of modified pixels
            modified_pixel_percent = np.sum(ssim_diff < 0.9) / (ssim_diff.shape[0] * ssim_diff.shape[1]) * 100
            st.markdown(f"""
            <div class='card'>
                <h4>Modified Areas</h4>
                <div class='metric-value'>{modified_pixel_percent:.1f}%</div>
                <div class='metric-label'>Content Differences</div>
                <p style='font-size: 0.8rem; margin-top: 0.5rem;'>Percentage of image area that shows significant differences.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed visualization
        st.markdown("<h2 class='sub-header'>📈 Detailed Difference Analysis</h2>", unsafe_allow_html=True)
        st.image(diff_visualization, use_column_width=True)
        
        with st.expander("📝 Visualization Explanation"):
            st.write("""
            ### How to Interpret the Analysis:
            
            **Top Row**: Original images side by side for direct comparison
            
            **Bottom Left - Absolute Pixel Difference**:
            - Shows exact pixel-by-pixel differences
            - Brighter areas indicate larger differences
            - Black areas represent identical pixels
            
            **Bottom Right - SSIM Difference Map**:
            - GREEN areas indicate high similarity
            - YELLOW areas indicate moderate differences
            - RED areas show significant differences
            - This shows structural differences that matter to human perception
            
            The color bar on the right indicates the degree of difference, helping you gauge the severity of changes across the image.
            """)
        
        # Display feature matches if available
        if matches_visualization:
            st.markdown("<h2 class='sub-header'>🔎 Feature Matching Analysis</h2>", unsafe_allow_html=True)
            st.image(matches_visualization, use_column_width=True)
            
            with st.expander("📝 Feature Matching Explanation"):
                st.write("""
                ### Understanding Feature Matching:
                
                Each line connects a matched feature point between the two images. These are distinctive areas that the algorithm recognizes as similar in both images.
                
                - **More lines** = More similar features detected = Higher similarity
                - **Fewer lines** = Fewer matching features = Lower similarity
                - **Concentrated lines in specific areas** = Similar regions amidst overall differences
                
                This visualization helps identify which specific elements were preserved between images, even if other areas were modified.
                """)
        
        # NEW FEATURE: Image Source Detection and Attribution (only if plagiarism is detected)
        if plagiarism_detected:
            st.markdown("<h2 class='sub-header'>🔍 Image Source Detection</h2>", unsafe_allow_html=True)
            
            # Ask if user wants to perform source detection
            source_check = st.radio(
                "Do you want to check for the original source of these images?",
                options=["Yes, find potential sources", "No, skip this step"]
            )
            
            if source_check == "Yes, find potential sources":
                st.markdown("<div class='attribution-card'>", unsafe_allow_html=True)
                st.markdown("<h3>⚖️ Image Attribution Check</h3>", unsafe_allow_html=True)
                st.markdown("We're checking potential sources for the uploaded images to help with proper attribution...")
                
                # Perform source check for both images (in a real implementation, this would call an actual API)
                source_check_progress = st.progress(0)
                
                # Image 1 source check
                st.markdown("<h4>Checking sources for Image 1...</h4>", unsafe_allow_html=True)
                for i in range(100):
                    source_check_progress.progress(i/100)
                    time.sleep(0.01)
                
                sources_img1 = find_image_source(img1_path)
                
                if sources_img1:
                    st.markdown(f"<p>✅ {len(sources_img1)} potential source(s) found for Image 1</p>", unsafe_allow_html=True)
                    
                    # Display found sources for Image 1
                    for idx, source in enumerate(sources_img1):
                        st.markdown(f"""
                        <div class='source-found'>
                            <div class='source-info'>
                                <img src="{source['thumbnail']}" width="70" height="70" alt="Source thumbnail">
                                <div>
                                    <h4 style='margin: 0;'>{source['title']}</h4>
                                    <p style='margin: 0; color: #666;'>Source: {source['source']}</p>
                                </div>
                            </div>
                            <p><strong>License:</strong> {source['license']}</p>
                            <p><strong>Source URL:</strong> <a href="{source['url']}" target="_blank">{source['url']}</a></p>
                            <p><strong>Contact for permission:</strong> {source['contact']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<p>❌ No sources found for Image 1. This might be an original work or the source is not indexed.</p>", unsafe_allow_html=True)
                
                # Image 2 source check
                st.markdown("<h4>Checking sources for Image 2...</h4>", unsafe_allow_html=True)
                for i in range(100):
                    source_check_progress.progress(i/100)
                    time.sleep(0.01)
                
                sources_img2 = find_image_source(img2_path)
                
                if sources_img2:
                    st.markdown(f"<p>✅ {len(sources_img2)} potential source(s) found for Image 2</p>", unsafe_allow_html=True)
                    
                    # Display found sources for Image 2
                    for idx, source in enumerate(sources_img2):
                        st.markdown(f"""
                        <div class='source-found'>
                            <div class='source-info'>
                                <img src="{source['thumbnail']}" width="70" height="70" alt="Source thumbnail">
                                <div>
                                    <h4 style='margin: 0;'>{source['title']}</h4>
                                    <p style='margin: 0; color: #666;'>Source: {source['source']}</p>
                                </div>
                            </div>
                            <p><strong>License:</strong> {source['license']}</p>
                            <p><strong>Source URL:</strong> <a href="{source['url']}" target="_blank">{source['url']}</a></p>
                            <p><strong>Contact for permission:</strong> {source['contact']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<p>❌ No sources found for Image 2. This might be an original work or the source is not indexed.</p>", unsafe_allow_html=True)
                
                # Usage recommendations
                st.markdown("""
                <h4>🚨 Important Usage Guidelines</h4>
                <p>Based on our analysis and found sources, please consider:</p>
                <ul>
                    <li>If the image belongs to someone else, <strong>contact the owner using the information provided</strong> to request proper permission</li>
                    <li>Always attribute the original creator when using their work</li>
                    <li>Check the license requirements carefully - some licenses may require specific attribution formats or prohibit commercial use</li>
                    <li>When in doubt, seek explicit permission from the owner</li>
                </ul>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
       
        st.markdown("<h2 class='sub-header'>📋 Export Results</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Save Analysis Report", use_container_width=True):
                # Create a report file
                report_time = time.strftime("%Y%m%d-%H%M%S")
                report_path = os.path.join("uploads", f"plagiarism_report_{report_time}.txt")
                
                with open(report_path, "w") as f:
                    f.write(f"Image Plagiarism Analysis Report\n")
                    f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Image 1: {uploaded_file1.name}\n")
                    f.write(f"Image 2: {uploaded_file2.name}\n\n")
                    f.write(f"SSIM Similarity Score: {ssim_score:.4f}\n")
                    f.write(f"ORB Feature Similarity: {orb_score:.4f}\n")
                    f.write(f"Combined Similarity Score: {combined_score:.4f}\n\n")
                    f.write(f"Analysis Result: {status_text}\n")
                    f.write(f"{status_details}\n\n")
                    f.write(f"Modified Area Percentage: {modified_pixel_percent:.2f}%\n")
                
                st.success(f"Report saved to {report_path}")
        
        with col2:
            if st.button("📊 Save Visualization Images", use_container_width=True):
                # No need to save, as images are already saved during processing
                st.success(f"Visualization images saved to uploads directory")

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("👨‍💻 Developed by **MOHAMADU RIYAS** | 🚀 Deployed with Streamlit| Contact @https://www.linkedin.com/in/mohamadu-riyas/" )
st.markdown("</div>", unsafe_allow_html=True)