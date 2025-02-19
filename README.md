# **Image Plagiarism Detection Using Computer Vision**  

## **Project Description**  
This project is designed to help detect and analyze image plagiarism using advanced computer vision techniques. It's useful for:  
- **Content Creators & Artists**: Protect intellectual property.  
- **Digital Publishers**: Ensure content authenticity and avoid copyright issues.  
- **Educational Institutions**: Maintain academic integrity by detecting image plagiarism.  

### **Key Features**  
1. **Image Similarity Analysis**  
   - Structural Similarity Index (SSIM) for pixel-level similarity.  
   - ORB (Oriented FAST and Rotated BRIEF) for feature matching.  

2. **Advanced Visualizations**  
   - Highlight differences.  
   - Show feature matches between images.  

3. **Interactive Web Interface**  
   - Built using Streamlit for an easy-to-use experience.  

4. **Report Generation**  
   - Generate detailed plagiarism analysis reports.  

---

## **Installation Steps**  

Follow these simple steps to set up and run the project:  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/MohamaduRiyas/Image-Plagiarism-Detection.git
   cd Image-Plagiarism-Detection
   ```

2. **Install Python**  
   - Download Python 3.8+ from [python.org](https://www.python.org/downloads/).  
   - Make sure to check "Add Python to PATH" during installation.  

3. **Install Required Libraries**  
   Inside the project folder, install the dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**  
   Launch the app with this command:  
   ```bash
   streamlit run app.py
   ```

5. **Access the Web App**  
   - The app will automatically open in your browser.  
   - Alternatively, visit `http://localhost:8501` in your browser.  

---

## **Future Enhancements**  
- Machine learning for advanced similarity detection.  
- API integration for real reverse image searches.  
- Enhanced reporting and batch processing features.
