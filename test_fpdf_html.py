import os
import markdown
from fpdf import FPDF

FONT_URL_HINDI = "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"
FONT_PATH_HINDI = "NotoSansDevanagari-Regular.ttf"

def test_fpdf_html():
    if not os.path.exists(FONT_PATH_HINDI):
        import requests
        r = requests.get(FONT_URL_HINDI)
        with open(FONT_PATH_HINDI, 'wb') as f:
            f.write(r.content)

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("NotoSansDevanagari", style="", fname=FONT_PATH_HINDI)
    # Add Bold and Italic variants if we want them to work in HTML, 
    # but for test just use regular for all
    pdf.add_font("NotoSansDevanagari", style="B", fname=FONT_PATH_HINDI)
    pdf.add_font("NotoSansDevanagari", style="I", fname=FONT_PATH_HINDI)
    
    pdf.set_font("NotoSansDevanagari", size=12)
    try:
        import uharfbuzz
        pdf.set_text_shaping(True)
    except ImportError:
        pass

    text = """
# परीक्षा प्रश्न पत्र
कक्षा 10 - विज्ञान
    
**प्रश्न 1:** प्रकाश संश्लेषण क्या है?
    
उत्तर: पौधे सूर्य के प्रकाश का उपयोग करते हैं।
"""
    html_content = markdown.markdown(text)
    
    # write_html
    pdf.write_html(html_content)
    
    with open("test_html.pdf", "wb") as f:
        f.write(bytes(pdf.output()))
    print("Done")

if __name__ == "__main__":
    test_fpdf_html()
