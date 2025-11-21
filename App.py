import streamlit as st
import pandas as pd
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
import json
import time
from io import BytesIO
from bs4 import BeautifulSoup

# Page configuration
st.set_page_config(
    page_title="HS Code Finder",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .product-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .info-label {
        font-weight: bold;
        color: #2c3e50;
        font-size: 16px;
    }
    .info-value {
        color: #34495e;
        font-size: 15px;
        margin-bottom: 12px;
    }
    .stButton>button {
        background-color: #27ae60;
        color: white;
        font-size: 18px;
        padding: 10px 30px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #229954;
    }
</style>
""", unsafe_allow_html=True)

# Initialize directories and files
def initialize_app():
    """Create necessary directories and files on first run"""
    if not os.path.exists('images'):
        os.makedirs('images')
    
    if not os.path.exists('hs_data.csv'):
        create_sample_data()
    else:
        try:
            df = pd.read_csv('hs_data.csv')
            required_columns = ['common_name', 'hs_code', 'scientific_name', 'family', 
                              'sub_family', 'order', 'genus', 'kingdom', 'description']
            if not all(col in df.columns for col in required_columns):
                create_sample_data()
        except:
            create_sample_data()

def create_sample_data():
    """Create empty database structure"""
    sample_data = {
        'common_name': [],
        'hs_code': [],
        'scientific_name': [],
        'family': [],
        'sub_family': [],
        'order': [],
        'genus': [],
        'kingdom': [],
        'description': []
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('hs_data.csv', index=False)

def generate_placeholder_image(product_name):
    """Generate a placeholder image for products without images"""
    img = Image.new('RGB', (400, 400), color=(230, 240, 250))
    draw = ImageDraw.Draw(img)
    
    draw.ellipse([100, 100, 300, 300], fill=(100, 180, 100), outline=(50, 150, 50), width=3)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text = product_name.upper()[:20]
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((400 - text_width) // 2, (400 - text_height) // 2)
    draw.text(position, text, fill=(255, 255, 255), font=font)
    
    return img

def fetch_image_from_wikipedia(product_name):
    """Fetch product image from Wikipedia (most accurate)"""
    try:
        search_terms = [
            product_name.replace(' ', '_'),
            product_name.title().replace(' ', '_'),
            product_name.capitalize().replace(' ', '_'),
        ]
        
        for search_term in search_terms:
            try:
                search_url = f"https://en.wikipedia.org/w/api.php"
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': search_term,
                    'prop': 'pageimages',
                    'pithumbsize': 400
                }
                
                response = requests.get(search_url, params=search_params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    pages = data.get('query', {}).get('pages', {})
                    
                    for page_id, page_data in pages.items():
                        if page_id != '-1' and 'thumbnail' in page_data:
                            image_url = page_data['thumbnail']['source']
                            
                            img_response = requests.get(image_url, timeout=10)
                            if img_response.status_code == 200:
                                img = Image.open(BytesIO(img_response.content))
                                img = img.resize((400, 400), Image.Resampling.LANCZOS)
                                return img
            except:
                continue
        
        return None
    except Exception as e:
        return None

def fetch_image_from_google(product_name):
    """Fetch product image from Google Images"""
    try:
        search_term = product_name.replace(' ', '+')
        url = f"https://www.google.com/search?q={search_term}+plant&tbm=isch"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            img_tags = soup.find_all('img')
            
            for img_tag in img_tags[1:6]:
                try:
                    img_url = img_tag.get('src') or img_tag.get('data-src')
                    
                    if img_url and img_url.startswith('http'):
                        img_response = requests.get(img_url, timeout=10)
                        
                        if img_response.status_code == 200 and len(img_response.content) > 5000:
                            img = Image.open(BytesIO(img_response.content))
                            
                            if img.size[0] >= 100 and img.size[1] >= 100:
                                img = img.resize((400, 400), Image.Resampling.LANCZOS)
                                return img
                except:
                    continue
        
        return None
    except Exception as e:
        return None

def get_product_image(product_name, img_path, scientific_name=None):
    """Get product image from multiple sources"""
    
    # Try Wikipedia first with scientific name
    if scientific_name and scientific_name != 'Not available':
        img = fetch_image_from_wikipedia(scientific_name)
        if img:
            img.save(img_path)
            st.success("✅ Image from Wikipedia (scientific name)")
            return img
    
    # Try Wikipedia with common name
    img = fetch_image_from_wikipedia(product_name)
    if img:
        img.save(img_path)
        st.success("✅ Image from Wikipedia")
        return img
    
    # Try Google Images
    img = fetch_image_from_google(product_name)
    if img:
        img.save(img_path)
        st.success("✅ Image from Google Images")
        return img
    
    # Try Google Images with scientific name
    if scientific_name and scientific_name != 'Not available':
        img = fetch_image_from_google(scientific_name)
        if img:
            img.save(img_path)
            st.success("✅ Image from Google (scientific name)")
            return img
    
    # Generate placeholder if all fail
    st.info("ℹ️ Using placeholder image")
    img = generate_placeholder_image(product_name)
    img.save(img_path)
    return img

def get_info_from_groq(product_name):
    """Get product information using Groq's free LLM API"""
    try:
        api_key = os.environ.get('GROQ_API_KEY', '')
        
        if not api_key:
            st.error("❌ Groq API key not found. Please add it in the sidebar.")
            return None
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""You are an expert in agricultural products and international trade classification. 
Provide ACCURATE information for the product: "{product_name}"

Return ONLY a valid JSON object with this exact structure (no additional text):
{{
    "common_name": "exact common name",
    "hs_code": "4 or 6 digit HS code",
    "scientific_name": "scientific binomial name",
    "family": "taxonomic family",
    "sub_family": "taxonomic sub-family",
    "order": "taxonomic order",
    "genus": "taxonomic genus",
    "kingdom": "Plantae or appropriate kingdom",
    "description": "2-3 sentence description"
}}

IMPORTANT: 
- Use REAL HS codes (Harmonized System codes used in international trade)
- Provide accurate botanical/scientific names
- If you're not certain about any field, use "Not available" as the value
- Return ONLY the JSON object, no markdown formatting, no explanations"""

        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            product_info = json.loads(content)
            st.success("✅ Information retrieved from Groq AI")
            return product_info
        else:
            st.error(f"❌ Groq API Error: {response.status_code} - {response.text}")
            return None
            
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse response: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error calling Groq API: {str(e)}")
        return None

def get_product_info_with_ai(product_name):
    """Try multiple AI sources to get accurate information"""
    
    st.info("🤖 Using AI to fetch accurate product information...")
    product_info = get_info_from_groq(product_name)
    
    if product_info:
        st.success("✅ Information retrieved from Groq AI")
        return product_info
    
    st.error("❌ Unable to fetch from AI. Please check your API key.")
    return None

def save_to_csv(product_info):
    """Save new product information to CSV"""
    try:
        df = pd.read_csv('hs_data.csv')
        
        existing = df[df['common_name'].str.lower() == product_info['common_name'].lower()]
        
        if not existing.empty:
            df.loc[df['common_name'].str.lower() == product_info['common_name'].lower()] = list(product_info.values())
        else:
            new_df = pd.DataFrame([product_info])
            df = pd.concat([df, new_df], ignore_index=True)
        
        df.to_csv('hs_data.csv', index=False)
        return True
    except Exception as e:
        return False

def translate_text(text, target_lang):
    """Translate text to target language"""
    if target_lang == 'en' or not text or text in ['Not available', 'Not found']:
        return text
    
    try:
        translator = GoogleTranslator(source='en', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        return text

def search_product(product_name, language_code):
    """Search for product in database or use AI"""
    try:
        df = pd.read_csv('hs_data.csv')
        
        if len(df) > 0:
            result = df[df['common_name'].str.lower() == product_name.lower()]
            
            if not result.empty:
                st.info("✅ Found in local database")
                product_info = result.iloc[0].to_dict()
                
                if language_code != 'en':
                    product_info['description'] = translate_text(product_info['description'], language_code)
                    product_info['family'] = translate_text(product_info['family'], language_code)
                    product_info['sub_family'] = translate_text(product_info['sub_family'], language_code)
                    product_info['order'] = translate_text(product_info['order'], language_code)
                    product_info['genus'] = translate_text(product_info['genus'], language_code)
                    product_info['kingdom'] = translate_text(product_info['kingdom'], language_code)
                
                return product_info
        
        st.info("🔍 Product not found in local database. Using AI...")
        product_info = get_product_info_with_ai(product_name)
        
        if product_info:
            required_fields = ['common_name', 'hs_code', 'scientific_name', 'family', 
                             'sub_family', 'order', 'genus', 'kingdom', 'description']
            
            for field in required_fields:
                if field not in product_info:
                    product_info[field] = 'Not available'
            
            if save_to_csv(product_info):
                st.success("💾 Product information saved to database!")
            
            if language_code != 'en':
                product_info['description'] = translate_text(product_info['description'], language_code)
                product_info['family'] = translate_text(product_info['family'], language_code)
                product_info['sub_family'] = translate_text(product_info['sub_family'], language_code)
                product_info['order'] = translate_text(product_info['order'], language_code)
                product_info['genus'] = translate_text(product_info['genus'], language_code)
                product_info['kingdom'] = translate_text(product_info['kingdom'], language_code)
            
            return product_info
        else:
            return None
    
    except Exception as e:
        st.error(f"❌ Error searching product: {str(e)}")
        return None

# Initialize app
initialize_app()

# Header
st.title("🌾 HS Code Finder for Agriculture & Products")
st.markdown("### AI-Powered Search for HS codes, scientific names, and classifications")

# API Setup in Sidebar
with st.sidebar:
    st.header("⚙️ API Setup")
    
    st.subheader("🔑 Enter API Key")
    
    groq_key = st.text_input("Groq API Key:", type="password", value=os.environ.get('GROQ_API_KEY', ''))
    if groq_key:
        os.environ['GROQ_API_KEY'] = groq_key
    
    st.markdown("---")
    
    st.markdown("""
    **🎯 Image Sources:**
    1. Wikipedia (no key needed) ✅
    2. Google Images (no key needed) ✅
    
    **How to get Groq API key:**
    1. Visit [console.groq.com](https://console.groq.com/)
    2. Sign up for free
    3. Get your API key
    4. Paste it above
    """)
    
    if os.environ.get('GROQ_API_KEY'):
        st.success("✅ Groq API configured")
    else:
        st.warning("⚠️ Groq API not configured")

# Language selection
languages = {
    'English': 'en',
    'Tamil': 'ta',
    'Hindi': 'hi',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Chinese': 'zh-CN',
    'Japanese': 'ja',
    'Arabic': 'ar'
}

col1, col2 = st.columns([3, 1])
with col1:
    product_input = st.text_input("🔍 Enter product name:", placeholder="e.g., Apple, Banana, Rice, Mango...")
with col2:
    selected_language = st.selectbox("🌐 Language:", list(languages.keys()))

search_button = st.button("🔎 Search", use_container_width=True)

st.markdown("**💡 Try:** Apple, Banana, Rice, Wheat, Turmeric, Cardamom, Coffee, Tea, Mango, etc.")

# Search functionality
if search_button and product_input:
    with st.spinner("🔍 Searching..."):
        language_code = languages[selected_language]
        product_info = search_product(product_input, language_code)
        
        if product_info:
            col_img, col_info = st.columns([1, 2])
            
            with col_img:
                img_path = f"images/{product_info['common_name'].lower().replace(' ', '_')}.png"
                
                col_a, col_b = st.columns([3, 1])
                with col_b:
                    refresh_image = st.button("🔄", key="refresh_img")
                
                if refresh_image and os.path.exists(img_path):
                    os.remove(img_path)
                    st.rerun()
                
                if os.path.exists(img_path) and not refresh_image:
                    st.image(img_path, use_container_width=True)
                else:
                    with st.spinner("📷 Fetching image..."):
                        scientific_name = product_info.get('scientific_name', None)
                        img = get_product_image(product_info['common_name'], img_path, scientific_name)
                        st.image(img, use_container_width=True)
            
            with col_info:
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                
                st.markdown(f"<p class='info-label'>Common Name:</p><p class='info-value'>{product_info['common_name']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='info-label'>HS Code:</p><p class='info-value'>{product_info['hs_code']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='info-label'>Category (Family):</p><p class='info-value'>{product_info['family']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='info-label'>Sub Category (Sub Family):</p><p class='info-value'>{product_info['sub_family']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='info-label'>Scientific Name:</p><p class='info-value'><em>{product_info['scientific_name']}</em></p>", unsafe_allow_html=True)
                st.markdown(f"<p class='info-label'>Order:</p><p class='info-value'>{product_info['order']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='info-label'>Genus:</p><p class='info-value'>{product_info['genus']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='info-label'>Kingdom:</p><p class='info-value'>{product_info['kingdom']}</p>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            st.markdown(f"<p class='info-label'>Description:</p>", unsafe_allow_html=True)
            st.write(product_info['description'])
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.error("❌ Unable to fetch product information. Please check your Groq API key in the sidebar.")

elif search_button and not product_input:
    st.warning("⚠️ Please enter a product name to search.")

st.markdown("---")
st.markdown("💡 **Powered by AI** | 🖼️ **Images from Wikipedia & Google**")
