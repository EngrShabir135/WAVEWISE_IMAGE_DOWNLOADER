import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse
import os
import mimetypes
import time
import re
from io import BytesIO
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import filetype
import tempfile
from pathlib import Path


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def clean_generic(s):
    """General cleaner: keep letters/numbers/dash/dot, drop everything else (spaces removed)."""
    s = str(s).strip()
    return "".join(c for c in s if c.isalnum() or c in ('-', '.'))

def clean_store_id(s):
    """STORE ID may already contain dashes -> keep them exactly as-is."""
    s = str(s).strip()
    return "".join(c for c in s if c.isalnum() or c in ('-', '.'))

def clean_store_name(s):
    """STORE NAME: replace spaces with a dash, keep alnum/dash/dot only."""
    s = str(s).strip()
    s = re.sub(r'\s+', '-', s)
    return "".join(c for c in s if c.isalnum() or c in ('-', '.'))

def detect_extension(content, content_type, url):
    """Detect proper file extension."""
    try:
        kind = filetype.guess(content)
        if kind:
            return kind.extension
        if content_type:
            guessed = mimetypes.guess_extension(content_type.split(';')[0].strip())
            if guessed:
                return guessed.lstrip('.')
        path = urlparse(url).path
        ext2 = os.path.splitext(path)[1]
        if ext2 and len(ext2) <= 6:
            return ext2.lstrip('.')
    except:
        pass
    return 'jpg'

def download_one(url, dest_name, folder, auth, timeout=30, max_retries=3):
    """Download a single file with retries."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            with requests.Session() as session:
                session.auth = auth
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                resp = session.get(url, stream=True, timeout=timeout)
                if resp.status_code == 200:
                    content = resp.content
                    content_type = resp.headers.get('Content-Type', '')
                    ext = detect_extension(content, content_type, url)
                    final_name = f"{dest_name}.{ext}"
                    final_path = os.path.join(folder, final_name)
                    with open(final_path, 'wb') as f:
                        f.write(content)
                    return True, final_name, None
                else:
                    last_exc = f'HTTP {resp.status_code}'
        except Exception as e:
            last_exc = str(e)
        if attempt < max_retries:
            time.sleep(1 * (attempt + 1))
    return False, None, last_exc


# -------------------------------------------------------
# Streamlit App
# -------------------------------------------------------

st.set_page_config(page_title="KOBO Image Downloader", layout="wide")
st.title("📊 Download images from KOBO")
st.write("This app downloads images from KOBO using the PEP LINK column.")

# Sidebar for instructions
with st.sidebar:
    st.markdown("### Instructions")
    st.write("Required columns: **PEP LINK, CITY, STORE ID, STORE NAME, ID**")
    st.write("Image file name pattern: `CITY_STOREID_STORE-NAME_ID`")
    st.write("- STORE NAME: spaces become dashes (e.g. `Al Fateh Store` → `Al-Fateh-Store`)")
    st.write("- STORE ID: kept exactly as-is")

# Main content
username = st.text_input('Kobo Username', '')
password = st.text_input('Kobo Password', type='password')

# Download settings
col1, col2, col3 = st.columns(3)
with col1:
    concurrency = st.slider('Concurrent downloads', min_value=1, max_value=5, value=3)
with col2:
    timeout = st.number_input('Request timeout (seconds)', value=30, min_value=10, max_value=120)
with col3:
    max_retries = st.number_input('Max retries per URL', value=3, min_value=0, max_value=5)

uploaded_file = st.file_uploader(
    'Upload Excel or CSV file with links',
    type=['xlsx', 'xls', 'csv']
)

REQUIRED_COLS = ["PEP LINK", "CITY", "STORE ID", "STORE NAME", "ID"]

if uploaded_file is not None and username and password:
    try:
        if uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        
        st.markdown('**Preview of file**')
        st.dataframe(df.head(50))

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Error: Missing required column(s): {', '.join(missing)}")
            st.stop()

        folder_name = st.text_input('Folder name for images', value='images_downloaded')

        if st.button('🚀 Start download', type='primary'):
            with st.spinner("Processing downloads..."):
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        auth = HTTPBasicAuth(username, password)
                        download_folder = os.path.join(temp_dir, folder_name)
                        os.makedirs(download_folder, exist_ok=True)

                        # Prepare tasks
                        tasks = []
                        for _, row in df.iterrows():
                            url = str(row["PEP LINK"]).strip()
                            if not url.startswith(('http://', 'https://')):
                                continue

                            city = clean_generic(row["CITY"])
                            store_id = clean_store_id(row["STORE ID"])
                            store_name = clean_store_name(row["STORE NAME"])
                            record_id = clean_generic(row["ID"])

                            city_folder = os.path.join(download_folder, city)
                            os.makedirs(city_folder, exist_ok=True)

                            dest_name = f"{city}_{store_id}_{store_name}_{record_id}"
                            tasks.append((url, dest_name, city_folder, city))

                        if not tasks:
                            st.warning("No valid URLs found to download.")
                            st.stop()

                        # Track progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = []
                        total = len(tasks)
                        done = 0

                        # Download with ThreadPool
                        with ThreadPoolExecutor(max_workers=concurrency) as executor:
                            future_to_task = {
                                executor.submit(
                                    download_one, url, dest_name, city_folder, auth, timeout, max_retries
                                ): (url, city)
                                for url, dest_name, city_folder, city in tasks
                            }

                            for future in as_completed(future_to_task):
                                url, city = future_to_task[future]
                                try:
                                    success, final_name, error = future.result(timeout=timeout+10)
                                except Exception as e:
                                    success, final_name, error = False, None, str(e)
                                
                                done += 1
                                progress_bar.progress(done / total)
                                status_text.text(f"Downloading {done}/{total} images...")
                                
                                if success:
                                    results.append((url, os.path.join(city, final_name), True, None))
                                else:
                                    results.append((url, None, False, error))

                        # Show results
                        succ = sum(1 for r in results if r[2])
                        fail = sum(1 for r in results if not r[2])
                        
                        st.success(f"✅ Download complete! Successful: {succ}, Failed: {fail}")

                        # Create ZIP if any successful downloads
                        if succ > 0:
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                for _, fname, ok, _ in results:
                                    if ok and fname:
                                        fpath = os.path.join(download_folder, fname)
                                        if os.path.exists(fpath):
                                            arcname = os.path.relpath(fpath, download_folder)
                                            zipf.write(fpath, arcname)
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                "📥 Download All Images (ZIP)",
                                data=zip_buffer,
                                file_name=f"{folder_name}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )

                        # Show failed links
                        if fail > 0:
                            failed_data = []
                            for url, _, ok, error in results:
                                if not ok:
                                    failed_data.append({'url': url, 'error': error})
                            fail_df = pd.DataFrame(failed_data)
                            csv_buffer = BytesIO()
                            fail_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                '📄 Download Failed Links CSV',
                                data=csv_buffer.getvalue(),
                                file_name='failed_links.csv',
                                mime='text/csv',
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"Error during download: {str(e)}")
                        st.exception(e)

else:
    st.info('📤 Upload a file and enter your Kobo credentials to begin.')
