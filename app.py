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


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def clean_generic(s):
    """General cleaner: keep letters/numbers/dash, drop everything else (spaces removed)."""
    s = str(s).strip()
    return "".join(c for c in s if c.isalnum() or c == '-')


def clean_store_id(s):
    """STORE ID may already contain dashes -> keep them exactly as-is.
    Only strip characters that are not alnum/dash (e.g. stray spaces, slashes)."""
    s = str(s).strip()
    return "".join(c for c in s if c.isalnum() or c == '-')


def clean_store_name(s):
    """STORE NAME: replace spaces with a dash, keep alnum/dash only."""
    s = str(s).strip()
    s = re.sub(r'\s+', '-', s)
    return "".join(c for c in s if c.isalnum() or c == '-')


def detect_extension(content, content_type, url):
    """Detect proper file extension."""
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
    return 'jpg'


def download_one(session, url, dest_name, folder, timeout=20, max_retries=2):
    """Download a single file with retries."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
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
        time.sleep(0.5 * (attempt + 1))
    return False, None, last_exc


# -------------------------------------------------------
# Streamlit App
# -------------------------------------------------------

st.title("📊 Download images from KOBO")
st.write("This app downloads images from KOBO using the PEP LINK column.")
st.write("Required columns: **PEP LINK, CITY, STORE ID, STORE NAME, ID**")
st.write("Image file name pattern: `CITY_STOREID_STORE-NAME_ID`")
st.write("- STORE NAME: spaces become dashes (e.g. `Al Fateh Store` → `Al-Fateh-Store`)")
st.write("- STORE ID: kept exactly as-is (existing dashes are preserved, not touched)")

# Username and Password
username = st.text_input('Kobo Username', '')
password = st.text_input('Kobo Password', type='password')

concurrency = st.slider('Concurrent downloads', min_value=1, max_value=10, value=3)
timeout = st.number_input('Request timeout (seconds)', value=20, min_value=5, max_value=120)
max_retries = st.number_input('Max retries per URL', value=2, min_value=0, max_value=5)

uploaded_file = st.file_uploader(
    'Upload Excel or CSV file with links (must include PEP LINK, CITY, STORE ID, STORE NAME, ID)',
    type=['xlsx', 'xls', 'csv']
)

REQUIRED_COLS = ["PEP LINK", "CITY", "STORE ID", "STORE NAME", "ID"]

if uploaded_file is not None and username and password:
    try:
        if uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f'Error reading file: {e}')
        st.stop()

    # normalize column names (strip extra whitespace) so header variations still match
    df.columns = [str(c).strip() for c in df.columns]

    st.markdown('**Preview of file**')
    st.dataframe(df.head(50))

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Error: Missing required column(s): {', '.join(missing)}")
        st.stop()

    folder_name = st.text_input('Grand folder to save images', value='images_downloaded')

    if st.button('Start download'):
        with st.spinner("Downloading..."):
            try:
                session = requests.Session()
                session.auth = HTTPBasicAuth(username, password)
                os.makedirs(folder_name, exist_ok=True)

                results = []
                future_to_row = {}

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    for _, row in df.iterrows():
                        url = str(row["PEP LINK"]).strip()
                        if not (url.startswith("http://") or url.startswith("https://")):
                            continue

                        city = clean_generic(row["CITY"])
                        store_id = clean_store_id(row["STORE ID"])
                        store_name = clean_store_name(row["STORE NAME"])
                        record_id = clean_generic(row["ID"])

                        city_folder = os.path.join(folder_name, city)
                        os.makedirs(city_folder, exist_ok=True)

                        dest_name = f"{city}_{store_id}_{store_name}_{record_id}"

                        future = executor.submit(
                            download_one, session, url, dest_name, city_folder, timeout, max_retries
                        )
                        future_to_row[future] = (url, city)

                    progress_bar = st.progress(0)
                    done = 0
                    total = len(future_to_row)
                    log_lines = []

                    if total == 0:
                        st.warning("No valid PEP LINK URLs found to download.")

                    for future in as_completed(future_to_row):
                        url, city = future_to_row[future]
                        success, final_name, error = future.result()
                        done += 1
                        if total > 0:
                            progress_bar.progress(done / total)

                        if success:
                            log_lines.append(f'✅ {city}: {url} -> {final_name}')
                            results.append((url, os.path.join(city, final_name), True, None))
                        else:
                            log_lines.append(f'❌ {city}: {url} -> {error}')
                            results.append((url, None, False, error))

                        if done % 10 == 0:
                            st.text("\n".join(log_lines[-20:]))

                succ = sum(1 for r in results if r[2])
                fail = sum(1 for r in results if not r[2])
                st.success(f"Download complete ✅ Successful: {succ}, Failed: {fail}")

                if succ > 0:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for _, fname, ok, _ in results:
                            if ok and fname:
                                fpath = os.path.join(folder_name, fname)
                                if os.path.exists(fpath):
                                    zipf.write(fpath, fname)
                    zip_buffer.seek(0)
                    st.download_button("Download ZIP", data=zip_buffer, file_name=f"{folder_name}.zip")

                if fail > 0:
                    failed_links = [url for url, _, ok, _ in results if not ok]
                    fail_df = pd.DataFrame(failed_links, columns=['failed_url'])
                    csv_buffer = BytesIO()
                    fail_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        'Download failed links CSV',
                        data=csv_buffer.getvalue(),
                        file_name='failed_links.csv',
                        mime='text/csv'
                    )

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info('Upload a file and enter your Kobo username & password to begin.')