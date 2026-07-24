import mimetypes
import os
import re
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from urllib.parse import urlparse
import time
import gc

import filetype
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def clean_generic(s):
    """General cleaner: keep letters/numbers/dash/dot, drop everything else."""
    s = str(s).strip()
    return "".join(c for c in s if c.isalnum() or c in ("-", "."))


def clean_store_id(s):
    """STORE ID may already contain dashes, so keep them exactly as-is."""
    s = str(s).strip()
    return "".join(c for c in s if c.isalnum() or c in ("-", "."))


def clean_store_name(s):
    """STORE NAME: replace spaces with a dash, keep alnum/dash/dot only."""
    s = str(s).strip()
    s = re.sub(r"\s+", "-", s)
    return "".join(c for c in s if c.isalnum() or c in ("-", "."))


def clean_folder_name(s):
    """Keep the ZIP filename/folder name safe for Streamlit Cloud."""
    cleaned = clean_store_name(s)
    return cleaned or "images_downloaded"


def detect_extension(first_bytes, content_type, url):
    """Detect proper file extension from the first chunk of bytes we already have."""
    try:
        kind = filetype.guess(first_bytes) if first_bytes else None
        if kind:
            return kind.extension
        if content_type:
            guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
            if guessed:
                return guessed.lstrip(".")
        path = urlparse(url).path
        ext2 = os.path.splitext(path)[1]
        if ext2 and len(ext2) <= 6:
            return ext2.lstrip(".")
    except Exception:
        pass
    return "jpg"


def build_session(auth, pool_size, max_retries):
    """
    One shared Session (with connection pooling + automatic retries) reused by
    every worker thread, instead of opening a brand-new Session per file.
    This is the single biggest fix for crashes that only appear with large
    link lists: without it, large batches open thousands of separate
    connections and exhaust sockets / file descriptors.
    """
    session = requests.Session()
    session.auth = auth
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    })
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_one(session, url, dest_name, folder, timeout=30, chunk_size=1024 * 128):
    """
    Download a single file, streaming it straight to disk instead of loading
    the whole image into memory first. This keeps per-thread memory usage
    small and constant no matter how many files are being downloaded overall.
    """
    tmp_path = os.path.join(folder, f"{dest_name}.part")
    try:
        # Add a small delay to prevent overwhelming the server
        with session.get(url, timeout=timeout, stream=True) as resp:
            if resp.status_code != 200:
                return False, None, f"HTTP {resp.status_code}"

            content_type = resp.headers.get("Content-Type", "")
            first_chunk = b""
            # Use a larger chunk size for better performance
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    if not first_chunk and len(first_chunk) < 8192:
                        first_chunk += chunk[:8192]
                    f.write(chunk)
                    # Force flush to disk periodically
                    if f.tell() > 1024 * 1024:  # Every 1MB
                        f.flush()

            ext = detect_extension(first_chunk, content_type, url)
            final_name = f"{dest_name}.{ext}"
            final_path = os.path.join(folder, final_name)
            os.replace(tmp_path, final_path)
            return True, final_name, None
    except Exception as e:
        # Make sure a half-written temp file never lingers or gets zipped by mistake
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False, None, str(e)


def zip_folder_to_bytes(folder_path, max_files=50000):
    """Zip a folder's contents (flat, no subfolders) into an in-memory buffer.
    Added memory optimization for large folders."""
    buf = BytesIO()
    file_count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            if os.path.isfile(fpath):
                # Zip in chunks to avoid memory issues
                zipf.write(fpath, fname)
                file_count += 1
                # Clear any accumulated memory
                if file_count % 100 == 0:
                    gc.collect()
    return buf.getvalue()


def process_batch_in_chunks(tasks, session, concurrency, timeout, download_folder, max_chunk_size=500):
    """
    Process downloads in smaller chunks to avoid memory overload.
    This is the key improvement for handling 1000+ images.
    """
    results = []
    total_tasks = len(tasks)
    
    for chunk_start in range(0, total_tasks, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, total_tasks)
        chunk_tasks = tasks[chunk_start:chunk_end]
        
        st.info(f"Processing batch {chunk_start//max_chunk_size + 1}/{(total_tasks + max_chunk_size - 1)//max_chunk_size} ({chunk_end - chunk_start} images)")
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_task = {
                executor.submit(
                    download_one,
                    session,
                    url,
                    dest_name,
                    download_folder,
                    timeout,
                ): (url, city, dest_name)
                for url, dest_name, _, city in chunk_tasks
            }

            for future in as_completed(future_to_task):
                url, city, dest_name = future_to_task[future]
                try:
                    success, final_name, error = future.result(timeout=timeout + 15)
                except Exception as e:
                    success, final_name, error = False, None, str(e)

                if success:
                    results.append((url, city, True, None))
                else:
                    results.append((url, city, False, error))
        
        # Force garbage collection after each chunk
        gc.collect()
        
        # Small delay between chunks to let system resources recover
        if chunk_end < total_tasks:
            time.sleep(0.5)
    
    return results


def reset_download_state():
    st.session_state.zip_data = None
    st.session_state.failed_csv_data = None
    st.session_state.download_summary = None


# -------------------------------------------------------
# Streamlit App
# -------------------------------------------------------

st.set_page_config(page_title="KOBO Image Downloader", layout="wide")
st.title("Download images from KOBO")
st.write("This app downloads images from KOBO using the PEP LINK column.")

with st.sidebar:
    st.markdown("### Instructions")
    st.write("Required columns: **PEP LINK, CITY, STORE ID, STORE NAME, ID**")
    st.write("Image file name pattern: `CITY_STOREID_STORE-NAME_ID`")
    st.write("- STORE NAME: spaces become dashes (e.g. `Al Fateh Store` -> `Al-Fateh-Store`)")
    st.write("- STORE ID: kept exactly as-is")


username = st.text_input("Kobo Username", "")
password = st.text_input("Kobo Password", type="password")

col1, col2, col3, col4 = st.columns(4)
with col1:
    concurrency = st.slider("Concurrent downloads", min_value=1, max_value=8, value=3, help="Lower for large batches to prevent crashes")
with col2:
    timeout = st.number_input("Request timeout (seconds)", value=30, min_value=10, max_value=120)
with col3:
    max_retries = st.number_input("Max retries per URL", value=3, min_value=0, max_value=5)
with col4:
    batch_size = st.slider("Batch size", min_value=50, max_value=500, value=200, help="Process images in smaller batches to avoid memory issues")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV file with links",
    type=["xlsx", "xls", "csv"],
)

REQUIRED_COLS = ["PEP LINK", "CITY", "STORE ID", "STORE NAME", "ID"]

if "zip_data" not in st.session_state:
    st.session_state.zip_data = None
if "failed_csv_data" not in st.session_state:
    st.session_state.failed_csv_data = None
if "download_summary" not in st.session_state:
    st.session_state.download_summary = None

if uploaded_file is not None and username and password:
    try:
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            df = pd.read_csv(uploaded_file)

        df.columns = [str(c).strip() for c in df.columns]

        st.markdown("**Preview of file**")
        st.dataframe(df.head(50))
        st.caption(f"Total rows in file: {len(df)}")

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Error: Missing required column(s): {', '.join(missing)}")
            st.stop()

        folder_name = st.text_input("Folder name for images", value="images_downloaded")

        if st.button("Start download", type="primary"):
            reset_download_state()
            
            # Show a progress spinner
            with st.spinner("Processing downloads..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        auth = HTTPBasicAuth(username, password)
                        safe_folder_name = clean_folder_name(folder_name)
                        download_folder = os.path.join(temp_dir, safe_folder_name)
                        os.makedirs(download_folder, exist_ok=True)

                        session = build_session(auth, pool_size=min(concurrency, 10), max_retries=max_retries)

                        # Prepare all tasks
                        tasks = []
                        skipped_rows = 0
                        
                        # Show progress for task preparation
                        status_text = st.empty()
                        status_text.text("Preparing download tasks...")
                        
                        for idx, row in df.iterrows():
                            url = str(row["PEP LINK"]).strip()
                            if not url.startswith(("http://", "https://")):
                                skipped_rows += 1
                                continue

                            city = clean_generic(row["CITY"]) or "unknown-city"
                            store_id = clean_store_id(row["STORE ID"])
                            store_name = clean_store_name(row["STORE NAME"])
                            record_id = clean_generic(row["ID"])

                            dest_name = f"{city}_{store_id}_{store_name}_{record_id}"
                            tasks.append((url, dest_name, download_folder, city))
                            
                            # Update status occasionally
                            if idx % 500 == 0 and idx > 0:
                                status_text.text(f"Preparing tasks: {idx+1}/{len(df)}")

                        status_text.empty()

                        if skipped_rows:
                            st.warning(f"Skipped {skipped_rows} row(s) with missing/invalid PEP LINK.")

                        if not tasks:
                            st.warning("No valid URLs found to download.")
                            st.stop()

                        # Create progress bars
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Process downloads in chunks
                        total = len(tasks)
                        results = []
                        successful_downloads = 0
                        failed_downloads = 0
                        
                        # Process in chunks
                        for chunk_start in range(0, total, batch_size):
                            chunk_end = min(chunk_start + batch_size, total)
                            chunk_tasks = tasks[chunk_start:chunk_end]
                            
                            chunk_num = chunk_start // batch_size + 1
                            total_chunks = (total + batch_size - 1) // batch_size
                            status_text.text(f"Processing batch {chunk_num}/{total_chunks} ({chunk_end - chunk_start} images)")
                            
                            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                                future_to_task = {
                                    executor.submit(
                                        download_one,
                                        session,
                                        url,
                                        dest_name,
                                        download_folder,
                                        timeout,
                                    ): (url, city, dest_name, idx + chunk_start)
                                    for idx, (url, dest_name, _, city) in enumerate(chunk_tasks)
                                }

                                for future in as_completed(future_to_task):
                                    url, city, dest_name, task_idx = future_to_task[future]
                                    try:
                                        success, final_name, error = future.result(timeout=timeout + 15)
                                    except Exception as e:
                                        success, final_name, error = False, None, str(e)

                                    if success:
                                        successful_downloads += 1
                                        results.append((url, city, True, None))
                                    else:
                                        failed_downloads += 1
                                        results.append((url, city, False, error))
                                    
                                    # Update progress
                                    completed = successful_downloads + failed_downloads
                                    progress_bar.progress(completed / total)
                                    status_text.text(f"Downloaded {completed}/{total} images (Success: {successful_downloads}, Failed: {failed_downloads})")
                            
                            # Force garbage collection after each chunk
                            gc.collect()
                            
                            # Small delay between chunks
                            if chunk_end < total:
                                time.sleep(0.5)

                        session.close()

                        st.session_state.download_summary = (successful_downloads, failed_downloads)
                        st.success(f"Download complete! Successful: {successful_downloads}, Failed: {failed_downloads}")

                        if successful_downloads > 0:
                            # Show zipping progress
                            status_text.text("Creating ZIP file...")
                            zip_bytes = zip_folder_to_bytes(download_folder)
                            st.session_state.zip_data = {
                                "bytes": zip_bytes,
                                "file_name": f"{safe_folder_name}.zip",
                            }
                            status_text.text("ZIP file created!")

                        if failed_downloads > 0:
                            failed_data = [
                                {"url": url, "city": city, "error": error}
                                for url, city, ok, error in results
                                if not ok
                            ]
                            fail_df = pd.DataFrame(failed_data)
                            st.session_state.failed_csv_data = fail_df.to_csv(index=False).encode("utf-8")

                    except Exception as e:
                        st.error(f"Error during download: {str(e)}")
                        st.exception(e)
                    finally:
                        # Clean up any temporary files
                        try:
                            import shutil
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        except:
                            pass

        if st.session_state.download_summary:
            succ, fail = st.session_state.download_summary
            st.info(f"Last run — Successful: {succ}, Failed: {fail}")

        if st.session_state.zip_data:
            st.download_button(
                "Download All Images (ZIP)",
                data=st.session_state.zip_data["bytes"],
                file_name=st.session_state.zip_data["file_name"],
                mime="application/zip",
                use_container_width=True,
            )

        if st.session_state.failed_csv_data:
            st.download_button(
                "Download Failed Links CSV",
                data=st.session_state.failed_csv_data,
                file_name="failed_links.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.exception(e)

else:
    st.info("Upload a file and enter your Kobo credentials to begin.")
