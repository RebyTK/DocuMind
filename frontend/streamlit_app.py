
import os

import requests
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv(
    "DOCUMIND_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# --------------------------------------------------
# API client
# --------------------------------------------------

class APIError(Exception):
    """Raised when the FastAPI backend returns an error."""


def api_request(method, endpoint, **kwargs):
    """Send an HTTP request to the FastAPI backend."""
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{endpoint}",
            timeout=300,
            **kwargs,
        )

        if not response.ok:
            try:
                error_data = response.json()
                detail = error_data.get(
                    "detail", error_data
                ) if isinstance(error_data, dict) else error_data
            except ValueError:
                detail = response.text

            raise APIError(
                f"HTTP {response.status_code}: {detail}"
            )

        if not response.content:
            return {}

        try:
            return response.json()
        except ValueError:
            return response.text

    except requests.ConnectionError as error:
        raise APIError(
            "Cannot connect to FastAPI. "
            "Make sure the backend is running."
        ) from error

    except requests.Timeout as error:
        raise APIError(
            "The request timed out. The operation may "
            "still be processing on the backend."
        ) from error


def check_api_health():
    try:
        api_request("GET", "/health")
        return True
    except APIError:
        return False


def check_database_health():
    try:
        api_request("GET", "/health/database")
        return True
    except APIError:
        return False


# --------------------------------------------------
# Document response helpers
# --------------------------------------------------

def extract_documents(data):
    """Handle common document-list response formats."""
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    for key in ("documents", "items", "results"):
        value = data.get(key)

        if isinstance(value, list):
            return value

    nested = data.get("data")

    if isinstance(nested, (dict, list)):
        return extract_documents(nested)

    return []

def extract_search_results(data):
    """Extract search results from the API response."""
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    results = data.get("results", [])

    if isinstance(results, list):
        return results

    nested = data.get("data")

    if isinstance(nested, (dict, list)):
        return extract_search_results(nested)

    return []


def document_value(document, *keys, default="N/A"):
    """Read a document field using possible field aliases."""
    if not isinstance(document, dict):
        return default

    for key in keys:
        value = document.get(key)

        if value is not None:
            return value

    return default


def display_upload_result(filename, result):
    """Display the outcome of an upload."""
    if not isinstance(result, dict):
        st.success(f"Uploaded {filename}.")
        return

    if result.get("inserted") is False:
        st.warning(
            f"{filename}: This document already exists. "
            "Duplicate ingestion was skipped."
        )
    else:
        st.success(f"{filename} uploaded successfully.")

    document_id = result.get("document_id")

    if document_id:
        st.caption(f"Document ID: {document_id}")

    chunks = result.get("total_chunks")

    if chunks is not None:
        st.caption(f"Chunks created: {chunks}")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Your AI-powered document assistant")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Document Library",
            "Semantic Search",
            "Chat with Documents",
        ],
    )

    st.divider()

    st.subheader("System Status")

    api_online = check_api_health()
    database_online = check_database_health()

    if api_online:
        st.success("FastAPI: Connected")
    else:
        st.error("FastAPI: Offline")

    if database_online:
        st.success("PostgreSQL: Connected")
    else:
        st.error("PostgreSQL: Unavailable")

    st.caption(f"Backend: {API_BASE_URL}")


# --------------------------------------------------
# Main application
# --------------------------------------------------

st.title("DocuMind")
st.caption("Intelligent document processing and retrieval")


# --------------------------------------------------
# Document Library
# --------------------------------------------------

if page == "Document Library":

    st.header("Document Library")

    st.markdown(
        "Upload, manage and explore your indexed documents."
    )

    # Display messages after successful operations
    messages = st.session_state.pop(
        "library_messages", []
    )

    for message_type, message in messages:
        if message_type == "success":
            st.success(message)
        elif message_type == "warning":
            st.warning(message)
        else:
            st.error(message)

    # ----------------------------
    # Upload documents
    # ----------------------------

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can select multiple documents.",
    )

    if st.button(
        "Upload Documents",
        type="primary",
        disabled=not uploaded_files,
    ):
        messages = []

        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):

            status.write(
                f"Processing {uploaded_file.name}..."
            )

            try:
                result = api_request(
                    "POST",
                    "/api/v1/documents",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                            or "application/octet-stream",
                        )
                    },
                )

                if isinstance(result, dict):
                    if result.get("inserted") is False:
                        messages.append(
                            (
                                "warning",
                                f"{uploaded_file.name}: "
                                "Duplicate document. "
                                "Ingestion was skipped.",
                            )
                        )
                    else:
                        chunks = result.get("total_chunks")
                        suffix = (
                            f" ({chunks} chunks)"
                            if chunks is not None
                            else ""
                        )

                        messages.append(
                            (
                                "success",
                                f"{uploaded_file.name} uploaded"
                                f" successfully{suffix}.",
                            )
                        )
                else:
                    messages.append(
                        (
                            "success",
                            f"{uploaded_file.name} uploaded.",
                        )
                    )

            except APIError as error:
                messages.append(
                    (
                        "error",
                        f"{uploaded_file.name}: {error}",
                    )
                )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        status.empty()
        progress.empty()

        st.session_state["library_messages"] = messages
        st.rerun()

    st.divider()

    # ----------------------------
    # Document collection
    # ----------------------------

    st.subheader("Indexed Documents")

    col_title, col_refresh = st.columns(
        [5, 1]
    )

    with col_refresh:
        refresh = st.button(
            "🔄 Refresh",
            use_container_width=True,
        )

    try:
        response_data = api_request(
            "GET",
            "/api/v1/documents",
        )

        documents = extract_documents(response_data)

        if not documents:
            st.info(
                "No documents found. Upload your first "
                "document to get started."
            )

        else:
            # Summary metrics
            total_documents = len(documents)

            total_chunks = sum(
                int(
                    document_value(
                        doc,
                        "total_chunks",
                        "chunk_count",
                        "chunks_count",
                        default=0,
                    )
                )
                for doc in documents
            )

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Total Documents",
                total_documents,
            )

            metric2.metric(
                "Total Chunks",
                total_chunks,
            )

            st.divider()

            # Individual document cards
            for index, document in enumerate(documents):

                if not isinstance(document, dict):
                    continue

                document_id = document_value(
                    document,
                    "document_id",
                    "id",
                    default=f"unknown-{index}",
                )

                filename = document_value(
                    document,
                    "filename",
                    "original_filename",
                    "name",
                )

                file_type = document_value(
                    document,
                    "file_type",
                    "extension",
                )

                chunks = document_value(
                    document,
                    "total_chunks",
                    "chunk_count",
                    "chunks_count",
                )

                created_at = document_value(
                    document,
                    "created_at",
                    "uploaded_at",
                )

                with st.container(border=True):

                    left, right = st.columns(
                        [5, 1]
                    )

                    with left:
                        st.markdown(
                            f"**📄 {filename}**"
                        )

                        st.caption(
                            f"Type: {file_type}  |  "
                            f"Chunks: {chunks}"
                        )

                        st.caption(
                            f"Added: {created_at}"
                        )

                        with st.expander(
                            "Document details"
                        ):
                            st.write(
                                "**Document ID**"
                            )
                            st.code(
                                str(document_id)
                            )

                            metadata = document.get(
                                "metadata"
                            )

                            if metadata:
                                st.write("**Metadata**")
                                st.json(metadata)

                    with right:
                        if st.button(
                            "Delete",
                            key=f"delete_{document_id}",
                            type="secondary",
                        ):
                            st.session_state[
                                "pending_delete_id"
                            ] = str(document_id)

                    # Deletion confirmation
                    if st.session_state.get(
                        "pending_delete_id"
                    ) == str(document_id):

                        st.warning(
                            f"Are you sure you want to "
                            f"delete {filename}? "
                            "This action cannot be undone."
                        )

                        confirm_col, cancel_col = st.columns(2)

                        with confirm_col:
                            if st.button(
                                "Confirm Delete",
                                key=f"confirm_{document_id}",
                                type="primary",
                            ):
                                try:
                                    api_request(
                                        "DELETE",
                                        f"/api/v1/documents/{document_id}",
                                    )

                                    st.session_state[
                                        "library_messages"
                                    ] = [
                                        (
                                            "success",
                                            f"{filename} deleted "
                                            "successfully.",
                                        )
                                    ]

                                    st.session_state[
                                        "pending_delete_id"
                                    ] = None

                                    st.rerun()

                                except APIError as error:
                                    st.error(str(error))

                        with cancel_col:
                            if st.button(
                                "Cancel",
                                key=f"cancel_{document_id}",
                            ):
                                st.session_state[
                                    "pending_delete_id"
                                ] = None

                                st.rerun()

    except APIError as error:
        st.error(
            f"Could not load your document library: {error}"
        )


# --------------------------------------------------
# Semantic Search
# --------------------------------------------------


elif page == "Semantic Search":

    st.header("Semantic Search")

    st.markdown(
        "Find relevant information across all your indexed documents."
    )

    # Search form
    with st.form("semantic_search_form"):

        question = st.text_input(
            "What would you like to search for?",
            placeholder="e.g. What is BERT's architecture?",
        )

        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=20,
            value=5,
        )

        submitted = st.form_submit_button(
            "Search Documents",
            type="primary",
        )

    # Execute search
    if submitted:

        if not question.strip():
            st.warning("Please enter a search query.")

        else:
            try:
                with st.spinner("Searching your documents..."):

                    response_data = api_request(
                        "POST",
                        "/api/v1/search",
                        json={
                            "question": question.strip(),
                            "top_k": top_k,
                        },
                    )

                    st.session_state["search_response"] = (
                        response_data
                    )

                    st.session_state["search_question"] = (
                        question.strip()
                    )

            except APIError as error:
                st.error(f"Search failed: {error}")

    # Display search results
    response_data = st.session_state.get(
        "search_response"
    )

    if response_data:

        results = extract_search_results(
            response_data
        )

        result_count = (
            response_data.get("total_results", len(results))
            if isinstance(response_data, dict)
            else len(results)
        )

        st.divider()

        st.subheader("Search Results")

        st.caption(
            f"Query: {st.session_state.get('search_question', '')}"
        )

        st.metric(
            "Retrieved Passages",
            result_count,
        )

        if not results:
            st.info(
                "No relevant passages found. "
                "Try a different search query."
            )

        for index, result in enumerate(results):

            if not isinstance(result, dict):
                continue

            # Search API stores filename and page information
            # inside metadata for each retrieved chunk.
            metadata = result.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}

            # Prefer top-level values when available, then fall back
            # to the nested metadata returned by the API.
            filename = document_value(
                result,
                "filename",
                "document_filename",
                "source",
                default=None,
            )
            if filename is None:
                filename = document_value(
                    metadata,
                    "filename",
                    "source",
                    default="Unknown document",
                )

            page_number = document_value(
                result,
                "page_number",
                "page",
                default=None,
            )
            if page_number is None:
                page_number = document_value(
                    metadata,
                    "page_number",
                    "page",
                    default=None,
                )

            content = document_value(
                result,
                "content",
                "chunk_text",
                "text",
                default="No passage content available.",
            )

            score = document_value(
                result,
                "score",
                "similarity_score",
                "similarity",
                default=None,
            )

            chunk_id = document_value(
                result,
                "chunk_id",
                default=None,
            )

            with st.container(border=True):

                st.markdown(
                    f"**{index + 1}. 📄 {filename}**"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.caption(
                        f"Page: {page_number}"
                        if page_number is not None
                        else "Page: Not available"
                    )

                with col2:
                    if isinstance(score, (int, float)):
                        st.caption(
                            f"Similarity: {score:.4f}"
                        )
                    else:
                        st.caption("Similarity: N/A")

                with col3:
                    st.caption(
                        f"Chunk: {chunk_id}"
                        if chunk_id is not None
                        else "Chunk ID: N/A"
                    )

                st.markdown(str(content))

                if metadata:
                    with st.expander("View metadata"):
                        st.json(metadata)



# --------------------------------------------------
# Chat with Documents
# --------------------------------------------------


elif page == "Chat with Documents":

    st.header("Chat with Documents")
    st.markdown(
        "Ask questions about your documents and get answers "
        "with verifiable source citations."
    )

    # ----------------------------
    # Chat settings
    # ----------------------------

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    settings_col, clear_col = st.columns([5, 1])

    with settings_col:
        chat_top_k = st.slider(
            "Retrieved passages per question",
            min_value=1,
            max_value=20,
            value=5,
            key="chat_top_k",
        )

    with clear_col:
        st.write("")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # ----------------------------
    # Display chat history
    # ----------------------------

    if not st.session_state.chat_history:
        st.info(
            "Start a conversation by asking a question "
            "about your indexed documents."
        )

    # ----------------------------
    # Chat input
    # ----------------------------

    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question and question.strip():

        question = question.strip()

        # Add the user's message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        # Call FastAPI
        try:
            with st.spinner(
                "Searching documents and generating your answer..."
            ):
                response_data = api_request(
                    "POST",
                    "/api/v1/ask",
                    json={
                        "question": question,
                        "top_k": chat_top_k,
                    },
                )

            if not isinstance(response_data, dict):
                raise APIError(
                    "The backend returned an unexpected response."
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_data.get(
                    "answer",
                    "The model did not return an answer.",
                ),
                "citations": response_data.get(
                    "citations", []
                ),
                "valid_citations": response_data.get(
                    "valid_citations", []
                ),
                "invalid_citations": response_data.get(
                    "invalid_citations", []
                ),
                "retrieval_count": response_data.get(
                    "retrieval_count", 0
                ),
            })

        except APIError as error:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Could not generate an answer: {error}",
                "error": True,
            })

    # ----------------------------
    # Render conversation
    # ----------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Display citations for assistant responses
            if message["role"] == "assistant":

                if message.get("error"):
                    continue

                citations = message.get("citations", [])
                valid_citations = set(
                    message.get("valid_citations", [])
                )
                invalid_citations = message.get(
                    "invalid_citations", []
                )

                retrieval_count = message.get(
                    "retrieval_count", 0
                )

                st.caption(
                    f"Retrieved {retrieval_count} passages"
                )

                if citations:
                    with st.expander(
                        f"📚 Sources and citations ({len(citations)})"
                    ):

                        for citation in citations:

                            if not isinstance(citation, dict):
                                continue

                            citation_label = citation.get(
                                "citation",
                                "Unknown",
                            )

                            source = citation.get(
                                "source",
                                "Unknown document",
                            )

                            page_number = citation.get(
                                "page_number"
                            )

                            chunk_id = citation.get(
                                "chunk_id"
                            )

                            is_valid = (
                                citation_label in valid_citations
                            )

                            if is_valid:
                                st.markdown(
                                    f"**[{citation_label}] "
                                    f"📄 {source}**"
                                )
                                st.caption("✓ Validated citation")
                            else:
                                st.markdown(
                                    f"**[{citation_label}] "
                                    f"📄 {source}**"
                                )
                                st.caption(
                                    "⚠ Not listed as a valid citation"
                                )

                            details = []

                            if page_number is not None:
                                details.append(
                                    f"Page {page_number}"
                                )

                            if chunk_id:
                                details.append(
                                    f"Chunk: {chunk_id}"
                                )

                            if details:
                                st.caption(" | ".join(details))

                            st.divider()

                else:
                    st.caption(
                        "No source citations were returned."
                    )

                # Show any invalid references detected by the backend
                if invalid_citations:
                    st.warning(
                        "The model produced citation references "
                        "that could not be validated: "
                        + ", ".join(invalid_citations)
                    )
