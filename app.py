"""Streamlit Web Application for the AI Engineering Crew.

Provides an interactive web dashboard to configure keys, monitor the LLM Router,
input software requests, track task execution with real-time streaming logs,
and view generated project artifacts.
"""

import sys
import os
import queue
import threading
import time
import datetime
from pathlib import Path
from typing import Any

# Ensure current directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from crewai.project.crew_loader import load_crew

# Import LLM Router singletons to display metrics
from llm.router import get_metrics_summary, get_provider_pool

# Initialize page config
st.set_page_config(
    page_title="AI Engineering Crew Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Design Aesthetics
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');

    /* Global variables */
    :root {
        --bg-color: #06070d;
        --card-bg: rgba(16, 20, 38, 0.45);
        --border-color: rgba(99, 102, 241, 0.15);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent-indigo: #6366f1;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
    }

    /* Apply premium typography globally, avoiding overriding icon web fonts */
    html, body, .stMarkdown, p, div, button, input, textarea, h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
        background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.04) 0%, transparent 45%),
                          radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.04) 0%, transparent 45%);
    }

    /* Glassmorphic Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(5, 6, 12, 0.95) !important;
        border-right: 1px solid var(--border-color) !important;
        backdrop-filter: blur(12px);
    }

    /* Premium interactive button styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-indigo) 0%, var(--accent-cyan) 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100%;
        letter-spacing: 0.03em;
    }

    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(6, 182, 212, 0.5) !important;
        filter: brightness(1.1);
    }

    .stButton>button:active {
        transform: translateY(-1px) !important;
    }

    .stButton>button:disabled {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        opacity: 0.4 !important;
        color: #64748b !important;
    }

    /* Cyberpunk Metric Cards */
    .metric-card {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 1.25rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, var(--accent-indigo), var(--accent-cyan));
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.15);
    }

    .metric-header {
        font-size: 0.72rem;
        text-transform: uppercase;
        color: var(--text-secondary);
        font-weight: 700;
        letter-spacing: 0.09em;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
    }

    /* Status Badges */
    .status-badge {
        padding: 0.35rem 0.95rem;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .status-active {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.1);
    }

    .status-inactive {
        background-color: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    .status-warning {
        background-color: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.1);
    }

    /* Code styling */
    pre, code {
        font-family: 'Fira Code', monospace !important;
        font-size: 0.9rem !important;
        background-color: #04050a !important;
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
    }

    /* Tabs styling */
    div[data-testid="stTabBar"] {
        background-color: rgba(14, 15, 26, 0.8);
        border-radius: 18px;
        padding: 0.4rem;
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
        backdrop-filter: blur(8px);
    }

    button[data-testid="stMarkdownTab"] {
        border-radius: 12px !important;
        color: var(--text-secondary) !important;
        padding: 0.6rem 1.75rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    button[data-testid="stMarkdownTab"]:hover {
        color: var(--text-primary) !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
    }

    button[data-testid="stMarkdownTab"][aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        color: #60a5fa !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1) !important;
    }

    /* Pipeline step cards */
    .pipeline-step {
        background: linear-gradient(135deg, rgba(20, 24, 46, 0.5) 0%, rgba(10, 12, 25, 0.7) 100%);
        padding: 16px 18px;
        border-radius: 16px;
        margin-bottom: 0px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        border: 1px solid rgba(255, 255, 255, 0.02);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .pipeline-step:hover {
        transform: scale(1.02);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
    }

    /* Text area styling */
    textarea {
        background-color: rgba(10, 12, 22, 0.8) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        color: var(--text-primary) !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    textarea:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2) !important;
    }

    /* Custom alert styling */
    .stAlert {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        background-color: rgba(15, 23, 42, 0.6) !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #06070d;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-cyan);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Thread-safe log capturing
# ---------------------------------------------------------------------------
class ThreadSafeQueueStream:
    """Redirects writes to a thread-safe queue for real-time log capture."""
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text: str):
        if text:
            self.q.put(text)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Thread-safe Execution State (shared between main and background threads)
# ---------------------------------------------------------------------------
class ExecutionState:
    """Holds all mutable execution state that the background thread writes to.

    Streamlit session_state is NOT thread-safe, so the background thread
    writes only to this object, and the main Streamlit thread copies from
    it during each rerun cycle.
    """
    def __init__(self):
        self.task_statuses: list[str] = ["Pending"] * 6
        self.artifacts: dict[str, str] = {
            "requirements": "",
            "architecture": "",
            "backend": "",
            "frontend": "",
            "qa": "",
            "devops": "",
        }
        self.is_finished: bool = False
        self.is_error: bool = False
        self.error_message: str = ""
        self.start_time: float | None = None
        self.end_time: float | None = None


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = ""
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "execution_state" not in st.session_state:
    st.session_state.execution_state = ExecutionState()
if "artifacts" not in st.session_state:
    st.session_state.artifacts = {
        "requirements": "", "architecture": "", "backend": "",
        "frontend": "", "qa": "", "devops": "",
    }
if "run_count" not in st.session_state:
    st.session_state.run_count = 0


# ---------------------------------------------------------------------------
# Background Execution Thread
# ---------------------------------------------------------------------------
def run_crew_in_background(
    user_request: str,
    log_q: queue.Queue,
    state: ExecutionState,
):
    """Execute the CrewAI workflow in a background thread.

    All mutable state writes go to ``state`` (an ExecutionState instance),
    NOT directly to ``st.session_state``, to avoid thread-safety issues.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = ThreadSafeQueueStream(log_q)
    sys.stderr = ThreadSafeQueueStream(log_q)

    state.start_time = time.time()

    try:
        print(f"[SYSTEM] Starting CrewAI execution for request: '{user_request}'")
        print(f"[SYSTEM] Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Load the crew definition
        crew_path = Path(__file__).resolve().parent / "crew.jsonc"
        crew, _ = load_crew(crew_path)

        # Wire callbacks to update statuses in real-time
        def make_callback(task_idx: int, exec_state: ExecutionState):
            def task_callback(output):
                exec_state.task_statuses[task_idx] = "Completed"
                # Save artifact immediately when task completes
                artifact_keys = ["requirements", "architecture", "backend",
                                 "frontend", "qa", "devops"]
                if task_idx < len(artifact_keys) and output is not None:
                    try:
                        raw = output.raw if hasattr(output, "raw") else str(output)
                        exec_state.artifacts[artifact_keys[task_idx]] = str(raw) if raw else ""
                    except Exception:
                        exec_state.artifacts[artifact_keys[task_idx]] = str(output)

                # Mark next task as Running
                if task_idx + 1 < len(exec_state.task_statuses):
                    exec_state.task_statuses[task_idx + 1] = "Running"
            return task_callback

        for idx, task in enumerate(crew.tasks):
            task.callback = make_callback(idx, state)

        # Run kickoff
        inputs = {"user_request": user_request}
        crew.kickoff(inputs=inputs)

        state.end_time = time.time()
        elapsed = state.end_time - state.start_time
        print()
        print(f"[SYSTEM] ✅ Crew execution finished successfully in {elapsed:.1f}s")
        state.is_finished = True

    except Exception as e:
        state.end_time = time.time()
        state.is_error = True
        state.error_message = str(e)
        print(f"\n[ERROR] ❌ Execution failed: {e}")
        # Mark remaining active tasks as failed
        for idx, status in enumerate(state.task_statuses):
            if status in ("Running", "Pending"):
                state.task_statuses[idx] = "Failed"
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# ---------------------------------------------------------------------------
# Sync state from background thread to session_state (called every rerun)
# ---------------------------------------------------------------------------
def sync_execution_state():
    """Copy results from thread-safe ExecutionState into session_state."""
    state = st.session_state.execution_state
    if state.is_finished or state.is_error:
        st.session_state.is_running = False
        # Copy artifacts from ExecutionState to session_state
        for key in state.artifacts:
            if state.artifacts[key]:
                st.session_state.artifacts[key] = state.artifacts[key]

sync_execution_state()


# ---------------------------------------------------------------------------
# Header Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #131525 0%, #080912 100%); padding: 2rem; border-radius: 20px; border: 1px solid #1f243d; margin-bottom: 2rem; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);">
    <div style="display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;">
        <span style="font-size: 3.5rem; filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.3));">🤖</span>
        <div style="flex: 1; min-width: 300px;">
            <h1 style="margin: 0; font-size: 2.3rem; font-weight: 700; background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Outfit', sans-serif;">AI Software Engineering Crew</h1>
            <p style="margin: 0.4rem 0 0 0; color: #94a3b8; font-size: 1.05rem; font-weight: 400; line-height: 1.5; font-family: 'Outfit', sans-serif;">Sequential multi-agent development pipeline powered by an intelligent multi-provider LLM Router (Groq & Gemini key rotation)</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Provider Status & Router Metrics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9; letter-spacing: 0.03em; font-family: 'Outfit', sans-serif;">⚡ Control Panel</div>
    </div>
    """, unsafe_allow_html=True)

    # Retrieve key statuses safely
    pool = get_provider_pool()
    
    groq_keys = 0
    if "groq" in pool.provider_names:
        try:
            groq_provider = pool.get_provider("groq")
            groq_keys = len(groq_provider.keys)
        except KeyError:
            pass

    gemini_keys = 0
    if "gemini" in pool.provider_names:
        try:
            gemini_provider = pool.get_provider("gemini")
            gemini_keys = len(gemini_provider.keys)
        except KeyError:
            pass

    # Provider cards
    providers_data = [
        ("Groq", groq_keys, "🟢" if groq_keys > 0 else "🔴"),
        ("Gemini", gemini_keys, "🟢" if gemini_keys > 0 else "🟡"),
    ]

    for name, key_count, icon in providers_data:
        if key_count > 0:
            status_html = f'<span class="status-badge status-active">● Active ({key_count} Keys)</span>'
        elif name == "Gemini":
            status_html = '<span class="status-badge status-warning">● Fallback (→ Groq)</span>'
        else:
            status_html = '<span class="status-badge status-inactive">● No Keys</span>'

        st.markdown(f"""
        <div style="background-color: #111322; border: 1px solid #1e223b; padding: 12px 16px; border-radius: 12px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'Outfit', sans-serif;">{icon} {name} Pool</span>
            </div>
            {status_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Router Metrics
    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 600; color: #f1f5f9; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">📊 Router Metrics</div>
    """, unsafe_allow_html=True)

    metrics = get_metrics_summary()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">Routed Calls</div>
            <div class="metric-value">{metrics.get("total_requests", 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-header">Avg Latency</div>
            <div class="metric-value">{metrics.get("average_latency_seconds", 0.0):.1f}s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-header">Active Cooldowns</div>
        <div class="metric-value">{len(metrics.get("active_cooldowns", []))}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Execution info in sidebar
    state = st.session_state.execution_state
    if state.start_time and state.end_time:
        elapsed = state.end_time - state.start_time
        st.markdown(f"""
        <div style="background-color: #111322; border: 1px solid #1e223b; padding: 12px 16px; border-radius: 12px;">
            <div class="metric-header">Last Run Duration</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: {'#34d399' if not state.is_error else '#f87171'}; font-family: 'Outfit', sans-serif;">{elapsed:.1f}s {'✅' if not state.is_error else '❌'}</div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.is_running and state.start_time:
        elapsed = time.time() - state.start_time
        st.markdown(f"""
        <div style="background-color: #111322; border: 1px solid #1e223b; padding: 12px 16px; border-radius: 12px;">
            <div class="metric-header">Running For</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #fbbf24; font-family: 'Outfit', sans-serif;">{elapsed:.0f}s ⏳</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Area Tabs
# ---------------------------------------------------------------------------
tab_input, tab_logs, tab_artifacts = st.tabs([
    "📥 Workspace & Kickoff",
    "💻 Real-Time Execution Logs",
    "📂 Generated Artifacts"
])


# ── Tab 1: Input & Run ──
with tab_input:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style="font-size: 1.2rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.8rem; font-family: 'Outfit', sans-serif;">
            💡 Define Your Software Requirements
        </div>
        """, unsafe_allow_html=True)

        if groq_keys == 0 and gemini_keys == 0:
            st.warning(
                "⚠️ **No API Keys Detected**: The application could not find any active API keys in the environment. "
                "If you are deploying on **Streamlit Community Cloud**, please open the app dashboard, click **Manage app** -> **Settings** -> **Secrets**, "
                "and paste your API keys (e.g. `GROQ_API_KEY_1=...`, `GEMINI_API_KEY_1=...`). "
                "If running locally, please verify that they are configured in your `.env` file."
            )

        user_request = st.text_area(
            "Describe the application you want the crew to build:",
            placeholder="Example: Build a task manager dashboard with FastAPI backend and React frontend, including user auth, task categorization, and JWT-based API security.",
            height=160,
            disabled=st.session_state.is_running,
            label_visibility="collapsed",
        )

        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            if st.button(
                "🚀 Launch Engineering Crew",
                disabled=st.session_state.is_running or not user_request.strip(),
                use_container_width=True,
            ):
                st.session_state.is_running = True
                st.session_state.execution_logs = ""
                st.session_state.log_queue = queue.Queue()
                st.session_state.run_count += 1

                # Reset execution state for fresh run
                new_state = ExecutionState()
                new_state.task_statuses = ["Running", "Pending", "Pending", "Pending", "Pending", "Pending"]
                st.session_state.execution_state = new_state
                st.session_state.artifacts = {
                    "requirements": "", "architecture": "", "backend": "",
                    "frontend": "", "qa": "", "devops": "",
                }

                # Start background thread
                thread = threading.Thread(
                    target=run_crew_in_background,
                    args=(user_request, st.session_state.log_queue, new_state),
                    daemon=True,
                )
                thread.start()
                st.rerun()

        with col_btn2:
            if st.button(
                "🗑️ Clear Results",
                disabled=st.session_state.is_running,
                use_container_width=True,
            ):
                st.session_state.execution_state = ExecutionState()
                st.session_state.artifacts = {
                    "requirements": "", "architecture": "", "backend": "",
                    "frontend": "", "qa": "", "devops": "",
                }
                st.session_state.execution_logs = ""
                st.session_state.is_running = False
                st.rerun()

        # Status message
        if st.session_state.is_running:
            st.info("⏳ Engineering crew is working... Switch to the **Real-Time Execution Logs** tab to watch them work.")
        elif st.session_state.execution_state.is_finished:
            st.success("✅ Pipeline completed successfully! View results in the **Generated Artifacts** tab.")
        elif st.session_state.execution_state.is_error:
            st.error(f"❌ Pipeline failed: {st.session_state.execution_state.error_message}")

    with col2:
        st.markdown("""
        <div style="font-size: 1.1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.8rem; font-family: 'Outfit', sans-serif;">
            🔄 Pipeline Roadmap
        </div>
        """, unsafe_allow_html=True)

        statuses = st.session_state.execution_state.task_statuses

        steps = [
            ("Project Manager", "📋", "Analyze Requirements"),
            ("Software Architect", "📐", "Design System Architecture"),
            ("Backend Engineer", "⚙️", "Implement Backend Services"),
            ("Frontend Engineer", "🎨", "Develop User Interfaces"),
            ("QA Engineer", "🔍", "Review & Validate Code"),
            ("DevOps Engineer", "🐳", "Configure Deployment"),
        ]

        for idx, (role, icon, goal) in enumerate(steps):
            status = statuses[idx]

            if status == "Completed":
                badge = '<span class="status-badge status-active">✓ Done</span>'
                border_color = "#10b981"
                bg = "#0d1a14"
            elif status == "Running":
                badge = '<span class="status-badge status-warning">⏳ Running</span>'
                border_color = "#f59e0b"
                bg = "#1a1710"
            elif status == "Failed":
                badge = '<span class="status-badge status-inactive">✗ Failed</span>'
                border_color = "#ef4444"
                bg = "#1a0f0f"
            else:
                badge = '<span style="padding: 0.25rem 0.7rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500; color: #475569; border: 1px solid #1e293b;">○ Pending</span>'
                border_color = "#1e293b"
                bg = "#111827"

            st.markdown(f"""
            <div class="pipeline-step" style="border-left: 4px solid {border_color}; background-color: {bg};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 0.6rem;">
                        <span style="font-size: 1.1rem;">{icon}</span>
                        <strong style="font-size: 0.9rem; color: #e2e8f0; font-family: 'Outfit', sans-serif;">{role}</strong>
                    </div>
                    {badge}
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px; margin-left: 2rem; font-family: 'Outfit', sans-serif;">{goal}</div>
            </div>
            """, unsafe_allow_html=True)

            if idx < 5:
                # Add connector line
                st.markdown("""
                <div style="display: flex; justify-content: center; align-items: center; margin: -5px 0 -5px 0;">
                    <div style="width: 2px; height: 18px; background: linear-gradient(to bottom, #1e293b, rgba(99, 102, 241, 0.3));"></div>
                </div>
                """, unsafe_allow_html=True)

        # Progress bar
        completed = sum(1 for s in statuses if s == "Completed")
        progress = completed / len(statuses)
        st.progress(progress, text=f"{completed}/{len(statuses)} agents completed")


# ── Tab 2: Logs ──
with tab_logs:
    st.markdown("""
    <div style="font-size: 1.1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.8rem; font-family: 'Outfit', sans-serif;">
        📟 Console Output Stream
    </div>
    """, unsafe_allow_html=True)

    log_area = st.empty()

    # Drain the log queue into execution_logs
    drained = False
    while not st.session_state.log_queue.empty():
        try:
            log_chunk = st.session_state.log_queue.get_nowait()
            st.session_state.execution_logs += log_chunk
            drained = True
        except queue.Empty:
            break

    # Render logs
    if st.session_state.execution_logs:
        log_area.code(st.session_state.execution_logs, language="bash")
    else:
        log_area.info("No execution logs yet. Start the crew on the **Workspace & Kickoff** tab.")

    # Auto-refresh while running (non-blocking via st.rerun with small delay)
    if st.session_state.is_running:
        time.sleep(1.5)
        st.rerun()


# ── Tab 3: Artifacts ──
with tab_artifacts:
    st.markdown("""
    <div style="font-size: 1.1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.8rem; font-family: 'Outfit', sans-serif;">
        📦 Engineering Deliverables
    </div>
    """, unsafe_allow_html=True)

    if not any(st.session_state.artifacts.values()):
        st.info("No artifacts generated yet. Run the pipeline to see outputs from the engineering agents.")
    else:
        art_tabs = st.tabs([
            "📋 Requirements",
            "📐 Architecture",
            "⚙️ Backend Code",
            "🎨 Frontend Code",
            "🔍 QA Review",
            "🐳 DevOps Config",
        ])

        artifact_keys = ["requirements", "architecture", "backend",
                         "frontend", "qa", "devops"]

        for tab, key in zip(art_tabs, artifact_keys):
            with tab:
                content = st.session_state.artifacts.get(key, "")
                if content:
                    st.markdown(content)
                else:
                    st.markdown(
                        f"<div style='color: #64748b; padding: 2rem; text-align: center; font-family: Outfit, sans-serif;'>"
                        f"⏳ This agent hasn't completed yet...</div>",
                        unsafe_allow_html=True,
                    )
