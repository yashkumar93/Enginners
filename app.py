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
    /* Premium styling overrides */
    .stApp {
        background-color: #0f111a;
        color: #e2e4e9;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .metric-card {
        background-color: #1a1d2e;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2e324c;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-inactive {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    pre {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Thread-safe log capturing queues and streams
# ---------------------------------------------------------------------------
class ThreadSafeQueueStream:
    """Redirects writes to a queue to capture logs in real-time."""
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text: str):
        if text:
            self.q.put(text)

    def flush(self):
        pass

# ---------------------------------------------------------------------------
# Thread-safe Task Execution State
# ---------------------------------------------------------------------------
class ExecutionState:
    """Class to hold execution progress status safely across thread boundaries."""
    def __init__(self):
        self.task_statuses = ["Pending"] * 6

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
        "requirements": "",
        "architecture": "",
        "backend": "",
        "frontend": "",
        "qa": "",
        "devops": ""
    }

# ---------------------------------------------------------------------------
# Background Execution Thread
# ---------------------------------------------------------------------------
def run_crew_in_background(user_request: str, log_q: queue.Queue, state: ExecutionState):
    """Executes the CrewAI workflow, capturing stdout and saving artifacts."""
    # Save original stdout
    original_stdout = sys.stdout
    sys.stdout = ThreadSafeQueueStream(log_q)

    try:
        print(f"[SYSTEM] Starting CrewAI execution for request: '{user_request}'")
        
        # Load the crew definition
        crew_path = Path(__file__).resolve().parent / "crew.jsonc"
        crew, _ = load_crew(crew_path)
        
        # Wire callbacks to update statuses in real-time
        def make_callback(task_idx: int, exec_state: ExecutionState):
            def task_callback(output):
                exec_state.task_statuses[task_idx] = "Completed"
                if task_idx + 1 < len(exec_state.task_statuses):
                    exec_state.task_statuses[task_idx + 1] = "Running"
            return task_callback

        for idx, task in enumerate(crew.tasks):
            task.callback = make_callback(idx, state)
        
        # Run kickoff
        inputs = {"user_request": user_request}
        result = crew.kickoff(inputs=inputs)
        
        print("[SYSTEM] Crew execution finished successfully!")
        
        # Save generated task outputs as session-state artifacts
        if len(crew.tasks) >= 6:
            st.session_state.artifacts["requirements"] = str(crew.tasks[0].output.raw)
            st.session_state.artifacts["architecture"] = str(crew.tasks[1].output.raw)
            st.session_state.artifacts["backend"] = str(crew.tasks[2].output.raw)
            st.session_state.artifacts["frontend"] = str(crew.tasks[3].output.raw)
            st.session_state.artifacts["qa"] = str(crew.tasks[4].output.raw)
            st.session_state.artifacts["devops"] = str(crew.tasks[5].output.raw)
            
    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        # Mark remaining active tasks as failed
        for idx, status in enumerate(state.task_statuses):
            if status in ("Running", "Pending"):
                state.task_statuses[idx] = "Failed"
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        st.session_state.is_running = False

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.title("🤖 Multi-Provider LLM Router & AI Engineering Crew")
st.markdown("Automate software development pipelines from requirements gathering to deployment using Groq and Gemini.")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar - Configuration and Key Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Provider Key Status")
    
    # Retrieve key statuses
    pool = get_provider_pool()
    groq_provider = pool.get_provider("groq")
    gemini_provider = pool.get_provider("gemini")
    
    groq_keys = len(groq_provider.keys) if groq_provider else 0
    gemini_keys = len(gemini_provider.keys) if gemini_provider else 0
    
    # Groq status
    if groq_keys > 0:
        st.markdown(f'<span class="status-badge status-active">Groq Active ({groq_keys} Key)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-inactive">Groq Inactive (0 Keys)</span>', unsafe_allow_html=True)
        
    # Gemini status
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    if gemini_keys > 0:
        st.markdown(f'<span class="status-badge status-active">Gemini Active ({gemini_keys} Key)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">Gemini Fallback (0 Keys -> Router will use Groq)</span>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Live LLM Router Metrics dashboard
    st.header("📊 Router Metrics")
    metrics = get_metrics_summary()
    
    # Display cards for metrics
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-header">Total Routed Calls</div>
        <div class="metric-value">{metrics.get("total_requests", 0)}</div>
    </div>
    <div style='margin-top: 10px;'></div>
    <div class="metric-card">
        <div class="metric-header">Average Latency</div>
        <div class="metric-value">{metrics.get("average_latency", 0.0):.2f}s</div>
    </div>
    <div style='margin-top: 10px;'></div>
    <div class="metric-card">
        <div class="metric-header">Active Cooldowns</div>
        <div class="metric-value">{len(metrics.get("active_cooldowns", []))}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Reset Metrics"):
        # Reset local metrics dictionary
        metrics["total_requests"] = 0
        metrics["average_latency"] = 0.0
        metrics["active_cooldowns"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Area Layout
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
        st.subheader("Define Software Requirements")
        user_request = st.text_area(
            "What software application would you like the engineering crew to build?",
            placeholder="Example: Create a task manager dashboard application with a FastAPI backend and a responsive React frontend, including user registration, task categorization, and JWT authentication.",
            height=150,
            disabled=st.session_state.is_running
        )
        
        if st.button("Launch Engineering Crew", disabled=st.session_state.is_running or not user_request.strip()):
            st.session_state.is_running = True
            st.session_state.execution_logs = ""
            st.session_state.log_queue = queue.Queue()
            
            # Reset task execution statuses for a new run
            st.session_state.execution_state.task_statuses = ["Running", "Pending", "Pending", "Pending", "Pending", "Pending"]
            
            # Start background thread
            thread = threading.Thread(
                target=run_crew_in_background, 
                args=(user_request, st.session_state.log_queue, st.session_state.execution_state),
                daemon=True
            )
            thread.start()
            st.success("Engineering crew initialized! Switch to the 'Real-Time Execution Logs' tab to watch them work.")
            
    with col2:
        st.subheader("Crew Pipeline Roadmap")
        
        # Read task statuses directly from the shared execution state
        statuses = st.session_state.execution_state.task_statuses

        # Draw stepping pipeline
        steps = [
            ("1. Project Manager", "Analyze Requirements"),
            ("2. Software Architect", "Design System Architecture"),
            ("3. Backend Engineer", "Implement backend services"),
            ("4. Frontend Engineer", "Develop User Interfaces"),
            ("5. QA Engineer", "Review and Validate Code"),
            ("6. DevOps Engineer", "Configure container deployment")
        ]
        
        for idx, (role, goal) in enumerate(steps):
            status = statuses[idx]
            if status == "Completed":
                badge = '<span class="status-badge status-active">✓ Done</span>'
            elif status == "Running":
                badge = '<span class="status-badge status-warning">⏳ Running</span>'
            elif status == "Failed":
                badge = '<span class="status-badge status-inactive" style="color:#ef4444; background-color:rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.3);">✗ Failed</span>'
            else:
                badge = '<span class="status-badge status-inactive">○ Pending</span>'
                
            st.markdown(f"""
            <div style='background-color:#1e293b; padding:10px; border-radius:8px; margin-bottom:8px; border-left: 4px solid {"#10b981" if status=="Completed" else "#f59e0b" if status=="Running" else "#475569"};'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <strong>{role}</strong>
                    {badge}
                </div>
                <div style='font-size:0.8rem; color:#94a3b8;'>{goal}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Logs ──
with tab_logs:
    st.subheader("Console Output Stream")
    
    # Set up scrolling placeholder
    log_area = st.empty()
    
    # Process log queues dynamically
    if st.session_state.is_running:
        while not st.session_state.log_queue.empty():
            try:
                log_chunk = st.session_state.log_queue.get_nowait()
                st.session_state.execution_logs += log_chunk
            except queue.Empty:
                break
                
        # Render logs
        log_area.code(st.session_state.execution_logs, language="bash")
        
        # Autorefresh container to keep logs streaming
        time.sleep(1)
        st.rerun()
    else:
        if st.session_state.execution_logs:
            log_area.code(st.session_state.execution_logs, language="bash")
        else:
            st.info("No active pipeline execution logs. Start the crew on the 'Workspace & Kickoff' tab.")

# ── Tab 3: Artifacts ──
with tab_artifacts:
    st.subheader("Engineering Deliverables")
    
    if not any(st.session_state.artifacts.values()):
        st.info("No artifacts generated yet. Run the pipeline to see outputs from the engineering agents.")
    else:
        # Display artifacts in tabs
        art_tab1, art_tab2, art_tab3, art_tab4, art_tab5, art_tab6 = st.tabs([
            "📋 Requirements",
            "📐 Architecture",
            "☕ Backend Code",
            "🎨 Frontend Code",
            "🔍 QA Review",
            "🐳 DevOps Config"
        ])
        
        with art_tab1:
            st.markdown(st.session_state.artifacts["requirements"])
            
        with art_tab2:
            st.markdown(st.session_state.artifacts["architecture"])
            
        with art_tab3:
            st.markdown(st.session_state.artifacts["backend"])
            
        with art_tab4:
            st.markdown(st.session_state.artifacts["frontend"])
            
        with art_tab5:
            st.markdown(st.session_state.artifacts["qa"])
            
        with art_tab6:
            st.markdown(st.session_state.artifacts["devops"])
