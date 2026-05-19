# ui.py - MAS-DQA Executive Dashboard (Complete Version)
import streamlit as st
import json
import os
import subprocess
import time

st.set_page_config(
    page_title="MAS-DQA Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ MAS-DQA Executive Dashboard")
st.markdown("*Multi-Agent System for Verifiable Data Quality Governance*")

# ==================== SIDEBAR CONTROLS ====================
st.sidebar.header("⚙️ Pipeline Configuration")

# Data source selector - mapped to main.py --data argument
data_mode = st.sidebar.radio(
    "📊 Data Source",
    options=["synthetic", "real"],
    format_func=lambda x: "🧪 Synthetic Data" if x == "synthetic" else "🚌 Real BusPas Data",
    horizontal=True,
    key="data_mode"
)

# Validator selector - mapped to main.py --validator argument
validator_mode = st.sidebar.radio(
    "🤖 Validator",
    options=["mock", "llm"],
    format_func=lambda x: "🤖 Mock (Rule-based)" if x == "mock" else "🧠 Real LLM",
    horizontal=True,
    key="validator_mode"
)

# Sample size slider - aligned with main.py default
sample_size = st.sidebar.slider(
    "📏 Records to Process",
    min_value=100, max_value=50000, value=300, step=100,
    help="Number of records to process in this run"
)

# API key warning for LLM mode
if validator_mode == "llm" and not os.getenv("LITELLM_API_KEY"):
    st.sidebar.warning("⚠️ Set `LITELLM_API_KEY` env var for LLM mode:\n\n`export LITELLM_API_KEY='sk-...'`")

# Run button
run_clicked = st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# ==================== SESSION STATE ====================
if "running" not in st.session_state:
    st.session_state.running = False
if "results" not in st.session_state:
    st.session_state.results = None
if "logs" not in st.session_state:
    st.session_state.logs = ""

# ==================== MAIN LAYOUT ====================
col1, col2 = st.columns([1.1, 1])

# ──────────────────────────────────────────────────────────────────────────────
# LEFT: Live Output
# ──────────────────────────────────────────────────────────────────────────────
with col1:
    st.subheader("📡 Live Pipeline Output")
    
    if run_clicked and not st.session_state.running:
        st.session_state.running = True
        st.session_state.logs = ""
        
        # Build command matching main.py arguments
        cmd = ["python", "main.py"]
        cmd.extend(["--data", data_mode])
        cmd.extend(["--validator", validator_mode])
        cmd.extend(["--sample", str(sample_size)])
        
        # Show command for transparency
        with st.expander("🔧 Command", expanded=False):
            st.code(" ".join(cmd), language="bash")
        
        # Run process with live output
        scroll = st.container(height=500)
        with scroll:
            log_area = st.empty()
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                bufsize=1
            )
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    st.session_state.logs += line
                    with log_area:
                        st.code(st.session_state.logs, language="text")
            
            process.wait()
            
            if process.returncode == 0:
                st.success("✅ Pipeline completed successfully!")
                # Load results
                results_path = "data/dashboard_results.json"
                if os.path.exists(results_path):
                    with open(results_path, "r") as f:
                        st.session_state.results = json.load(f)
            else:
                st.error(f"❌ Pipeline failed with code {process.returncode}")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            st.session_state.running = False
    
    # Show previous logs if not running
    elif st.session_state.logs and not st.session_state.running:
        with st.container(height=500):
            st.code(st.session_state.logs, language="text")

# ──────────────────────────────────────────────────────────────────────────────
# RIGHT: Results Dashboard - COMPLETE VERSION
# ──────────────────────────────────────────────────────────────────────────────
with col2:
    st.subheader("📊 Phase I REPORT")
    
    if st.session_state.results:
        results = st.session_state.results
        config = results.get("config", {})
        summary = results.get("summary", {})
        routing = results.get("routing", {})
        metrics = results.get("phase1_metrics", {})
        quality = results.get("quality_safety", {})  # NEW SECTION
        insights = results.get("key_insights", [])    # NEW SECTION
        
        # Config summary with emojis
        st.markdown(f"""
**Configuration**  
• Data: {'🧪 Synthetic' if config.get('data_mode') == 'synthetic' else '🚌 Real BusPas'}  
• Validator: {'🤖 Mock' if config.get('validator_mode') == 'mock' else '🧠 Real LLM'}  
• Sample: {config.get('sample_size', 0):,} records
        """)
        
        # Summary metrics
        st.markdown(f"""
**Performance**  
• ✅ Processed: {summary.get('records_processed', 0):,}  
• ⏱️ Latency: {summary.get('avg_latency_ms', 0):.1f} ms/record  
• 🕒 Total: {summary.get('total_time_sec', 0):.1f} sec
        """)
        
        # ── QUALITY & SAFETY METRICS SECTION ─────────────────────────────
        if quality:
            st.markdown("**📈 Quality & Safety Metrics:**")
            trust_pct = quality.get('trust_rate_percent', 0)
            anomaly_pct = quality.get('anomaly_capture_rate_percent', 0)
            quarantined = quality.get('quarantined_count', 0)
            judge_count = quality.get('judge_escalations', 0)
            xai_enabled = quality.get('xai_enabled', False)
            
            st.markdown(f"""
• Trust Rate (Clean Data):  {trust_pct:.1f}% {'✅ PASS (≥75%)' if trust_pct >= 75 else '⚠️ REVIEW'}  
• Anomaly Capture Rate:     {anomaly_pct:.1f}%  
• Quarantined/Blocked:      {quarantined}  
• Escalated to Judge:       {judge_count}  
• XAI Audit Trail:          {'✅ Enabled' if xai_enabled else '❌ Disabled'}
            """)
        
        # ── KEY INSIGHTS FOR LEADERSHIP SECTION ──────────────────────────
        if insights:
            st.markdown("**💡 Key Insights for Leadership:**")
            for insight in insights:
                st.markdown(f"• {insight}")
        # ────────────────────────────────────────────────────────────────
        
        # Routing distribution with progress bars
        st.markdown("**🔀 Routing Distribution:**")
        total = summary.get("records_processed", 1)
        for decision, count in routing.items():
            pct = (count / total) * 100 if total > 0 else 0
            color = "🟢" if decision == "TRUST" else "🟡" if decision == "JUDGE" else "🔴"
            bar = "█" * int(pct / 3)
            st.markdown(f"   {color} **{decision}** {bar} {count} ({pct:.1f}%)")
        
        # Phase I metrics with target indicators
        st.markdown("**📈 Phase I Validation Metrics:**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}", 
                     delta="≥0.85 ✓" if metrics.get('precision', 0) >= 0.85 else "≥0.85 ✗")
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}",
                     delta="≥0.80 ✓" if metrics.get('recall', 0) >= 0.80 else "≥0.80 ✗")
        with col_m2:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}",
                     delta="≥0.82 ✓" if metrics.get('f1_score', 0) >= 0.82 else "≥0.82 ✗",
                     delta_color="normal")
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        
        # Sample decisions (collapsible, with full reasons)
        if results.get("sample_decisions"):
            with st.expander("🔍 Sample Decisions", expanded=True):  # Expanded by default for demos
                for d in results["sample_decisions"][:5]:  # Show first 5 for clarity
                    # Display as formatted text instead of JSON for better readability
                    st.markdown(f"""
**Record:** `{d.get('record_id', 'N/A')}`  
• Speed: {d.get('speed', 'N/A')}  
• Profiler: {d.get('prof_verdict', 'N/A')} (dev={d.get('prof_dev', 'N/A')})  
• Validator: {d.get('val_verdict', 'N/A')}  
• Routing: **{d.get('routing', 'N/A')}**  
• Reason: *{d.get('reason', 'No reason provided')}*  
• Ground Truth: {d.get('ground_truth', 'N/A')}
                    """)
                    st.divider()
        
        # Final status badge
        if results.get("phase1_ready"):
            st.success("🎯 **Phase I Readiness: ✅ READY FOR PHASE I**")
        else:
            st.warning("⚠️ **Phase I Readiness: Needs Tuning**")
            
    else:
        st.info("👈 Run the pipeline to see results here.")
        
        # Quick start guide
        with st.expander("💡 Quick Start Guide", expanded=True):
            st.markdown("""
**4 Demo Modes:**

| Mode | Command | Use Case |
|------|---------|----------|
| 🧪 Synthetic + Mock | `--data synthetic --validator mock` | Fast debugging, no API |
| 🤖 Synthetic + LLM | `--data synthetic --validator llm` | Show LLM reasoning |
| 🚌 Real + Mock | `--data real --validator mock` | Validate pipeline on real data |
| 🚀 Real + LLM | `--data real --validator llm` | Full production demo |

**Before running Real modes:**  
1. Run adapter: `python scripts/test_adapter_flow.py`  
2. Ensure `data/normalised.buspas.ndjson` exists

**For LLM modes:**  
`export LITELLM_API_KEY='sk-your-key-here'`
            """)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("MAS-DQA • Multi-Agent Framework for Verifiable Data Quality Governance • v2.0")