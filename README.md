# 🤖 Barnawapara Hotel Booking chatbot : AI-Powered Hotel Booking Chatbot & Virtual Assistant

An enterprise-grade, conversational AI assistant and virtual concierge engineered to automate the hospitality reservation ecosystem. It implements a **hybrid data validation architecture** combining real-time **Semantic Vector Search (ChromaDB)** and a **Persistent Transaction Engine (SQLite)** to eliminate rigid booking forms entirely while maintaining strict ACID data integrity.

---

## ⚡ Core Features

* **Conversational Reservation Engine (`agent_logic.py`)**
  * **Intent Parsing:** Dynamically isolates temporal tokens (check-in/check-out dates), room tier preferences, budgets, and guest demographics directly from raw human text payloads.
  * **Semantic Recommendation Engine:** Embeds granular resort profiles and lifestyle configurations to execute high-fidelity vector searches via **ChromaDB**, matching implicit guest moods (e.g., *"quiet wellness retreat"*) to property parameters[cite: 3].
* **Autonomous Booking Engine & ACID Transaction Safeguards**[cite: 3]
  * Evaluates multi-room inventory pools and live availability matrices in real time to actively prevent race conditions or overbooking anomalies[cite: 3].
  * Executes atomic structural mutations securely within a localized database layout, instantly provisioning immutable reservation tokens and confirmation IDs[cite: 3].
* **Contextual State-Machine Snapshotting**[cite: 3]
  * Background memory threads continuously stream interactive historical logs across flat-file storage targets (`.json` / `.csv`)[cite: 3].
  * Assures absolute multi-turn conversational session persistence, ensuring contextual state properties are retained across the entire user journey[cite: 3].

---

## 🛠️ Technology Stack

* **Backend Engine:** Python 3.10+, LangChain Orchestrator, ChromaDB (Vector Index), SQLite3 (ACID Relational Core)[cite: 3].
* **Compute Optimization & UI:** PyTorch / CUDA hardware acceleration middleware (`test_gpu.py`), Native HTML5/CSS3 Jinja templates (`templates/`)[cite: 3].

---

## 📂 Architecture at a Glance

```text
├── chroma_db/               # Persistent Vector DB instances for semantic knowledge management
├── templates/               # UI/UX structural layout files for web view interfaces
├── agent_logic.py           # Core NLP pipeline, conversational state engine, and LLM middleware
├── main.py                  # Entrypoint server script orchestrating routing and API loops
├── bookings.db              # Relational production database handling active reservation states
├── booking_details.json     # Continuous state-machine logs tracking contextual user journeys
├── booking_details.csv      # Flat-file transactional exports optimized for downstream analytics
├── resorts.json / .csv      # Knowledge matrices housing property attributes, tiers, and pricing
├── requirements.txt         # Immutable list of deterministic system dependencies
└── test_gpu.py              # Computational hardware diagnostics tool for local acceleration paths
