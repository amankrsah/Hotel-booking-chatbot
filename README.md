```markdown
# 🤖 AI-Powered Hotel Booking Chatbot & Virtual Concierge

An enterprise-grade, conversational AI assistant and virtual concierge engineered to automate the hospitality reservation ecosystem. It implements a **hybrid data architecture** combining **Semantic Vector Search (ChromaDB)** and a **Persistent Transaction Engine (SQLite)** to transform rigid booking forms into fluid, natural human interactions while maintaining strict data integrity.

---

## ⚡ Core Features

* **Conversational Reservation Engine (`agent_logic.py`)**
  * **Intent Parsing:** Extracts temporal tokens (check-in/check-out dates), room preferences, budgets, and guest demographics from raw human text.
  * **Semantic Recommendation Engine:** Embeds resort profiles and lifestyle offerings to execute high-fidelity vector searches, matching implicit guest moods (e.g., *"quiet wellness retreat"*) to property parameters.
* **Autonomous Booking Engine & ACID Transaction Safeguards**
  * Evaluates multi-room inventory pools in real time to avoid overbooking conditions.
  * Executes atomic structural mutations securely within a localized database layout, issuing instant confirmation IDs.
* **Contextual State-Machine Snapshotting**
  * Logs interactive historical threads continually across flat-file formats (`.json` / `.csv`).
  * Assures full session memory persistence, ensuring the bot remembers preceding user references throughout the conversation.

---

## 🛠️ Technology Stack

* **Core Runtime:** Python 3.10+
* **NLP & Vector Orchestration:** LangChain / ChromaDB / Local or Cloud LLM API Gateway
* **Relational Database Engine:** SQLite3 (For structured transaction logs and concurrency control)
* **Compute Optimization:** PyTorch / CUDA-optimized verification pipelines (`test_gpu.py`)
* **Interface & Templates:** HTML5 / CSS3 / Web UI layout handlers (`templates/`)

---

## 🔄 System Pipeline Flow

                               ┌──────────────────────────┐
                               │    User / Guest Input    │
                               └────────────┬─────────────┘
                                            │ (Natural Language Query)
                                            ▼
                               ┌──────────────────────────┐
                               │      main.py Server      │
                               └────────────┬─────────────┘
                                            │ (Routing & Session API)
                                            ▼
                               ┌──────────────────────────┐
                               │      agent_logic.py      │
                               │ [Parsing & State Engine] │
                               └──────┬────────────┬──────┘
                                      │            │
            (Semantic Vibe Query)     │            │ (ACID SQL Mutations)
                                      ▼            ▼
                           ┌─────────────┐      ┌─────────────┐
                           │  chroma_db  │      │ bookings.db │
                           │ (Vector DB) │      │ (SQLite DB) │
                           └─────────────┘      └──────┬──────┘
                                                       │
                                                       │ (State Snapshot Sync)
                                                       ▼
                                                ┌─────────────┐
                                                │   booking_  │
                                                │   details   │
                                                │ (.json/.csv)│
                                                └─────────────┘

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
