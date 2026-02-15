# Newton–Raphson Load Flow — Flowchart

```mermaid
flowchart TD
    A([START]) --> B["Load IEEE 9-bus data<br>default_system_data()<br>Lines 91-118"]
    B --> C["Build Y-bus matrix<br>build_ybus()<br>Lines 162-183"]
    C --> D["Flat start: V=1.0 pu, d=0<br>Identify SLACK / PV / PQ<br>newton_raphson_pf()<br>Lines 275-296"]
    D --> E["Set iteration counter<br>k = 1<br>Line 298"]
    E --> F["Form complex voltages<br>Vi = |Vi| exp(j di)<br>Line 299"]
    F --> G["Compute P, Q injections<br>power_injections()<br>Lines 187-197"]
    G --> H["Compute scheduled<br>P_spec, Q_spec<br>Lines 302-303"]
    H --> I["Build mismatch vector<br>dP non-slack, dQ PQ<br>Lines 306-311"]
    I --> J{"max mismatch < 1e-4 ?<br>Line 320"}
    J -- Yes --> K["Store final voltages<br>Line 336"]
    J -- No --> L["Assemble Jacobian J1-J4<br>assemble_jacobian()<br>Lines 207-264"]
    L --> M["Solve J . dx = mismatch<br>try_numpy_solve() Lines 53-63<br>gaussian_elimination() Lines 65-87<br>Line 324"]
    M --> N["Update d non-slack<br>Update |V| PQ only<br>Lines 327-331"]
    N --> O["k = k + 1<br>Line 298"]
    O --> F
    K --> P["Compute line flows and losses<br>compute_line_flows()<br>Lines 339-377"]
    P --> Q["Print / Export results<br>Bus voltages, angles<br>Line flows, total loss<br>run_demo()<br>Lines 536-569"]
    Q --> R([END])

    style A fill:#2d6a4f,color:#fff,stroke:#1b4332
    style R fill:#2d6a4f,color:#fff,stroke:#1b4332
    style J fill:#e76f51,color:#fff,stroke:#264653
    style B fill:#264653,color:#fff
    style C fill:#264653,color:#fff
    style D fill:#264653,color:#fff
    style F fill:#2a9d8f,color:#fff
    style G fill:#2a9d8f,color:#fff
    style H fill:#2a9d8f,color:#fff
    style I fill:#2a9d8f,color:#fff
    style L fill:#e9c46a,color:#000
    style M fill:#e9c46a,color:#000
    style N fill:#e9c46a,color:#000
    style K fill:#264653,color:#fff
    style P fill:#264653,color:#fff
    style Q fill:#264653,color:#fff
    style E fill:#6c757d,color:#fff
    style O fill:#6c757d,color:#fff
```