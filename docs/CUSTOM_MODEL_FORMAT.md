# Open-Sym-EPR Custom Model Upload Format

Open-Sym-EPR can import custom model libraries from JSON, YAML, or CSV files in the Model builder tab.

## CSV

Required column:

- `name`

Useful optional columns:

- `component_id`
- `assignment`
- `category`
- `g`
- `g_min`
- `g_max`
- `nuclei`
- `linewidth_mT`
- `lw_min`
- `lw_max`
- `eta`
- `weight`
- `spin_S`
- `gx`
- `gy`
- `gz`
- `D_MHz`
- `E_MHz`
- `mode`

Nuclei syntax uses semicolon-separated isotope/coupling pairs in mT:

```text
14N:1.55; 1H:0.30; 63Cu:8.5
```

## JSON / YAML

JSON and YAML model packs can contain:

```yaml
schema: Open-Sym-EPR.user_model
name: my_model
components:
  - component_id: custom_radical
    display_name: Custom radical
    radical_assignment: literature assignment
    category: uploaded
    g: 2.003
    g_bounds: [1.99, 2.02]
    linewidth_mT: 0.15
    linewidth_bounds: [0.02, 2.0]
    eta: 0.5
    weight: 0.3
    nuclei:
      - isotope: 14N
        A_mT: 1.55
        label: N
```

For anisotropic or powder components, add:

```yaml
spin_S: 0.5
g_tensor: [2.06, 2.06, 2.27]
D_MHz: 0.0
E_MHz: 0.0
mode: powder
```

Chemical assignments from uploaded models are treated as candidate models, not proof.
