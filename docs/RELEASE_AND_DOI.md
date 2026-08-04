# Release & DOI archiving guide

Open-Sym-EPR is set up for a **DOI-linked release archive** of the exact manuscript
version, so the code that produced the paper can be cited permanently.

## What is already in place

- `CITATION.cff` — software + preferred-citation metadata, with a DOI field.
- `.zenodo.json` — Zenodo deposition metadata (title, authors, license, keywords).
- Versioned release: **v0.2.0** (general isotropic + anisotropic engine).

## One-time setup (links GitHub → Zenodo)

1. Sign in at <https://zenodo.org> with your GitHub account.
2. Go to **Zenodo → Account → GitHub**, find `qleapai/Open-Sym-EPR`, and toggle it **On**.
   (Zenodo can archive private repos once you grant access.)

## Mint a DOI for the manuscript version

1. Tag and push the release:
   ```bash
   git tag -a v0.2.0 -m "Open-Sym-EPR v0.2.0 — general cw-EPR engine (manuscript version)"
   git push origin v0.2.0
   ```
2. On GitHub: **Releases → Draft a new release → choose tag v0.2.0 → Publish**.
3. Zenodo automatically archives the tagged snapshot and mints a DOI.
4. Copy the **concept DOI** (always points to the latest version) into:
   - `CITATION.cff` → `identifiers: value`
   - `.zenodo.json` (optional)
   - the paper's *Software Availability* section.
5. Commit the DOI update and add the Zenodo DOI badge to `README.md`:
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

## Versioning policy

- **Patch** (0.2.x): bug fixes, no API change.
- **Minor** (0.x.0): new components/engines/features, backward compatible.
- **Major** (x.0.0): breaking changes to the model schema or file formats.

Each tagged release gets its own version DOI; the concept DOI resolves to the
newest. Always cite the concept DOI in papers unless a specific version is required.
