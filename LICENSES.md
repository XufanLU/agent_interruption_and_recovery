# Licenses and attribution

This file collects license-related notices for this repository: experimental fixture data attribution and third-party Python packages used by the A/B evaluation.

## Data and asset attribution

The experimental fixture data used by the A/B evaluation is re-packaged in
`ab_test/fixtures/`. Per-pair source details, URLs, material IDs, DOIs where
available, and license notes are recorded in
`ab_test/fixtures/xas_cif_pairs.json` (`schema_version`: 1).

- **XAS spectra:** sourced from the Japanese National Institute for Materials
  Science Materials Data Repository (NIMS MDR), available at
  https://mdr.nims.go.jp/. The fixture metadata lists the specific NIMS MDR
  dataset URL for each spectrum. The license recorded for these spectra is
  Creative Commons Attribution 4.0 International (CC-BY 4.0) where indicated by
  the NIMS MDR dataset metadata.

- **CIF/structure files:** sourced from the Materials Project entries identified by
  their `mp-*` material IDs and URLs in the fixture metadata. Materials Project is powered by open-source software; its data are licensed under
  a Creative Commons Attribution 4.0 International License (CC BY 4.0); and
  contributed data are owned by the respective contributors.
No additional scraped web data is used for the experiment fixtures. The
re-packaged fixture files are included only to support reproducibility of the
paper's checkpoint-interruption experiments; users should also follow the
original source licenses and terms for NIMS MDR and Materials Project when
reusing the data.

## Third-party Python packages

Third-party Python packages used by the A/B evaluation are declared in
`ab_test/requirements.txt`.

The A/B evaluation depends on those packages as installed from their upstream
distributions; they are not re-licensed by this repository. Keep the original
license notices from each package when redistributing an environment or derived
bundle.

This project depends on the following open-source packages. License labels are
typical SPDX-style identifiers from upstream packaging metadata; confirm against
each repository before publication or redistribution.

| Package | License |
| --- | --- |
| `lmfit` | BSD-3-Clause |
| `matplotlib` | Matplotlib License (PSF-based) |
| `mp-api` | BSD-3-Clause-LBNL | 
| `numpy` | BSD-3-Clause |
| `openai` | Apache-2.0 |
| `openai-agents` | MIT |
| `pandas` | BSD-3-Clause |
| `plotly` | MIT |
| `pydantic` | MIT |
| `pydash` | MIT |
| `pymatgen` | MIT |
| `pytest` | MIT |
| `python-dotenv` | BSD-3-Clause |
| `requests` | Apache-2.0 |
| `scipy` | BSD-3-Clause |
| `uncertainties` | BSD-3-Clause |
| `xraydb` | MIT |
| `xraylarch` | MIT |

See the respective project repositories for full license texts.

