# CUMBIA_PY
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21926251.svg)](https://doi.org/10.5281/zenodo.21926251)

**Contact:** luis.montejo@upr.edu
---

CUMBIA is a comprehensive analytical tool originally developed to evaluate the monotonic behavior of reinforced concrete (RC) members with circular or rectangular cross-sections. The program performs rigorous moment-curvature analyses and computes the analytical force-displacement response of the member, providing structural engineers and researchers with a clear evaluation of potential deformation limit states. 

To evaluate these performance levels, CUMBIA requires basic input variables defining the cross-section dimensions, member length, applied axial load, reinforcement details, and material properties. The program accounts for complex structural behaviors, including strain penetration, shear deformations, and various buckling mechanics, ultimately outputting detailed moment-curvature responses, structural limit states, and interaction diagrams.

### The Transition to Python (CUMBIA_PY)
To enhance accessibility, computational stability, and integration with modern data science workflows, the original MATLAB algorithms (`CUMBIACIR.m` and `CUMBIARECT.m`) have been entirely refactored into Python (`CUMBIA_CIR.py` and `CUMBIA_RECT.py`). This modernization leverages powerful open-source libraries—such as `numpy`, `pandas`, and `matplotlib`—to deliver a faster, more robust, and visually comprehensive analysis framework. 

### Key Upgrades and Enhancements
The transition to Python includes significant deep theoretical advancements, bug fixes, and cosmetic upgrades that modernize the CUMBIA workflow:

**Advanced Theoretical Mechanics**
*   **Modified Plastic-Hinge Method:** Introduces the Goodnight et al. (2016) modified plastic-hinge method. This approach differentiates itself by decoupling column flexure and strain penetration deformation components. 
*   **Modern Bar Buckling Limits:** Integrates the Goodnight et al. (2015) Strain-Based and Drift-Based buckling models. 
*   **Spiral Yielding Limit State:** The circular module now automatically calculates the compressive strain and associated displacement at the initial yielding of the confinement steel. 
*   **P-Delta Effects:** Introduces a new dedicated toggle to account for P-Delta effects.

**Modernized Workflows and Outputs**
*   **Scaled 2D Cross-Section Plotting:** Automatically generates perfectly scaled 2D visuals of the cross-section.
*   **Automated Reinforcement Generation:** The rectangular module features a smart toggle to auto-generate a perfectly spaced, uniform peripheral reinforcement matrix.
*   **Unified Multi-Page PDF Reports:** Compiles all generated figures and a cleanly formatted text summary table into a single, professional PDF document.
*   **True Excel Export:** Replaces raw text file outputs with native `.xlsx` workbooks.

## Citation

If you use CUMBIA_PY please cite it as:

> Luis Montejo. (2026). LuisMontejo/CUMBIA_PY: CUMBIA_PY [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21926251

**BibTeX:**
```bibtex
@software{montejo_cumbia_py,
  author       = {Luis Montejo},
  title        = {LuisMontejo/CUMBIA\_PY: CUMBIA\_PY},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21926251},
  url          = {https://doi.org/10.5281/zenodo.21926251}
}
