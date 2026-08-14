'''
Montejo, L. (2026). LuisMontejo/CUMBIA_PY: CUMBIA_PY [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21926251
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import material_models as mm  
from matplotlib.backends.backend_pdf import PdfPages
plt.close('all')

# ==============================================================================
# INPUT DATA: 
# ==============================================================================

# ------------------------------------------------------------------------------
# General Analysis Controls
# ------------------------------------------------------------------------------
name = 'CUMBIACIR_example'  # Prefix for output files (e.g., CUMBIACIREX_DB_PY_Results.xlsx)
interaction = 'y'           # Perform axial load-moment (M-P) interaction analysis? ('y' or 'n')

# ------------------------------------------------------------------------------
# Section Properties
# ------------------------------------------------------------------------------
D = 1000                    # Section diameter [mm]
clb = 50                    # Clear cover to longitudinal bars [mm]

# ------------------------------------------------------------------------------
# Member Properties
# ------------------------------------------------------------------------------
L = 3000                    # Member clear length [mm]
bending = 'single'          # Bending configuration: 'single' (cantilever) or 'double' (fixed-fixed)
ductilitymode = 'biaxial'   # Ductility loading mode: 'uniaxial' or 'biaxial' (Alters UCSD shear degradation factor)
p_delta = 'y'               # Include P-Delta effects? ('y' or 'n') approximated by subtracting the geometric 
                            # overturning moment from the baseline force capacity using the total displacement
hinge_method = 'pck'        # Options: 'pck' (Priestley et al 2007) or 'modified_lpr' (Goodnight et al 2016)   
# ------------------------------------------------------------------------------
# Material Models Selection
# ------------------------------------------------------------------------------
# Constitutive models for concrete and steel. Select 'mc' (Mander confined), 
# 'mu' (Mander unconfined), 'mclw' / 'mulw' (lightweight), 'ra' (Raynor), or 'ks' (King).
confined = 'mc'             # Confined concrete model 
unconfined = 'mu'           # Unconfined (cover) concrete model 
rebar = 'ra'                # Reinforcing steel model 

# ------------------------------------------------------------------------------
# Reinforcement Details
# ------------------------------------------------------------------------------
nbl = 22                    # Number of longitudinal bars
Dbl = 25                    # Longitudinal bar diameter [mm]
Dh = 9                      # Diameter of transverse reinforcement (hoops/spirals) [mm]
type_reinf = 'spirals'      # Type of transverse reinforcement: 'spirals' or 'hoops'
s = 120                     # Spacing of transverse steel [mm]

# ------------------------------------------------------------------------------
# Applied Loads
# ------------------------------------------------------------------------------
P_kN = 2000                  # Applied axial load [kN] (+ for compression, - for tension)

# ------------------------------------------------------------------------------
# Concrete Material Properties
# ------------------------------------------------------------------------------
fpc = 35                    # Concrete compressive strength (f'c) [MPa]
Ec = 5000 * (fpc**(0.5))    # Concrete modulus of elasticity [MPa] (Default: 5000*sqrt(f'c))
eco = 0.002                 # Unconfined concrete strain at peak stress (0.002 for normal wt, 0.004 for light wt)
esm = 0.11                  # Maximum transverse steel strain (usually 0.10 - 0.15)
espall = 0.0064             # Maximum unconfined concrete strain (spalling strain)

# ------------------------------------------------------------------------------
# Steel Material Properties
# ------------------------------------------------------------------------------
fy = 460                    # Longitudinal steel yielding stress (fy) [MPa]
fyh = 400                   # Transverse steel yielding stress (fyh) [MPa]
Es = 200000                 # Steel modulus of elasticity [MPa]
fsu = 620                   # Longitudinal steel ultimate stress [MPa]
esh = 0.008                 # Longitudinal steel strain at onset of strain hardening
esu = 0.12                  # Longitudinal steel ultimate strain

# Raynor Model specific parameters
Ey = 350                    # Slope of the yield plateau [MPa] 
C1 = 3.5                    # Parameter defining the curvature of the strain hardening curve 

# ------------------------------------------------------------------------------
# Strain Limits for Limit States & Yield Surface
# ------------------------------------------------------------------------------
csid = 0.004                # Concrete strain limit for yield surface (interaction diagram)
ssid = 0.015                # Steel strain limit for yield surface (interaction diagram)
ecser = 0.004               # Concrete serviceability compressive strain limit
esser = 0.015               # Steel serviceability tensile strain limit
ecdam = 'twth'              # Concrete damage control strain ('twth' for 2/3 of ultimate, or numeric)
esdam = 0.060               # Steel damage control tensile strain limit

# ------------------------------------------------------------------------------
# Environmental & Phenomenological Parameters
# ------------------------------------------------------------------------------
temp = 40                   # Temperature of the specimen in Celsius (affects tensile strength if < 0)
kLsp = 0.022                # Strain penetration constant (0.022 at ambient temp, 0.011 at -40C)

# ------------------------------------------------------------------------------
# Numerical Control Parameters
# ------------------------------------------------------------------------------
itermax = 1000              # Maximum number of iterations for neutral axis search
ncl = 40                    # Number of concrete layer discretizations
tolerance = 0.001           # Force equilibrium tolerance multiplier (tol = tolerance * area * f'c)
dels = 0.0001               # Strain step increment (delta strain) for section analysis

# ------------------------------------------------------------------------------
# Derived Geometric Variables
# ------------------------------------------------------------------------------
Dsp = D - 2*clb + Dh        # Core diameter (center-to-center of transverse reinforcement) [mm]
dcore = clb - Dh*0.5        # Distance from extreme fiber to center of transverse reinf [mm]
Ast = nbl * 0.25 * np.pi * (Dbl**2) # Total area of longitudinal steel [mm^2]

P = P_kN * 1000             # Convert axial load to base units [N] for internal calculations

# ==============================================================================
# GLOBAL PLOT STYLING 
# ==============================================================================
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 11

import matplotlib.patches as patches

def plot_circular_section(D, clb, Dh, Dbl, nbl, yl):
    """Generates a scaled 2D plot of the discretized circular column section."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    R = D / 2.0
    Dsp = D - 2*clb + Dh
    Rcore = Dsp / 2.0
    Rbar = D / 2.0 - clb - Dbl / 2.0
    
    # 1. Unconfined Concrete (Outer Circle)
    outer_circ = patches.Circle((0, 0), R, color='silver', alpha=0.3, ec='black', zorder=1, label='Unconfined Concrete')
    ax.add_patch(outer_circ)
    
    # 2. Confined Concrete (Inner Circle)
    inner_circ = patches.Circle((0, 0), Rcore, color='cornflowerblue', alpha=0.6, ec='none', zorder=2, label='Confined Concrete')
    ax.add_patch(inner_circ)
    
    # 3. Spiral
    spiral_circ = patches.Circle((0, 0), Rcore, fill=False, ec='firebrick', linestyle='--', linewidth=1.5, zorder=3, label='Spiral/Hoop')
    ax.add_patch(spiral_circ)
    
    # 4. Discretization Layers
    # yl represents depths from the extreme compression fiber (top, y = R)
    for i, depth in enumerate(yl):
        y_line = R - depth
        # Calculate intersection with the outer boundary
        x_val = np.sqrt(max(0, R**2 - y_line**2))
        lbl = 'Discretized Layers' if i == 0 else ""
        ax.plot([-x_val, x_val], [y_line, y_line], color='black', linewidth=0.5, alpha=0.5, zorder=4, label=lbl)
            
    # 5. Longitudinal Bars
    theta = np.linspace(0, 2*np.pi, nbl, endpoint=False)
    x_bars = Rbar * np.sin(theta)
    y_bars = Rbar * np.cos(theta)
    
    for i, (xb, yb) in enumerate(zip(x_bars, y_bars)):
        lbl = 'Longitudinal Bars' if i == 0 else ""
        bar = patches.Circle((xb, yb), Dbl/2.0, color='black', zorder=5, label=lbl)
        ax.add_patch(bar)
        
    ax.set_aspect('equal')
    ax.set_xlim(-R * 1.15, R * 1.15)
    ax.set_ylim(-R * 1.15, R * 1.15)
    ax.set_xlabel('Width (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('Discretized Circular Section', fontweight='bold', pad=15)

    ax.legend(
    loc='upper center',          # Anchors the top-center of the legend box...
    bbox_to_anchor=(0.5, -0.15), # ...to the bottom-center of the plot (x=0.5, y=-0.15)
    ncol=2,                      # Forces the items into exactly 2 columns
    frameon=True                 # Optional: keeps a border box around it
    )
    
    fig.tight_layout()
    
    return fig
# ==============================================================================
# INPUT VALIDATION
# ==============================================================================
if interaction.lower() not in ['y', 'n']:
    raise ValueError("Error: 'interaction' must be specified as 'y' or 'n'.")
if p_delta.lower() not in ['y', 'n']:
    raise ValueError("Error: 'p_delta' must be specified as 'y' or 'n'.")  
if bending.lower() not in ['single', 'double']:
    raise ValueError("Error: 'bending' must be specified as 'single' or 'double'.")
if ductilitymode.lower() not in ['uniaxial', 'biaxial']:
    raise ValueError("Error: 'ductilitymode' must be specified as 'uniaxial' or 'biaxial'.")
if type_reinf.lower() not in ['spirals', 'hoops']:
    raise ValueError("Error: 'type_reinf' must be specified as 'spirals' or 'hoops'.")
if confined.lower() not in ['mc', 'mu', 'mclw']:
    raise ValueError("Error: 'confined' must be 'mc', 'mu', or 'mclw'.")
if unconfined.lower() not in ['mc', 'mu', 'mclw', 'mulw']:
    raise ValueError("Error: 'unconfined' must be 'mc', 'mu', 'mclw', or 'mulw'.")
if rebar.lower() not in ['ks', 'ra']:
    raise ValueError("Error: 'rebar' must be 'ks' or 'ra'.")
if hinge_method.lower() not in ['pck', 'modified_lpr']:
    raise ValueError("Error: 'hinge_method' must be specified as 'pck' or 'modified_lpr'.")
    
if temp < 0:
    Ct = (1 - 0.0105*temp) * 0.56 * (fpc**(0.5))
else:
    Ct = 0.56 * (fpc**(0.5))
eccr = Ct / Ec

# ==============================================================================
# LIGHTWEIGHT CONCRETE SAFETY OVERRIDE
# ==============================================================================
if 'lw' in confined.lower() or 'lw' in unconfined.lower():
    eco = 0.004

# ==============================================================================
# MATERIAL MODELS & PLOTS 1 & 2
# ==============================================================================
if unconfined.lower() == 'mu':
    ecun, fcun = mm.manderun(Ec, Ast, Dh, clb, s, fpc, fyh, eco, esm, espall, 'circular', D, 0, 0, 0, 0, 0, dels)
elif unconfined.lower() == 'mclw':
    ecun, fcun = mm.manderconflw(Ec, Ast, Dh, clb, s, fpc, fyh, eco, esm, espall, 'circular', D, 0, 0, 0, 0, 0, dels, type_reinf)
elif unconfined.lower() == 'mulw':
    ecun, fcun = mm.manderunlw(Ec, nbl, Dbl, Dh, clb, s, fpc, fyh, eco, esm, espall, 'circular', D, 0, 0, 0, 0, 0, dels)
else:
    ecun, fcun = mm.manderconf(Ec, Ast, Dh, clb, s, fpc, fyh, eco, esm, espall, 'circular', D, 0, 0, 0, 0, 0, dels, type_reinf)

if confined.lower() == 'mc':
    ec, fc = mm.manderconf(Ec, Ast, Dh, clb, s, fpc, fyh, eco, esm, espall, 'circular', D, 0, 0, 0, 0, 0, dels, type_reinf)
elif confined.lower() == 'mclw':
    ec, fc = mm.manderconflw(Ec, Ast, Dh, clb, s, fpc, fyh, eco, esm, espall, 'circular', D, 0, 0, 0, 0, 0, dels, type_reinf)
else:
    ec, fc = mm.manderun(Ec, Ast, Dh, clb, s, fpc, fyh, eco, esm, espall, 'circular', D, 0, 0, 0, 0, 0, dels)
    
if rebar.lower() == 'ra':
    es_steel, fs_steel = mm.Raynor(Es, fy, fsu, esh, esu, dels, C1, Ey)
else:
    es_steel, fs_steel = mm.steelking(Es, fy, fsu, esh, esu, dels)

ecu = ec[-1]
ecumander = ecu / 1.5

if str(ecdam).lower() == 'twth':
    ecdam = ecumander
else:
    ecdam = float(ecdam)

ec_full = np.concatenate(([-1e10], ec, [ec[-1] + dels, 1e10]))
fc_full = np.concatenate(([0], fc, [0, 0]))
ecun_full = np.concatenate(([-1e10], ecun, [ecun[-1] + dels, 1e10]))
fcun_full = np.concatenate(([0], fcun, [0, 0]))

esaux = es_steel[::-1]
fsaux = fs_steel[::-1]
es_steel_full = np.concatenate((-esaux, es_steel[1:]))
fs_steel_full = np.concatenate((-fsaux, fs_steel[1:]))

# Plot 1: Concrete Stress-Strain
fig1, ax1_p1 = plt.subplots(figsize=(6.5, 4.5))
ax1_p1.fill_between(ec, fc, color='cornflowerblue', alpha=0.8, label='Confined Concrete')
ecun_plot = np.concatenate((ecun, [ecu]))
fcun_plot = np.concatenate((fcun, [0]))
ax1_p1.fill_between(ecun_plot, fcun_plot, color='silver', alpha=0.8, label='Unconfined Concrete')
ax1_p1.grid(True, linestyle=':', alpha=0.7)
ax1_p1.set_xlabel('Strain')
ax1_p1.set_ylabel('Stress (MPa)')
ax1_p1.set_title('Stress-Strain Relation for Confined and Unconfined Concrete', fontweight='bold')
ax1_p1.legend()
ax1_p1.set_xlim(0, np.ceil(ecu*1000)/1000 + 0.001)
ax1_p1.set_ylim(0, 1.05 * np.max(fc))
fig1.tight_layout()

# Plot 2: Steel Stress-Strain
fig2, ax1_p2 = plt.subplots(figsize=(6.5, 4.5))
ax1_p2.fill_between(es_steel_full, fs_steel_full, color='silver', alpha=0.8)
ax1_p2.grid(True, linestyle=':', alpha=0.7)
ax1_p2.set_xlabel('Strain')
ax1_p2.set_ylabel('Stress (MPa)')
ax1_p2.set_title('Stress-Strain Relation for Reinforcing Steel', fontweight='bold')
ax1_p2.set_xlim(-0.12, 0.12)
ax1_p2.set_ylim(-1.05 * fsu, 1.05 * fsu)
fig2.tight_layout()

# ==============================================================================
# SECTION ANALYSIS (MOMENT-CURVATURE)
# ==============================================================================
tcl = D / ncl
yl = np.arange(1, ncl + 1) * tcl
yl = np.sort(np.unique(np.concatenate((yl, [dcore, D - dcore]))))
yc = yl - dcore
yc = np.append(yc[(yc > 0) & (yc < Dsp)], Dsp)

Aaux_tc = ((D/2)**2) * np.arccos(1 - 2*yl/D) - (D/2 - yl) * np.sqrt(np.maximum(0, D*yl - yl**2))
Atc = np.append(Aaux_tc[0], np.diff(Aaux_tc))

Aaux_tcc = ((Dsp/2)**2) * np.arccos(1 - 2*yc/Dsp) - (Dsp/2 - yc) * np.sqrt(np.maximum(0, Dsp*yc - yc**2))
Atcc = np.append(Aaux_tcc[0], np.diff(Aaux_tcc))

conclay = np.zeros((len(yl), 5))
conclay[:, 0] = np.concatenate(([yl[0]/2], 0.5 * (yl[:-1] + yl[1:])))
conclay[:, 3] = yl

k = 0
for i in range(len(yl)):
    if yl[i] <= dcore or yl[i] > D - dcore:
        conclay[i, 1] = Atc[i]
        conclay[i, 2] = 0
    else:
        conclay[i, 1] = Atc[i] - Atcc[k]
        conclay[i, 2] = Atcc[k]
        k += 1

Asb = 0.25 * np.pi * (Dbl**2)
r_bar = 0.5 * (D - 2*clb - Dbl)
theta = (2 * np.pi / nbl) * np.arange(nbl)
distld = np.sort(0.5 * (D - 2*r_bar) + r_bar * np.sin(theta) * np.tan(0.5 * theta))
Asbs = Asb * np.ones(nbl)

for k in range(1, len(conclay) - 1):
    conclay[k, 4] = np.sum(Asbs[(distld <= conclay[k, 3]) & (distld > conclay[k-1, 3])])
    if conclay[k, 2] == 0:
        conclay[k, 1] -= conclay[k, 4]
    else:
        conclay[k, 2] -= conclay[k, 4]


# Create Visual Plot of the Discretized Section
fig_section = plot_circular_section(D, clb, Dh, Dbl, nbl, yl)

eps = 1e-6
if ecu <= 0.0018:
    def_strain = np.arange(0.0001, 20*ecu + eps, 0.0001)
elif 0.0018 < ecu <= 0.0025:
    def_strain = np.concatenate((np.arange(0.0001, 0.0016 + eps, 0.0001), np.arange(0.0018, 20*ecu + eps, 0.0002)))
elif 0.0025 < ecu <= 0.006:
    def_strain = np.concatenate((np.arange(0.0001, 0.0016 + eps, 0.0001), np.arange(0.0018, 0.002 + eps, 0.0002), np.arange(0.0025, 20*ecu + eps, 0.0005)))
elif 0.006 < ecu <= 0.012:
    def_strain = np.concatenate((np.arange(0.0001, 0.0016 + eps, 0.0001), np.arange(0.0018, 0.002 + eps, 0.0002), np.arange(0.0025, 0.005 + eps, 0.0005), np.arange(0.006, 20*ecu + eps, 0.001)))
else:
    def_strain = np.concatenate((np.arange(0.0001, 0.0016 + eps, 0.0001), np.arange(0.0018, 0.002 + eps, 0.0002), np.arange(0.0025, 0.005 + eps, 0.0005), np.arange(0.006, 0.01 + eps, 0.001), np.arange(0.012, 20*ecu + eps, 0.002)))

# Filter out small strains that cannot mathematically support heavy 
# axial compression, preventing solver drift.
if P > 0:
    valid_strains = []
    for d_strain in def_strain:
        fcunconf_pre = np.interp(d_strain, ecun_full, fcun_full, right=0)
        fcconf_pre = np.interp(d_strain, ec_full, fc_full, right=0)
        fsteel_pre = np.interp(d_strain, es_steel_full, fs_steel_full)

        # Note: Circular uses A_bars instead of Asbs
        compch = (np.sum(fcunconf_pre * conclay[:, 1]) +
                  np.sum(fcconf_pre * conclay[:, 2]) +
                  np.sum(Asbs * fsteel_pre))
        
        if compch >= P:
            valid_strains.append(d_strain)
            
    def_strain = np.array(valid_strains)

message = 0
curv, mom, ejen = [0], [0], [0]
coverstrain, corestrain, steelstrain = [0], [0], [0]
tol = tolerance * 0.25 * np.pi * (D**2) * fpc
x = D / 2

for k in range(len(def_strain)):
    if len(mom) > 1 and mom[-1] < 0.8 * max(mom):
        message = 4
        break
    
    message = 0
    F = 10 * tol
    niter = 0
    current_def = def_strain[k]
    
    while abs(F) > tol:
        niter += 1
        eec = (current_def / x) * (conclay[:, 0] - (D - x))
        ees = (current_def / x) * (distld - (D - x))
        
        fcunconf = np.interp(eec, ecun_full, fcun_full, right=0)
        fcconf = np.interp(eec, ec_full, fc_full, right=0)
        fsteel = np.interp(ees, es_steel_full, fs_steel_full)
        
        FUNCON = fcunconf * conclay[:, 1]
        FCONF = fcconf * conclay[:, 2]
        FST = Asbs * fsteel
        
        F = np.sum(FUNCON) + np.sum(FCONF) + np.sum(FST) - P
        
        if abs(F) <= tol:
            break
            
        if F > 0: x -= 0.05 * x
        elif F < 0: x += 0.05 * x
        if niter > itermax:
            message = 3
            break

    cores = (current_def / x) * abs(x - dcore)
    
    # 1. Check for physical failure WITHOUT breaking immediately
    if unconfined.lower() != confined.lower() and cores >= ecu: message = 1
    if unconfined.lower() == confined.lower() and current_def >= ecu: message = 1
    if abs(ees[0]) > esu: message = 2

    # 2. Handle non-convergence vs physical failure
    if message == 3:
        x = D / 2  # Reset corrupted NA guess before next step
        continue   # Skip this impossible strain and proceed to next increment.
    elif message != 0:
        break      # Terminate analysis

    # 3. Calculate moment and append (Safe from garbage data)
    current_mom = (np.sum(FUNCON * conclay[:, 0]) + np.sum(FCONF * conclay[:, 0]) + np.sum(FST * distld) - P*(D/2)) / 10**6
    if current_mom < 0: current_mom = -0.01 * current_mom
        
    ejen.append(x)
    mom.append(current_mom)
    curv.append(1000 * current_def / x)
    coverstrain.append(current_def)
    corestrain.append(cores)
    steelstrain.append(ees[0])
    
    x_int = x  # Pass successful NA to the next step to speed up solver

curv, mom = np.array(curv), np.array(mom)
coverstrain = np.array(coverstrain)
steelstrain = np.array(steelstrain)
ejen = np.array(ejen)

# ==============================================================================
# MEMBER RESPONSE & PLOT 3
# ==============================================================================
Agross = 0.25 * np.pi * (D**2)
AsLong = nbl * Asb
LongSteelRatio = AsLong / Agross
TransvSteelRatio = np.pi * (Dh**2) / (s * Dsp)
AxialRatio = P / (fpc * Agross)

Mn_c = np.interp(ecser, coverstrain, mom)

# Check what the steel strain was at the EXACT step concrete reached ecser
esaux = np.interp(ecser, coverstrain, np.abs(steelstrain))

cr = 0  # 0 = concrete controls, 1 = steel controls
Mn = Mn_c

# If steel exceeded its limit before or when concrete reached its limit, steel controls.
if esaux > esser or np.isnan(Mn_c):
    if np.max(np.abs(steelstrain)) >= esser:
        Mn_s = np.interp(esser, np.abs(steelstrain), mom)
        cr = 1
        Mn = Mn_s

# Interpolate neutral axis securely against the controlling STRAIN, not moment
if cr == 0:
    cMn = np.interp(ecser, coverstrain, ejen)
else:
    cMn = np.interp(esser, np.abs(steelstrain), ejen)

fycurvC = np.interp(1.8 * fpc / Ec, coverstrain, curv)
fycurvS = np.interp(fy / Es, -steelstrain, curv)
fycurv = min(fycurvC, fycurvS)
fyM = np.interp(fycurv, curv, mom)

eqcurv = max((Mn / fyM) * fycurv, fycurv)
curvbilin = [0, eqcurv, curv[-1]]
mombilin = [0, Mn, mom[-1]]
SectionCurvatureDuctility = curv[-1] / eqcurv

# Plot 3: Moment-Curvature Relation

fig3, ax1_p3 = plt.subplots(figsize=(6.5, 5))
ax1_p3.plot(curvbilin, mombilin, color='silver', linestyle='--', linewidth=2, label='Bilinear')
ax1_p3.plot(curv, mom, color='black', linewidth=2, label='Total Response')
ax1_p3.grid(True, linestyle=':', alpha=0.7)

# Fix limits to anchor 0 perfectly on both sides
ax1_p3.set_xlim(0, max(curv) * 1.05)
ax1_p3.set_ylim(0, max(mom) * 1.1)

# Secondary X: Curvature Ductility (Forces primary ticks to align)
mu_phi_max = int(np.ceil(ax1_p3.get_xlim()[1] / eqcurv))

# Adjust curvature ductility tick spacing to prevent crowding
if mu_phi_max > 19:
    desired_mu_ticks = np.arange(0, mu_phi_max + 1, 3) 
else:
    desired_mu_ticks = np.arange(0, mu_phi_max + 1, 2)  

ax1_p3.set_xticks(desired_mu_ticks * eqcurv)

ax2_p3 = ax1_p3.twiny()
ax2_p3.set_xlim([x / eqcurv for x in ax1_p3.get_xlim()])
ax2_p3.set_xticks(desired_mu_ticks)

# Secondary Y: M / Mn (Forces primary ticks to align)
m_ratio_max = int(np.ceil(ax1_p3.get_ylim()[1] / Mn * 10))
desired_m_ticks = np.array([i/10 for i in range(0, m_ratio_max + 1, 2)])  # Tick every 0.2
ax1_p3.set_yticks(desired_m_ticks * Mn)

ax3_p3 = ax1_p3.twinx()
ax3_p3.set_ylim([y / Mn for y in ax1_p3.get_ylim()])
ax3_p3.set_yticks(desired_m_ticks)

ax1_p3.set_xlabel('Curvature (1/m)')
ax1_p3.set_ylabel('Moment (kN-m)')
ax2_p3.set_xlabel('Curvature Ductility ($\mu_{\phi}$)')
ax3_p3.set_ylabel('$M / M_n$')
ax1_p3.set_title('Moment - Curvature Relation', fontweight='bold', pad=15)
ax1_p3.legend(loc='lower right')
fig3.tight_layout()
# ==============================================================================
# FLEXURE, SHEAR, & BUCKLING
# ==============================================================================
Lsp_arr = np.zeros_like(steelstrain)
for j in range(len(steelstrain)):
    ffss = min(-steelstrain[j] * Es, fy)
    Lsp_arr[j] = kLsp * ffss * Dbl

kkk = min(0.2 * (fsu / fy - 1), 0.08)

if bending.lower() == 'single':
    Lp = max(kkk * L + kLsp * fy * Dbl, 2 * kLsp * fy * Dbl)
    LBE = L
elif bending.lower() == 'double':
    Lp = max(kkk * (L/2) + kLsp * fy * Dbl, 2 * kLsp * fy * Dbl)
    LBE = L/2

displf_c = np.zeros_like(curv)
displf_t = np.zeros_like(curv)

if hinge_method.lower() == 'pck':
    if bending.lower() == 'single':
        for i in range(len(curv)):
            if coverstrain[i] < eccr:
                displf_c[i] = curv[i] * ((L/1000)**2) / 3
            elif coverstrain[i] >= eccr and curv[i] < fycurv:
                displf_c[i] = curv[i] * (((L + Lsp_arr[i])/1000)**2) / 3
            else:
                displf_c[i] = (curv[i] - fycurv*(mom[i]/fyM)) * (Lp/1000) * ((L + Lsp_arr[i] - 0.5*Lp)/1000) + \
                              (fycurv * (((L + Lsp_arr[i])/1000)**2) / 3) * (mom[i]/fyM)
        Force = mom / (L/1000)
    elif bending.lower() == 'double':
        for i in range(len(curv)):
            if coverstrain[i] < eccr:
                displf_c[i] = curv[i] * ((L/1000)**2) / 6
            elif coverstrain[i] >= eccr and curv[i] < fycurv:
                displf_c[i] = curv[i] * (((L + 2*Lsp_arr[i])/1000)**2) / 6
            else:
                displf_c[i] = (curv[i] - fycurv*(mom[i]/fyM)) * (Lp/1000) * ((L + 2*(Lsp_arr[i] - 0.5*Lp))/1000) + \
                              (fycurv * (((L + 2*Lsp_arr[i])/1000)**2) / 6) * (mom[i]/fyM)
        Force = 2 * mom / (L/1000)
    displf_t = displf_c.copy()

elif hinge_method.lower() == 'modified_lpr':
    Lc = L if bending.lower() == 'single' else L/2
    k_val = min(0.2 * (fsu / fy - 1), 0.08)
    Lpr_c = 2 * k_val * Lc
    Lpr_t = 2 * k_val * Lc + 0.75 * D

    for i in range(len(curv)):
        # Decoupled strain penetration displacement: D_sp = L_sp * phi_base * L
        delta_sp = (Lsp_arr[i]/1000) * curv[i] * (L/1000) if bending.lower() == 'single' else 2 * (Lsp_arr[i]/1000) * curv[i] * (L/1000)
        
        if bending.lower() == 'single':
            if curv[i] <= fycurv:
                delta_e = curv[i] * ((L/1000)**2) / 3
                delta_p_c, delta_p_t = 0, 0
            else:
                delta_e = fycurv * (mom[i] / fyM) * ((L/1000)**2) / 3
                phi_p = curv[i] - fycurv * (mom[i] / fyM)
                delta_p_c = phi_p * (Lpr_c/2000) * ((L - Lpr_c/3)/1000)
                delta_p_t = phi_p * (Lpr_t/2000) * ((L - Lpr_t/3)/1000)
            Force = mom / (L/1000)

        elif bending.lower() == 'double':
            if curv[i] <= fycurv:
                delta_e = curv[i] * ((L/1000)**2) / 6
                delta_p_c, delta_p_t = 0, 0
            else:
                delta_e = fycurv * (mom[i] / fyM) * ((L/1000)**2) / 6
                phi_p = curv[i] - fycurv * (mom[i] / fyM)
                delta_p_c = phi_p * (Lpr_c/2000) * ((L - 2*Lpr_c/3)/1000)
                delta_p_t = phi_p * (Lpr_t/2000) * ((L - 2*Lpr_t/3)/1000)
            Force = 2 * mom / (L/1000)

        displf_c[i] = delta_e + delta_p_c + delta_sp
        displf_t[i] = delta_e + delta_p_t + delta_sp

# The main backbone uses the compressive equivalent curvature
displf = displf_c.copy()

G = 0.43 * Ec
As_shear = 0.9 * Agross
Ig = np.pi * (D**4) / 64
Ieff = (Mn * 1000 / (Ec * (10**6) * eqcurv)) * (10**12)
beta = min(0.5 + 20 * LongSteelRatio, 1)

kscr_base = ((0.39 * TransvSteelRatio) * 0.25 * Es * ((0.8 * D / 1000)**2) / (0.25 + 10 * (0.39 * TransvSteelRatio))) * 1000

if bending.lower() == 'single':
    alpha = min(max(1, 3 - L/D), 1.5)
    ksg = (G * As_shear / L) / 1000
    kscr = kscr_base / L
    forcebilin = np.array(mombilin) / (L/1000)
elif bending.lower() == 'double':
    alpha = min(max(1, 3 - L/(2*D)), 1.5)
    ksg = (G * As_shear / (L/2)) / 1000
    kscr = kscr_base / (L/2)
    forcebilin = 2 * np.array(mombilin) / (L/1000)

Vc1 = 0.29 * alpha * beta * 0.8 * (fpc**(0.5)) * Agross / 1000
kseff = ksg * (Ieff / Ig)

aux = (Vc1 / kseff) / 1000
displsh = np.zeros_like(curv)
passed_Mn = False

for i in range(len(curv)):
    if mom[i] > Mn or passed_Mn:
        passed_Mn = True
        displsh[i] = (displf[i] / displf[i-1]) * displsh[i-1] if displf[i-1] != 0 else 0
    else:
        if Force[i] < Vc1:
            displsh[i] = (Force[i] / kseff) / 1000
        else:
            displsh[i] = ((Force[i] - Vc1) / kscr) / 1000 + aux

displ = displsh + displf
displ_t = displsh + displf_t  # tracks tension-shifted displacements
dy1 = np.interp(fycurv, curv, displ)
dy = (Mn / fyM) * dy1
du = displ[-1]
displbilin = [0, dy, du]
Dduct = displ / dy
Dduct_t = displ_t / dy  # Tracks tensile ductility

# ==============================================================================
# P-DELTA EFFECTS (SIMPLIFIED POST-PROCESSING)
# ==============================================================================
if p_delta.lower() == 'y':
    # The overturning moment is P*Delta. The reduction in lateral force is P*Delta/L
    # regardless of whether the restoring moment is 1M (single) or 2M (double).
    Force = Force - (P_kN * displ / (L / 1000))
    forcebilin = forcebilin - (P_kN * np.array(displbilin) / (L / 1000))
    
    
# Plot 4: Moyer - Kowalsky & Goodnight Strain Buckling Models

bucritMK = 0
failCuDuMK, buckldispl, bucklforce = 0, 0, 0
bucklcurv, bucklDd, bucklmom, failss = 0, 0, 0, 0

bucritGN_strain = 0
failCuDuGN_strain, buckldisplGN_strain, bucklforceGN_strain = 0, 0, 0
bucklcurvGN_strain, bucklDdGN_strain, bucklmomGN_strain, failss_gn = 0, 0, 0, 0

CuDu = curv / eqcurv

# Goodnight Eqn 2: Peak tensile strain prior to bar buckling
es_bb = 0.03 + 700 * TransvSteelRatio * (fyh / Es) - 0.1 * AxialRatio

# Evaluate Goodnight Strain Limit
fail_gn_strain = es_bb - (-steelstrain)

if fail_gn_strain[-1] <= 0:
    for i in range(len(fail_gn_strain)-1):
        if fail_gn_strain[i] >= 0 and fail_gn_strain[i+1] <= 0:
            bucritGN_strain = 1
            fraction = fail_gn_strain[i] / (fail_gn_strain[i] - fail_gn_strain[i+1])
            failCuDuGN_strain = CuDu[i] + fraction * (CuDu[i+1] - CuDu[i])
            failss_gn = (-steelstrain[i]) + fraction * ((-steelstrain[i+1]) - (-steelstrain[i]))
            buckldisplGN_strain = np.interp(failCuDuGN_strain, CuDu, displ_t)
            bucklforceGN_strain = np.interp(failCuDuGN_strain, CuDu, Force)
            bucklcurvGN_strain = np.interp(failCuDuGN_strain, CuDu, curv)
            bucklDdGN_strain = np.interp(failCuDuGN_strain, CuDu, Dduct_t)
            bucklmomGN_strain = np.interp(failCuDuGN_strain, CuDu, mom)
            break


fig4, ax1_p4 = plt.subplots(figsize=(6.5, 4.5))
ax1_p4.plot(CuDu, -steelstrain, color='salmon', linewidth=2, label='Column strain ductility behavior')
ax1_p4.plot(CuDu, es_bb * np.ones_like(CuDu), color='firebrick', linestyle='-.', linewidth=2, label='Buckling Strain Limit (GKN)')

if bucritGN_strain == 1:
    ax1_p4.plot(failCuDuGN_strain, failss_gn, marker='^', color='firebrick', markersize=10, linestyle='None', zorder=5, label='Buckling (GKN)')

# 2. Evaluate Moyer-Kowalsky Limit (Only run the math if Ductility > 4)
if SectionCurvatureDuctility > 4:
    esgr4 = -0.5 * np.interp(4, CuDu, steelstrain)
    escc = 3 * ((s / Dbl)**(-2.5))
    esgr = np.zeros_like(steelstrain)
    for i in range(len(steelstrain)):
        if CuDu[i] < 1: esgr[i] = 0
        elif 1 <= CuDu[i] <= 4: esgr[i] = (esgr4 / 4) * CuDu[i]
        else: esgr[i] = -0.5 * steelstrain[i]
    esfl = escc - esgr

    ax1_p4.plot(CuDu, esfl, color='cornflowerblue', linestyle='--', linewidth=2, label='Flexural Tension Strain (M&K)')

    fail_mk = esfl - (-steelstrain)
    if fail_mk[-1] <= 0:
        for i in range(len(fail_mk)-1):
            if fail_mk[i] >= 0 and fail_mk[i+1] <= 0:
                bucritMK = 1
                fraction = fail_mk[i] / (fail_mk[i] - fail_mk[i+1])
                failCuDuMK = CuDu[i] + fraction * (CuDu[i+1] - CuDu[i])
                failss = (-steelstrain[i]) + fraction * ((-steelstrain[i+1]) - (-steelstrain[i]))
                buckldispl = np.interp(failCuDuMK, CuDu, displ_t)
                bucklforce = np.interp(failCuDuMK, CuDu, Force)
                bucklcurv = np.interp(failCuDuMK, CuDu, curv)
                bucklDd = np.interp(failCuDuMK, CuDu, Dduct_t)
                bucklmom = np.interp(failCuDuMK, CuDu, mom)
                ax1_p4.plot(failCuDuMK, failss, marker='o', color='cornflowerblue', markersize=10, linestyle='None', zorder=5, label='Buckling (M & K)')
                break

# 3. Format and label the figure unconditionally
ax1_p4.grid(True, linestyle=':', alpha=0.7)
ax1_p4.set_xlabel('Curvature Ductility')
ax1_p4.set_ylabel('Steel Tension Strain')
ax1_p4.set_title('Strain-Based Buckling Models', fontweight='bold')
ax1_p4.legend()
fig4.tight_layout()

# Plot 5: Berry - Eberhard
bucritBE = 0
failCuDuBE, buckldisplBE, bucklforceBE = 0, 0, 0
bucklcurvBE, bucklDdBE, bucklmomBE, failplrot = 0, 0, 0, 0

C0, C1, C2, C3, C4 = 0.006, 7.190, 3.129, 0.651, 0.227

roeff = TransvSteelRatio * fyh / fpc
rotb = C0 * (1 + C1*roeff) * ((1 + C2*P/(Agross*fpc))**-1) * (1 + C3*LBE/D + C4*Dbl*fy/D)
plrot = (curv - fycurv) * Lp / 1000

fig5, ax1_p5 = plt.subplots(figsize=(6.5, 4.5))
ax1_p5.plot(CuDu, rotb * np.ones_like(CuDu), color='salmon', linestyle='--', linewidth=2, label='Plastic Rotation for Buckling')
ax1_p5.plot(CuDu, plrot, color='cornflowerblue', linewidth=2, label='Plastic Rotation')

if np.max(plrot) > rotb:
    fail_be = plrot - rotb
    for i in range(len(fail_be)-1):
        if fail_be[i] <= 0 and fail_be[i+1] >= 0:
            bucritBE = 1
            fraction = -fail_be[i] / (fail_be[i+1] - fail_be[i])
            failCuDuBE = CuDu[i] + fraction * (CuDu[i+1] - CuDu[i])
            failplrot = plrot[i] + fraction * (plrot[i+1] - plrot[i])
            buckldisplBE = np.interp(failCuDuBE, CuDu, displ)
            bucklforceBE = np.interp(failCuDuBE, CuDu, Force)
            bucklcurvBE = np.interp(failCuDuBE, CuDu, curv)
            bucklDdBE = np.interp(failCuDuBE, CuDu, Dduct)
            bucklmomBE = np.interp(failCuDuBE, CuDu, mom)
            ax1_p5.plot(failCuDuBE, failplrot, marker='o', color='blueviolet', markersize=10, linestyle='None', zorder=5, label='Buckling')
            break

ax1_p5.grid(True, linestyle=':', alpha=0.7)
ax1_p5.set_xlabel('Curvature Ductility')
ax1_p5.set_ylabel('Plastic Rotation')
ax1_p5.set_title('Berry - Eberhard Buckling Model', fontweight='bold')
ax1_p5.legend()
fig5.tight_layout()

# Goodnight et al. (2015) Drift-Based Buckling Model
bucritGN_drift = 0
failCuDuGN_drift, buckldisplGN_drift, bucklforceGN_drift = 0, 0, 0
bucklcurvGN_drift, bucklDdGN_drift, bucklmomGN_drift = 0, 0, 0

# Eqn 4: Drift-Based limit (%)
drift_bb_pct = 0.9 - 3.13 * AxialRatio + 142000 * TransvSteelRatio * (fyh / Es) + 0.45 * (L / D)
buckldisplGN_drift = (drift_bb_pct / 100.0) * (L / 1000)

if 0 < buckldisplGN_drift <= displ[-1]:
    bucritGN_drift = 1
    failCuDuGN_drift = np.interp(buckldisplGN_drift, displ, CuDu)
    bucklforceGN_drift = np.interp(buckldisplGN_drift, displ, Force)
    bucklcurvGN_drift = np.interp(buckldisplGN_drift, displ, curv)
    bucklDdGN_drift = np.interp(buckldisplGN_drift, displ, Dduct)
    bucklmomGN_drift = np.interp(buckldisplGN_drift, displ, mom)
    
# Shear Capacity
dy1f = np.interp(fycurv, curv, displf)
dyf = (Mn / fyM) * dy1f
Dductf = displ / dyf

Vs = (0.5 * np.pi * (0.25 * np.pi * (Dh**2)) * fyh * (1/np.tan(np.pi/6)) * (D - clb + 0.5*Dh - cMn) / s) / 1000
Vsd = (0.5 * np.pi * (0.25 * np.pi * (Dh**2)) * fyh * (1/np.tan((35/180)*np.pi)) * (D - clb + 0.5*Dh - cMn) / s) / 1000

if bending.lower() == 'single':
    Vp = (P * (D - cMn) / (2 * L)) / 1000 if P > 0 else 0
elif bending.lower() == 'double':
    Vp = (P * (D - cMn) / L) / 1000 if P > 0 else 0

Vc = np.zeros_like(Dductf)
if ductilitymode.lower() == 'uniaxial':
    for i in range(len(Dductf)):
        Vc[i] = alpha * beta * min(max(0.05, 0.37 - 0.04*Dductf[i]), 0.29) * 0.8 * (fpc**(0.5)) * Agross / 1000
elif ductilitymode.lower() == 'biaxial':
    for i in range(len(Dductf)):
        Vc[i] = alpha * beta * min(max(0.05, 0.33 - 0.04*Dductf[i]), 0.29) * 0.8 * (fpc**(0.5)) * Agross / 1000

Vcd = 0.862 * Vc
Vpd = 0.85 * Vp
V = Vc + Vs + Vp
Vd = 0.85 * (Vcd + Vsd + Vpd)

criteria = 1
faildispl, failforce = 0, 0
if V[-1] < Force[-1]:
    failure = V - Force
    for i in range(len(failure)-1):
        if failure[i] >= 0 and failure[i+1] <= 0:
            fraction = failure[i] / (failure[i] - failure[i+1])
            faildispl = displ[i] + fraction * (displ[i+1] - displ[i])
            failforce = np.interp(faildispl, displ, Force)
            failduct = np.interp(faildispl, displ, Dduct)
            failmom = np.interp(faildispl, displ, mom)
            failcurv = np.interp(faildispl, displ, curv)
            failCuDu = np.interp(faildispl, displ, CuDu)
            
            if bending.lower() == 'single':
                if faildispl <= 2 * dy: criteria = 2
                elif faildispl < 8 * dy: criteria = 3
                else: criteria = 4
            elif bending.lower() == 'double':
                if faildispl <= 1 * dy: criteria = 2
                elif faildispl < 7 * dy: criteria = 3
                else: criteria = 4
            break
# ==============================================================================
# GOODNIGHT ET AL. (2015) SPIRAL YIELDING LIMIT STATE
# ==============================================================================
# Eqn 1: Compressive strain at initial yielding of confinement steel
ec_sy = 0.009 - 0.3 * (AsLong / Agross) + 3.9 * (fyh / Es)

disp_sy, Force_sy = np.inf, 0
if np.max(coverstrain) >= ec_sy:
    disp_sy = np.interp(ec_sy, coverstrain, displ)
    Force_sy = np.interp(disp_sy, displ, Force)
    
# Plot 6: Force - Displacement
fig6, ax1_p6 = plt.subplots(figsize=(6.5, 6.5))
ax1_p6.plot(displbilin, forcebilin, color='silver', linestyle='--', linewidth=2, label='Bilinear')
ax1_p6.plot(displ, Force, color='black', linewidth=2, label='Total Response (Compression)')

if hinge_method.lower() == 'modified_lpr':
    ax1_p6.plot(displ_t, Force, color='dimgray', linestyle='--', linewidth=1.5, label='Total Response (Tension)')

ax1_p6.plot(displ, V, color='salmon', linestyle='-.', linewidth=2, label='Shear Capacity (Assessment)')
ax1_p6.plot(displ, Vd, color='pink', linestyle=':', linewidth=2, label='Shear Capacity (Design)')

if criteria != 1:
    ax1_p6.plot(faildispl, failforce, marker='o', color='blueviolet', markersize=10, linestyle='None', zorder=5, label='Shear Failure')
if bucritMK == 1:
    ax1_p6.plot(buckldispl, bucklforce, marker='o', color='cornflowerblue', markersize=10, linestyle='None', zorder=5, label='Buckling (M & K)')
if bucritBE == 1:
    ax1_p6.plot(buckldisplBE, bucklforceBE, marker='s', color='cornflowerblue', markersize=10, linestyle='None', zorder=5, label='Buckling (B & E)')
if bucritGN_strain == 1:
    ax1_p6.plot(buckldisplGN_strain, bucklforceGN_strain, marker='^', color='firebrick', markersize=10, linestyle='None', zorder=5, label='Buckling (GKN Strain)')
if bucritGN_drift == 1:
    ax1_p6.plot(buckldisplGN_drift, bucklforceGN_drift, marker='v', color='darkorange', markersize=10, linestyle='None', zorder=5, label='Buckling (GKN Drift)')
if disp_sy != np.inf:
    ax1_p6.plot(disp_sy, Force_sy, marker='d', color='darkorange', markersize=9, linestyle='None', zorder=6, label='Spiral Yielding')

ax1_p6.grid(True, linestyle=':', alpha=0.7)

# Fix limits to anchor 0 perfectly on both sides (ensuring tension backbone fits)
max_x = np.max(displ)
if hinge_method.lower() == 'modified_lpr': max_x = max(max_x, np.max(displ_t))
if bucritMK == 1: max_x = max(max_x, buckldispl)
if bucritGN_strain == 1: max_x = max(max_x, buckldisplGN_strain)
if bucritBE == 1: max_x = max(max_x, buckldisplBE)
if bucritGN_drift == 1: max_x = max(max_x, buckldisplGN_drift)
if disp_sy != np.inf: max_x = max(max_x, disp_sy)

ax1_p6.set_xlim(0, max_x * 1.05)
max_F = max(max(Force), max(V), max(Vd))
ax1_p6.set_ylim(0, max_F * 1.1)

# Secondary X: Displacement Ductility 
mu_delta_max = int(np.ceil(ax1_p6.get_xlim()[1] / dy))
if mu_delta_max <= 10:
    desired_mu_d_ticks = np.arange(1, mu_delta_max + 1, 1) 
else:
    desired_mu_d_ticks = np.arange(1, mu_delta_max + 1, 2)

ax1_p6.set_xticks(desired_mu_d_ticks * dy)
ax1_p6.set_xticklabels([f"{t:.3f}" for t in (desired_mu_d_ticks * dy)])

ax2_p6 = ax1_p6.twiny()
ax2_p6.set_xlim([x / dy for x in ax1_p6.get_xlim()])
ax2_p6.set_xticks(desired_mu_d_ticks)
ax2_p6.set_xlabel('Displacement Ductility ($\mu_{\Delta}$)')

# Secondary Y: F / P_axial
if P_kN != 0:
    fp_ratio_max = int(np.ceil(ax1_p6.get_ylim()[1] / abs(P_kN) * 10))
    desired_fp_ticks = np.array([i/10 for i in range(0, fp_ratio_max + 1, 4)])
    ax1_p6.set_yticks(desired_fp_ticks * abs(P_kN))
    
    ax3_p6 = ax1_p6.twinx()
    ax3_p6.set_ylim([y / abs(P_kN) for y in ax1_p6.get_ylim()])
    ax3_p6.set_yticks(desired_fp_ticks)
    ax3_p6.set_ylabel('$F / P_{axial}$')

ax1_p6.set_xlabel('Displacement (m)')
ax1_p6.set_ylabel('Force (kN)')
ax1_p6.set_title('Force - Displacement Relation', fontweight='bold', pad=15)
ax1_p6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
fig6.subplots_adjust(bottom=0.3, top=0.85)

# Plot 7: Limit States
displserc = np.interp(ecser, coverstrain, displ)
displsers = np.interp(esser, -steelstrain, displ_t)  
displser = min(displserc, displsers)

displdamc = np.interp(ecdam, coverstrain, displ)
displdams = np.interp(esdam, -steelstrain, displ_t)  
displdam = min(displdamc, displdams)

fig7, ax1_p7 = plt.subplots(figsize=(6.5, 6.5))
ax1_p7.fill_between(displ, Force, color='cornflowerblue', alpha=0.2, label='Ultimate Zone')
ax1_p7.fill_between(displ[displ <= displdam], Force[displ <= displdam], color='cornflowerblue', alpha=0.5, label='Damage Control Zone')
ax1_p7.fill_between(displ[displ <= displser], Force[displ <= displser], color='cornflowerblue', alpha=0.8, label='Serviceability Zone')

ax1_p7.plot(displbilin, forcebilin, color='silver', linestyle='--', linewidth=2, label='Bilinear Approximation')
ax1_p7.plot(displ, Force, color='black', linewidth=2, label='Total Response (Compression)')

if hinge_method.lower() == 'modified_lpr':
    ax1_p7.plot(displ_t, Force, color='dimgray', linestyle='--', linewidth=1.5, label='Total Response (Tension)')

ax1_p7.plot(displ, V, color='salmon', linestyle='-.', linewidth=2, label='Shear Capacity (Assessment)')
ax1_p7.plot(displ, Vd, color='pink', linestyle=':', linewidth=2, label='Shear Capacity (Design)')

if criteria != 1:
    ax1_p7.plot(faildispl, failforce, marker='o', color='blueviolet', markersize=10, linestyle='None', zorder=5, label='Shear Failure')
if bucritMK == 1:
    ax1_p7.plot(buckldispl, bucklforce, marker='o', color='cornflowerblue', markersize=10, linestyle='None', zorder=5, label='Buckling (M & K)')
if bucritBE == 1:
    ax1_p7.plot(buckldisplBE, bucklforceBE, marker='s', color='cornflowerblue', markersize=10, linestyle='None', zorder=5, label='Buckling (B & E)')
if bucritGN_strain == 1:
    ax1_p7.plot(buckldisplGN_strain, bucklforceGN_strain, marker='^', color='firebrick', markersize=10, linestyle='None', zorder=5, label='Buckling (GKN Strain)')
if bucritGN_drift == 1:
    ax1_p7.plot(buckldisplGN_drift, bucklforceGN_drift, marker='v', color='darkorange', markersize=10, linestyle='None', zorder=5, label='Buckling (GKN Drift)')
if disp_sy != np.inf:
    ax1_p7.plot(disp_sy, Force_sy, marker='d', color='darkorange', markersize=9, linestyle='None', zorder=6, label='Spiral Yielding')

ax1_p7.grid(True, linestyle=':', alpha=0.7)

ax1_p7.set_xlim(0, max_x * 1.05)
ax1_p7.set_ylim(0, max_F * 1.1)

ax1_p7.set_xticks(desired_mu_d_ticks * dy)
ax1_p7.set_xticklabels([f"{t:.3f}" for t in (desired_mu_d_ticks * dy)])

ax2_p7 = ax1_p7.twiny()
ax2_p7.set_xlim([x / dy for x in ax1_p7.get_xlim()])
ax2_p7.set_xticks(desired_mu_d_ticks)
ax2_p7.set_xlabel('Displacement Ductility ($\mu_{\Delta}$)')

# Secondary Y: F / P_axial
if P_kN != 0:
    ax1_p7.set_yticks(desired_fp_ticks * abs(P_kN))
    ax3_p7 = ax1_p7.twinx()
    ax3_p7.set_ylim([y / abs(P_kN) for y in ax1_p7.get_ylim()])
    ax3_p7.set_yticks(desired_fp_ticks)
    ax3_p7.set_ylabel('$F / P_{axial}$')

ax1_p7.set_xlabel('Displacement (m)')
ax1_p7.set_ylabel('Force (kN)')
ax1_p7.set_title('Potential Deformation Limit States', fontweight='bold', pad=15)
ax1_p7.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
fig7.subplots_adjust(bottom=0.3, top=0.85)
# ==============================================================================
# INTERACTION DIAGRAM & PLOT 8
# ==============================================================================
Acore = 0.25 * np.pi * (Dsp**2)
PCid = np.interp(csid, ec_full, fc_full)*(Acore - AsLong) + np.interp(csid, ecun_full, fcun_full)*(Agross - Acore) + AsLong*np.interp(csid, es_steel_full, fs_steel_full)

Mni, PPn = [], []
if interaction.lower() == 'y':
    PTid = AsLong * np.interp(ssid, es_steel_full, fs_steel_full)
    
    PP = np.concatenate((np.linspace(-0.95 * PTid, 0, 7,endpoint=False),
                        np.linspace(0, 0.70 * PCid, 30,endpoint=False),
                        np.linspace(0.70 * PCid, 0.98 * PCid, 3)))

    def interp_with_nan(x_target, xp, yp):
        if len(xp) == 0: return np.nan
        if x_target < np.min(xp) or x_target > np.max(xp): return np.nan
        sort_idx = np.argsort(xp)
        return np.interp(x_target, np.array(xp)[sort_idx], np.array(yp)[sort_idx])

    for Pi in PP:
        current_def_strain = def_strain.copy()
        
        if Pi > 0:
            valid_start = 0
            for current_def in current_def_strain:
                compch = np.sum(np.interp(np.full(len(yl), current_def), ecun_full, fcun_full) * conclay[:, 1]) + \
                         np.sum(np.interp(np.full(len(yl), current_def), ec_full, fc_full) * conclay[:, 2]) + \
                         np.sum(Asbs * np.interp(np.full(len(distld), current_def), es_steel_full, fs_steel_full))
                if compch < Pi:
                    valid_start += 1
                else:
                    break
            if valid_start < len(current_def_strain):
                current_def_strain = current_def_strain[valid_start:]
            else:
                continue 

        mom_i, coverstrain_i, steelstrain_i = [0], [0], [0]
        x_int = D / 2

        for k in range(len(current_def_strain)):
            F = 10 * tol
            niter = 0
            current_def = current_def_strain[k]

            while abs(F) > tol:
                niter += 1
                if x_int == 0 or np.isinf(x_int) or np.isnan(x_int):
                    x_int = 1e-10       
                    current_def = 0.0
                eec = (current_def / x_int) * (conclay[:, 0] - (D - x_int))
                ees = (current_def / x_int) * (distld - (D - x_int))

                fcunconf = np.interp(eec, ecun_full, fcun_full, right=0)
                fcconf = np.interp(eec, ec_full, fc_full, right=0)
                fsteel = np.interp(ees, es_steel_full, fs_steel_full)

                FUNCON = fcunconf * conclay[:, 1]
                FCONF = fcconf * conclay[:, 2]
                FST = Asbs * fsteel

                F = np.sum(FUNCON) + np.sum(FCONF) + np.sum(FST) - Pi

                if abs(F) <= tol:
                    break

                if F > 0: x_int -= 0.05 * x_int
                elif F < 0: x_int += 0.05 * x_int
                if niter > itermax: break

            cores = (current_def / x_int) * abs(x_int - dcore)
            if cores >= ecu: break
            if abs(ees[0]) > esu: break

            current_mom = (np.sum(FUNCON * conclay[:, 0]) + np.sum(FCONF * conclay[:, 0]) + np.sum(FST * distld) - Pi*(D/2)) / 10**6
            if current_mom < 0: current_mom = -0.01 * current_mom

            mom_i.append(current_mom)
            coverstrain_i.append(current_def)
            steelstrain_i.append(ees[0])

        mn_concrete = interp_with_nan(csid, coverstrain_i, mom_i)
        es_at_csid = interp_with_nan(csid, coverstrain_i, steelstrain_i)

        cr = 0
        mn_final = mn_concrete

        if pd.isna(mn_concrete) or (not pd.isna(es_at_csid) and abs(es_at_csid) > ssid):
            mn_steel = interp_with_nan(-ssid, steelstrain_i, mom_i)
            if not pd.isna(mn_steel):
                cr = 1
                mn_final = mn_steel

        if not pd.isna(mn_final):
            Mni.append(mn_final)
            PPn.append(Pi/1000)

    # Mni = np.concatenate(([0], Mni, [0]))
    # PPn = np.concatenate(([-PTid/1000], PPn, [PCid/1000]))

    MB = max(Mni)
    PB = PPn[np.argmax(Mni)]
    MB0 = np.interp(0, PPn, Mni)
    MB13 = np.interp(PB/3, PPn, Mni)
    MB23 = np.interp(PB*2/3, PPn, Mni)

    MnL = [0, MB0, MB13, MB23, MB, 0]
    PPL = [-PTid/1000, 0, PB/3, PB*2/3, PB, PCid/1000]

    # Plot 8: Interaction Diagram
    fig8, ax1_p8 = plt.subplots(figsize=(6.5, 5))
    ax1_p8.plot(Mni, PPn, marker='o', color='salmon', linewidth=2, markersize=6, label='Interaction Diagram')
    ax1_p8.plot(MnL, PPL, marker='s', color='cornflowerblue', linestyle='--', linewidth=2, markersize=6, label='Approximation for NLTHA')
    ax1_p8.plot([0, 1.1*MB], [0, 1.1*PB], color='silver', linestyle='-.', linewidth=2, label='Balance Condition')
    ax1_p8.grid(True, linestyle=':', alpha=0.7)
    ax1_p8.set_xlabel('Moment (kN-m)')
    ax1_p8.set_ylabel('Axial Load (kN)')
    ax1_p8.set_title('Interaction Diagram', fontweight='bold')
    ax1_p8.legend()
    fig8.tight_layout()



# ==============================================================================
# DATA EXPORT & FORMATTED EXCEL REPORT
# ==============================================================================
def safe_interp(x_target, xp, yp):
    sort_idx = np.argsort(xp)
    return np.interp(x_target, np.array(xp)[sort_idx], np.array(yp)[sort_idx])

displdam, displser, Dductdam, Dductser = np.inf, np.inf, 0, 0
curvdam, curvser, CuDudam, CuDuser = 0, 0, 0, 0
coverstraindam, coverstrainser, steelstraindam, steelstrainser = 0, 0, 0, 0
momdam, momser, Forcedam, Forceser = 0, 0, 0, 0

if np.max(coverstrain) > ecser or np.max(np.abs(steelstrain)) > abs(esser):
    if np.max(coverstrain) > ecdam or np.max(np.abs(steelstrain)) > abs(esdam):
        displdamc = safe_interp(ecdam, coverstrain, displ) if np.max(coverstrain) >= ecdam else np.inf
        displdams = safe_interp(abs(esdam), np.abs(steelstrain), displ_t) if np.max(np.abs(steelstrain)) >= abs(esdam) else np.inf  # Updated to displ_t
        displdam = min(displdamc, displdams)
        
        Dductdam = safe_interp(displdam, displ, Dduct)
        curvdam = safe_interp(displdam, displ, curv)
        CuDudam = safe_interp(displdam, displ, CuDu)
        coverstraindam = safe_interp(displdam, displ, coverstrain)
        steelstraindam = -safe_interp(displdam, displ, np.abs(steelstrain))
        momdam = safe_interp(displdam, displ, mom)
        Forcedam = safe_interp(displdam, displ, Force)
        
    displserc = safe_interp(ecser, coverstrain, displ) if np.max(coverstrain) >= ecser else np.inf
    displsers = safe_interp(abs(esser), np.abs(steelstrain), displ_t) if np.max(np.abs(steelstrain)) >= abs(esser) else np.inf  # Updated to displ_t
    displser = min(displserc, displsers)
    
    Dductser = safe_interp(displser, displ, Dduct)
    curvser = safe_interp(displser, displ, curv)
    CuDuser = safe_interp(displser, displ, CuDu)
    coverstrainser = safe_interp(displser, displ, coverstrain)
    steelstrainser = -safe_interp(displser, displ, np.abs(steelstrain))
    momser = safe_interp(displser, displ, mom)
    Forceser = safe_interp(displser, displ, Force)

disp_sy_val, Dduct_sy, curv_sy, CuDu_sy, cover_sy, steel_sy, mom_sy, Force_sy_val = np.inf, 0, 0, 0, 0, 0, 0, 0
if np.max(coverstrain) >= ec_sy:
    disp_sy_val = safe_interp(ec_sy, coverstrain, displ)
    Dduct_sy = safe_interp(disp_sy_val, displ, Dduct)
    curv_sy = safe_interp(disp_sy_val, displ, curv)
    CuDu_sy = safe_interp(disp_sy_val, displ, CuDu)
    cover_sy = ec_sy
    steel_sy = -safe_interp(disp_sy_val, displ, np.abs(steelstrain))
    mom_sy = safe_interp(disp_sy_val, displ, mom)
    Force_sy_val = safe_interp(disp_sy_val, displ, Force)

outputlimit = [
    ["Serviceability", coverstrainser, steelstrainser, momser, Forceser, curvser, CuDuser, displser, Dductser],
    ["Spiral Yielding", cover_sy, steel_sy, mom_sy, Force_sy_val, curv_sy, CuDu_sy, disp_sy_val, Dduct_sy],
    ["Damage Control", coverstraindam, steelstraindam, momdam, Forcedam, curvdam, CuDudam, displdam, Dductdam],
    ["Ultimate", np.max(coverstrain), np.min(steelstrain), mom[-1], Force[-1], np.max(curv), np.max(CuDu), np.max(displ), np.max(Dduct)]
]

outputlimit.sort(key=lambda x: x[7])

DisplDuct = np.max(Dduct)
Ieq_val = (Mn / (eqcurv * Ec)) / 1000
Bi = 1 / ((mombilin[1] / curvbilin[1]) / ((mombilin[2] - mombilin[1]) / (curvbilin[2] - curvbilin[1]))) if (curvbilin[2]-curvbilin[1])!=0 else 0

Ig_m4 = np.pi * (D / 1000)**4 / 64
K_factor = Ieq_val / Ig_m4

report_data = []
def add_line(text):
    row = text.split('\t')
    safe_row = []
    for cell in row:
        if str(cell).startswith('='):
            safe_row.append(" " + str(cell))
        else:
            safe_row.append(cell)
    report_data.append(safe_row)
    
add_line("---------------------------------------------------------")
add_line("CUMBIA_PY: Analysis of Reinforced Concrete Members")
add_line("luis.montejo@upr.edu")
add_line("---------------------------------------------------------")

if hinge_method.lower() == 'pck':
    add_line("ACTIVE HINGE METHOD: Priestley et al. (2007) [PCK]")
elif hinge_method.lower() == 'modified_lpr':
    add_line("ACTIVE HINGE METHOD: Goodnight et al. (2016) [Modified Lpr]")
add_line("---------------------------------------------------------")

if confined.lower() == 'mclw':
    add_line("lightweight concrete")
else:
    add_line("normalweight concrete")
add_line(f"Diameter:  {D:.1f} mm")
add_line(f"cover to longitudinal bars:  {clb:.1f} mm")
add_line(f"number of longitudinal bars:    {nbl}")
add_line(f"diameter of longitudinal bars:  {Dbl:.1f} mm")
add_line(f"diameter of transverse steel:   {Dh:.1f} mm")
add_line(f"spacing of transverse steel:  {s:.1f} mm")
add_line(f"type of tranverse reinforcement: {type_reinf}")
add_line(f"axial load:    {P_kN:.2f} kN")
add_line(f"concrete compressive strength:  {fpc:.2f} MPa")
add_line(f"long steel yielding stress:  {fy:.2f} MPa")
add_line(f"long steel max. stress:  {np.max(fs_steel_full):.2f} MPa")
add_line(f"transverse steel yielding stress:  {fyh:.2f} MPa")
add_line(f"Member Length:  {L:.1f} mm")
add_line(f"{bending.capitalize()} Bending")
add_line(f"{ductilitymode.capitalize()} Bending")
add_line(f"Longitudinal Steel Ratio:  {LongSteelRatio:.3f}")
add_line(f"Transverse Steel Ratio:  {TransvSteelRatio:.3f}")
add_line(f"Axial Load Ratio:  {AxialRatio:.3f}")

if p_delta.lower() == 'y':
    add_line("")
    add_line("   *** P-DELTA EFFECTS INCLUDED ***")
    add_line("   Note: P-Delta was approximated by subtracting the geometric overturning")
    add_line("   moment from the baseline force capacity using the total displacement.")

add_line("")
add_line("Bilinear Approximation:")
add_line("")
add_line("Curvature\tMoment\tDispl.\tForce")
add_line("[1/m]\t[kN-m]\t[m]\t[kN]")

for c, m, d, f in zip(curvbilin, mombilin, displbilin, forcebilin):
    add_line(f"{c:.5f}\t{m:.2f}\t{d:.5f}\t{f:.2f}")
    
add_line("")
msg_dict = {
    1: " *** concrete strain exceeds maximum ***",
    2: " *** steel strain exceeds maximum ***",
    3: " *** number of iteration exceeds maximum ***",
    4: " *** excessive lost of strength ***"
}
if message in msg_dict:
    add_line(f"{msg_dict[message]}")
    add_line("")
    
add_line(f"Displacement at First Yield:   {dy1:.5f} m")
add_line(f"Displacement at Ductility One:   {dy:.5f} m")
add_line(f"Moment for First Yielding:   {fyM:.2f} kN-m")
add_line(f"Curvature for First Yielding:  {fycurv:.5f} 1/m")
add_line(f"Potential Section Nominal Moment:   {Mn:.2f} kN-m")
add_line(f"Equivalent Curvature:  {eqcurv:.5f} 1/m")
add_line(f"Potential Section Curvature Ductility:  {SectionCurvatureDuctility:.2f}")
add_line(f"Potential Displacement Ductility:  {DisplDuct:.2f}")
add_line("")

crit_msg = {
    1: " *** flexural failure ***",
    2: " *** brittle shear failure ***",
    3: " *** shear failure at some ductility ***",
    4: " *** ductil shear failure ***"
}
add_line(f"{crit_msg.get(criteria, '')}")

if criteria != 1:
    add_line("")
    add_line(f"Displacement for Shear Failure:  {faildispl:.5f} m")
    add_line(f"Displacement Ductility at Shear Failure:       {failduct:.2f}")
    add_line(f"Force for Shear Failure:   {failforce:.2f} kN")
    add_line(f"Curvature for Shear Failure:  {failcurv:.5f} 1/m")
    add_line(f"Curvature Ductility at Shear Failure:       {failCuDu:.2f}")
    add_line(f"Moment for Shear Failure:   {failmom:.2f} kN-m")
    
add_line("")

if bucritMK == 1:
    add_line("Moyer - Kowalsky buckling model:")
    add_line("")
    add_line(f"Curvature Ductility for Buckling:      {failCuDuMK:.2f}")
    add_line(f"Curvature at Buckling:  {bucklcurv:.5f} m") 
    add_line(f"Displacement Ductility at Buckling:       {bucklDd:.2f}")
    add_line(f"Displacement at Buckling:  {buckldispl:.5f} m")
    add_line(f"Force for Buckling:   {bucklforce:.2f} kN")
    add_line(f"Moment for Buckling:   {bucklmom:.2f} kN")
    add_line("")
    
if bucritBE == 1:
    add_line("Berry - Eberhard buckling model:")
    add_line("")
    add_line(f"Curvature Ductility for Buckling:      {failCuDuBE:.2f}")
    add_line(f"Curvature at Buckling:  {bucklcurvBE:.5f} m")
    add_line(f"Displacement Ductility at Buckling:       {bucklDdBE:.2f}")
    add_line(f"Displacement at Buckling:  {buckldisplBE:.5f} m")
    add_line(f"Force for Buckling:   {bucklforceBE:.2f} kN")
    add_line(f"Moment for Buckling:   {bucklmomBE:.2f} kN")
    add_line("")

if bucritGN_strain == 1:
    add_line("Goodnight, Kowalsky, Nau. Strain-Based Buckling Model:")
    add_line("")
    add_line(f"Curvature Ductility for Buckling:      {failCuDuGN_strain:.2f}")
    add_line(f"Curvature at Buckling:  {bucklcurvGN_strain:.5f} 1/m") 
    add_line(f"Displacement Ductility at Buckling:       {bucklDdGN_strain:.2f}")
    add_line(f"Displacement at Buckling:  {buckldisplGN_strain:.5f} m")
    add_line(f"Force for Buckling:   {bucklforceGN_strain:.2f} kN")
    add_line(f"Moment for Buckling:   {bucklmomGN_strain:.2f} kN-m")
    add_line("")

if bucritGN_drift == 1:
    add_line("Goodnight, Kowalsky, Nau. Drift-Based Buckling Model:")
    add_line("")
    add_line(f"Curvature Ductility for Buckling:      {failCuDuGN_drift:.2f}")
    add_line(f"Curvature at Buckling:  {bucklcurvGN_drift:.5f} 1/m") 
    add_line(f"Displacement Ductility at Buckling:       {bucklDdGN_drift:.2f}")
    add_line(f"Displacement at Buckling:  {buckldisplGN_drift:.5f} m")
    add_line(f"Force for Buckling:   {bucklforceGN_drift:.2f} kN")
    add_line(f"Moment for Buckling:   {bucklmomGN_drift:.2f} kN-m")
    add_line("")

# Force a page break by padding blank lines until the next multiple of 72
while (len(report_data) - 4) % 72 != 0:
    add_line("")
    
add_line("")
add_line("")
add_line("== Potential Deformation Limit States ==")
add_line("")

add_line("Limit State\tCover\tSteel\tMoment\tForce\tCurvature\tCurv.\tDispl.\tDispl.")
add_line("\tStrain\tStrain\t[kN-m]\t[kN]\t[1/m]\tDuctility\t[m]\tDuctility")

for row in outputlimit:
    # If the displacement is infinity, the limit state was never reached. Print N/A.
    if row[7] == np.inf:
        add_line(f"{row[0]}\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A")
    else:
        add_line(f"{row[0]}\t{row[1]:.5f}\t{row[2]:.5f}\t{row[3]:.2f}\t{row[4]:.2f}\t{row[5]:.5f}\t{row[6]:.2f}\t{row[7]:.5f}\t{row[8]:.2f}")
 
add_line("")
add_line("Deformation Limit States Criteria :")
add_line(f"   serviceability concrete strain: {ecser:.4f}")
add_line(f"   serviceability steel strain: {-abs(esser):.4f}")
add_line(f"   spiral yielding concrete strain (GKN): {ec_sy:.4f}")
add_line(f"   bar buckling steel strain (GKN): {-abs(es_bb):.4f}")
add_line(f"   damage control concrete strain: {ecdam:.4f}")
add_line(f"   damage control steel strain: {-abs(esdam):.4f}")


add_line("")
add_line(f"Original Mander Model Ultimate Concrete Strain: {ecumander:.4f}")
add_line("")
add_line("for non-linear THA:")
add_line("")
add_line(f"E: {Ec*(10**6):.2f} Pa")
add_line(f"G: {G*(10**6):.2f} Pa")
add_line(f"A:      {Agross/(10**6):.4f} m2")
add_line(f"I:   {Ieq_val:.6f} m4")
add_line(f"K (I_eff/I_g): {K_factor:.3f}")
add_line(f"Bi-Factor: {Bi:.3f}")

Lc_rep = L if bending.lower() == 'single' else L/2
k_val_rep = min(0.2 * (fsu / fy - 1), 0.08)
Lpr_c_rep = 2 * k_val_rep * Lc_rep
Lpr_t_rep = 2 * k_val_rep * Lc_rep + 0.75 * D
Lsp_nom_rep = kLsp * fy * Dbl

add_line("")
add_line("Equivalent Plastic Hinge Parameters (All Models):")
add_line(f"   Priestley et al. (2007) Hinge Length (Lp):           {Lp/1000:.3f} m")
add_line(f"   Goodnight et al. (2016) Compressive Hinge (Lpr_c):   {Lpr_c_rep/1000:.3f} m")
add_line(f"   Goodnight et al. (2016) Tensile Hinge (Lpr_t):       {Lpr_t_rep/1000:.3f} m")
add_line(f"   Nominal Strain Penetration Length (Lsp):      {Lsp_nom_rep/1000:.3f} m")
add_line("")

add_line(f"Tension Yield: {PTid:.2f} N")
add_line(f"Compression Yield: {PCid:.2f} N")
add_line(f"Moment Yield: {Mn*1000:.2f} N-m")
add_line("")

if interaction.lower() == 'y':
    add_line("NLTHA Approximation:")
    add_line("")
    add_line(f"PT:  {-PTid/1000:.1f} kN")
    add_line(f"PC:  {PCid/1000:.1f} kN")
    add_line(f"     PB:  {PB:.1f} kN\t\tMB:  {MB:.1f} kN-m")
    add_line(f"(1/3)PB:  {PB/3:.1f} kN\t(1/3)MB:  {MB13:.1f} kN-m")
    add_line(f"(2/3)PB:  {PB*2/3:.1f} kN\t(2/3)MB:  {MB23:.1f} kN-m")

df_report = pd.DataFrame(report_data)
df_mc = pd.DataFrame({
    'Cover Strain': coverstrain, 'Core Strain': corestrain, 'N.A [mm]': ejen,
    'Steel Strain': steelstrain, 'Moment [kN-m]': mom, 'Curvature [1/m]': curv,
    'Force [kN]': Force, 'Sh displ. [m]': displsh, 'Fl displ. [m]': displf,
    'Total displ. [m]': displ, 'Shear(assess) [kN]': V, 'Shear(design) [kN]': Vd
})

with pd.ExcelWriter(f"{name}_Results.xlsx") as writer:
    df_report.to_excel(writer, sheet_name='Summary_Report', index=False, header=False)
    df_mc.to_excel(writer, sheet_name='Moment_Curvature', index=False)
    if interaction.lower() == 'y':
        pd.DataFrame({'Moment [kN-m]': Mni, 'Axial Load [kN]': PPn}).to_excel(writer, sheet_name='Interaction', index=False)


# ==============================================================================
# EXPORT FIGURES (PNG) & COMPLETE PDF REPORT
# ==============================================================================
# 1. Save Individual Figures as PNGs
fig_section.savefig(f"{name}_Fig0_Section.png", dpi=300, bbox_inches='tight')
fig1.savefig(f"{name}_Fig1_Concrete.png", dpi=300, bbox_inches='tight')
fig2.savefig(f"{name}_Fig2_Steel.png", dpi=300, bbox_inches='tight')
fig3.savefig(f"{name}_Fig3_MomCurv.png", dpi=300, bbox_inches='tight')
fig4.savefig(f"{name}_Fig4_StrainBuckling.png", dpi=300, bbox_inches='tight')
fig5.savefig(f"{name}_Fig5_BerryEberhard.png", dpi=300, bbox_inches='tight')
fig6.savefig(f"{name}_Fig6_ForceDisp.png", dpi=300, bbox_inches='tight')
fig7.savefig(f"{name}_Fig7_LimitStates.png", dpi=300, bbox_inches='tight')
if interaction.lower() == 'y':
    fig8.savefig(f"{name}_Fig8_Interaction.png", dpi=300, bbox_inches='tight')

# 2. Generate Multi-Page PDF Report
pdf_filename = f"{name}_Full_Report.pdf"
with PdfPages(pdf_filename) as pdf:
    
    # A. Format and Write the Summary Report Text to PDF
    lines_per_page = 72
    report_lines = []
    
    for row in report_data[4:]:
        
        cleaned_row = [str(cell).lstrip('=') for cell in row]
        
        if len(cleaned_row) == 1:
            report_lines.append(cleaned_row[0])
        else:
            # Format as a table: 22 chars for the first column, 14 chars right-aligned for numbers
            line_str = f"{cleaned_row[0]:<22}" + "".join([f"{c:>14}" for c in cleaned_row[1:]])
            report_lines.append(line_str)
            
    header_text = (
        "--------------------------------------------------------\n"
        "CUMBIA_PY: Analysis of Reinforced Concrete Members\n"
        "luis.montejo@upr.edu\n"
        "--------------------------------------------------------"
    )
            
    # Create blank figures for the text pages
    for i in range(0, len(report_lines), lines_per_page):
        fig_text = plt.figure(figsize=(8.5, 11))
        ax_text = fig_text.add_axes([0, 0, 1, 1])
        ax_text.axis('off')
        
        # Add the persistent header
        ax_text.text(0.04, 0.96, header_text, transform=ax_text.transAxes, 
                     fontsize=7.0, verticalalignment='top', fontfamily='monospace', fontweight='bold')
        
        # Add the persistent logo
        try:
            logo = plt.imread("logo.png")
            ax_logo = fig_text.add_axes([0.70, 0.86, 0.20, 0.10], anchor='NE', zorder=10)
            ax_logo.imshow(logo)
            ax_logo.axis('off')
        except FileNotFoundError:
            pass 
        
        # Add the body text shifted down to y=0.88
        page_text = "\n".join([line.replace('\t', '    ') for line in report_lines[i:i+lines_per_page]])
        ax_text.text(0.04, 0.88, page_text, transform=ax_text.transAxes, 
                     fontsize=7.0, verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig_text)
        plt.close(fig_text)
    
    # B. Append all Generated Figures to the PDF
    pdf.savefig(fig_section)
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.savefig(fig3)
    pdf.savefig(fig4)
    pdf.savefig(fig5)
    pdf.savefig(fig6)
    pdf.savefig(fig7)
    if interaction.lower() == 'y':
        pdf.savefig(fig8)

print(f"\nExport complete: Excel, PDF, and PNG figures saved with prefix '{name}'.")
