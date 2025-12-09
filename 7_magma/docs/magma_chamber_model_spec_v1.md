# v1 MAGMA CHAMBER MODEL – DEVELOPER SPEC

**Author**: Ziyan Fiona Fang (*zf276@cam.ac.uk*)

**Goal:**  
0D magma chamber model with two eruption triggers:
1. **Crystallisation-induced degassing**
2. **Magma recharge (basaltic or rhyolitic), with constant or Poisson influx**

Eruptions occur when a stochastic overpressure threshold is exceeded, and eruption size is elastic-limited. 

This spec defines all state variables, parameters, equations, and numerical workflow for v1.

---

## 1.1 State variables

**State Variables**
- $t$ : time (s)
- $T$ : magma temperature (K)
- $\phi$ : crystallinity (0–1)
- $P$ : chamber pressure (Pa)
- $M$ : total bulk magma mass (kg) in the chamber
- $X_{\text{tot}}$ : bulk volatile mass fraction (kg volatiles / kg bulk)
- $V$ : total chamber magma volume (m³) (diagnostic)
- `eruption_log`: list of eruption events (time, $\Delta M$, $\Delta P$)

**Diagnostic Variables**
- Melt fraction: $f_m = 1 - \phi$
- Volatile concentration in melt: $C_m = X_{\text{tot}} / f_m$
- Solubility: $C_{\text{eq}}(P)$
- Exsolved gas fraction: $X_{\text{gas}} = \max(0, X_{\text{tot}} - C_{\text{eq}} \cdot f_m)$
- Gas and liquid-crystal volumes
- Overpressure: $\Delta P = P - P_{\text{lith}}$


## 1.2 Fixed parameters

Physical parameters:
- $K_{\text{eff}}$ : effective bulk modulus (Pa)
- $P_{\text{lith}}$ : lithostatic pressure (Pa)
- $\rho_l$ : liquid+crystal density (kg/m³)
- $R_g$ : gas constant (J kg⁻¹ K⁻¹)
- Solubility law: $C_{\text{eq}} = k \cdot P^n$
- Crystallinity vs $T$ (lever rule): between $T_{\text{liq}}$ and $T_{\text{sol}}$
- Cooling law:
  $$\frac{dT}{dt} = -\frac{T - T_{\text{host}}}{\tau_T}$$

Initial state:
- $T_0$, $\phi_0$, $P_0 = P_{\text{lith}}$, $M_0$, $X_{\text{tot},0}$
- Compute initial $V_0 = M_0 \cdot v_0$

---

## 2.1.1 Cooling & crystallisation

1. Euler cooling:
   $$T^{n+1} = T^n + \left(-\frac{T^n - T_{\text{host}}}{\tau_T}\right) \Delta t$$
2. Crystallinity from $T$ (lever rule)
3. Melt fraction: $f_m = 1 - \phi$


## 2.1.2 Crystallisation-induced degassing

1. $$C_m = \frac{X_{\text{tot}}}{f_m}$$
2. $$C_{\text{eq}} = k \cdot P^n$$
3. $$X_{\text{gas}} = \max(0, X_{\text{tot}} - C_{\text{eq}} \cdot f_m)$$
4. Gas volume per kg:
   $$v_g = \frac{X_{\text{gas}} \cdot R_g \cdot T}{P}$$
5. Liquid-crystal volume per kg:
   $$v_l = \frac{1 - X_{\text{gas}}}{\rho_l}$$
6. Specific volume:
   $$v = v_l + v_g$$
7. Chamber volume:
   $$V = M \cdot v$$

---

## 2.2.1 Magma recharge

Two influx options:

1. Constant influx
- Volume added: $\Delta V_{\text{in}} = Q \cdot \Delta t$

2. Poisson pulses
- Probability per step: $p = \lambda \cdot \Delta t$
- If triggered: pulse volume assigned

Two composition options:

1. Basaltic recharge (higher $T_{\text{infl}}$, lower $X_{\text{tot,infl}}$)
2. Rhyolitic recharge (higher $X_{\text{tot,infl}}$)

Mass & composition update:

$$\Delta M_{\text{in}} = \rho_l \cdot \Delta V_{\text{in}}$$

$$M' = M + \Delta M_{\text{in}}$$

$$X'_{\text{tot}} = \frac{M \cdot X_{\text{tot}} + \Delta M_{\text{in}} \cdot X_{\text{tot,infl}}}{M'}$$

$$T' = \frac{M \cdot T + \Delta M_{\text{in}} \cdot T_{\text{infl}}}{M'}$$


## 2.2.2 Elastic pressurisation

$$\Delta P = \frac{K_{\text{eff}} \cdot (V - V_0)}{V_0}$$

$$P = P_{\text{lith}} + \Delta P$$

---

## 2.3 Failure & eruption

### Stochastic failure criterion

Draw $\xi \sim \mathcal{N}(0, \sigma)$  

Erupt if:

$$P > P_{\text{lith}} + P_{\text{crit}} + \xi$$

### Eruption size (elastic-limited)

$$\Delta V_{\text{erupt}} = \frac{\Delta P_{\text{max}} \cdot V}{K_{\text{eff}}}$$

$$\Delta M_{\text{erupt}} = \rho_l \cdot \Delta V_{\text{erupt}}$$

### Update state after eruption

$$M = M - \Delta M_{\text{erupt}}$$

Recompute $v_g$, $v_l$, $v$, $V$

$$P = P_{\text{lith}} + \frac{K_{\text{eff}} \cdot (V - V_0)}{V_0}$$

Log event

---

## 3. Numerical scheme (Euler)

For each timestep:
1. Update $t$
2. Cooling → $T$
3. Crystallinity → $\phi$, $f_m$
4. Recharge → $M$, $X_{\text{tot}}$, $T$
5. Degassing → $X_{\text{gas}}$, $v$, $V$
6. Pressure → $P$
7. Check failure & erupt if needed
8. Store outputs

---

## 4. Outputs & scenarios

Time series:
- $t$, $P$, $T$, $\phi$, $X_{\text{gas}}$, $V$, $M$
- `eruption_log`

Example presets:
- Stromboli-like  
- Pinatubo-like  
- Yellowstone-like  
