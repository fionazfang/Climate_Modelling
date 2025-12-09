# v1 MAGMA CHAMBER MODEL – DEVELOPER SPEC

**Goal:**  
0D magma chamber model with two eruption triggers:
1. **Crystallisation-induced degassing**
2. **Magma recharge (basaltic or rhyolitic), with constant or Poisson influx**

Eruptions occur when a **stochastic overpressure threshold** is exceeded, and eruption size is **elastic-limited**. 

This spec defines all state variables, parameters, equations, and numerical workflow for v1.

---

## 1.1 State variables

**State Variables**
- `t` : time (s)
- `T` : magma temperature (K)
- `phi` : crystallinity (0–1)
- `P` : chamber pressure (Pa)
- `M` : total bulk magma mass (kg) in the chamber
- `X_tot` : **bulk** volatile mass fraction (kg volatiles / kg bulk)
- `V` : total chamber magma volume (m³) (diagnostic)
- `eruption_log`: list of eruption events (time, ΔM, ΔP)

**Diagnostic Variables**
- Melt fraction: `f_m = 1 - phi`
- Volatile concentration in melt: `C_m = X_tot / f_m`
- Solubility: `C_eq(P)`
- Exsolved gas fraction: `X_gas = max(0, X_tot - C_eq*f_m)`
- Gas and liquid-crystal volumes
- Overpressure: `ΔP = P - P_lith`

---

## 1.2 Fixed parameters

Physical parameters:
- `K_eff` : effective bulk modulus (Pa)
- `P_lith` : lithostatic pressure (Pa)
- `rho_l` : liquid+crystal density (kg/m³)
- `R_g` : gas constant (J kg⁻¹ K⁻¹)
- Solubility law: `C_eq = k * P^n`
- Crystallinity vs T (lever rule): between `T_liq` and `T_sol`
- Cooling law:  
  `dT/dt = -(T - T_host)/tau_T`

Initial state:
- `T0, phi0, P0 = P_lith, M0, X_tot0`
- Compute initial `V0 = M0 * v0`

---

## 2.1.1 Cooling & crystallisation

1. Euler cooling:  
   `T = T + (-(T - T_host)/tau_T) * dt`
2. Crystallinity from T (lever rule)
3. Melt fraction: `f_m = 1 - phi`

---

## 2.1.2 Crystallisation-induced degassing

1. `C_m = X_tot / f_m`
2. `C_eq = k * P^n`
3. `X_gas = max(0, X_tot - C_eq * f_m)`
4. Gas volume per kg: `v_g = X_gas * R_g * T / P`
5. Liquid-crystal volume per kg: `v_l = (1 - X_gas)/rho_l`
6. Specific volume: `v = v_l + v_g`
7. Chamber volume: `V = M * v`

---

## 2.2.1 Magma recharge

Two influx regimes:

### Constant influx
- Volume added: `ΔV_in = Q * dt`

### Poisson pulses
- Probability per step: `p = λ * dt`
- If triggered: pulse volume assigned

Composition options:
- Basaltic recharge (higher T_infl, lower X_tot_infl)
- Rhyolitic recharge (higher X_tot_infl)

Mass & composition update:
```
ΔM_in = rho_l * ΔV_in
M' = M + ΔM_in
X_tot' = (M*X_tot + ΔM_in * X_tot_infl) / M'
T' = (M*T + ΔM_in*T_infl) / M'
```

---

## 2.2.2 Elastic pressurisation

`ΔP = K_eff * (V - V0) / V0`  
`P = P_lith + ΔP`

---

## 2.3 Failure & eruption

### Stochastic failure criterion
Draw `xi ~ N(0, sigma)`  
Erupt if:
`P > P_lith + P_crit + xi`

### Eruption size (elastic-limited)
`ΔV_erupt = (ΔP_max * V) / K_eff`  
`ΔM_erupt = rho_l * ΔV_erupt`

### Update state after eruption
```
M = M - ΔM_erupt
Recompute v_g, v_l, v, V
P = P_lith + K_eff * (V - V0)/V0
Log event
```

---

## 3. Numerical scheme (Euler)

For each timestep:
```
1) Update t
2) Cooling -> T
3) Crystallinity -> phi, f_m
4) Recharge -> M, X_tot, T
5) Degassing -> X_gas, v, V
6) Pressure -> P
7) Check failure & erupt if needed
8) Store outputs
```

---

## 4. Outputs & scenarios

Time series:
- `t, P, T, phi, X_gas, V, M`
- `eruption_log`

Example presets:
- Stromboli-like  
- Pinatubo-like  
- Yellowstone-like  
