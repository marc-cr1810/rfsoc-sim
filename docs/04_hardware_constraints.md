# 4. Hardware Constraints

The Xilinx Zynq UltraScale+ RFSoC (specifically the ZU48DR silicon used on the ZCU208) is not infinitely flexible. It has strict clocking, structural, and silicon constraints. The simulator actively enforces these constraints to ensure that configurations generated in the UI are physically achievable in hardware (via `src/rfdc.rs`).

## 1. PLL and Clocking Limits

The RF-ADC requires an extremely stable, high-speed clock to operate (up to $5.0 \text{ GHz}$ for Gen3). The ZCU208 can provide this clock externally, or it can be derived internally using the RFSoC's on-chip PLLs.

When the **Internal PLL** is enabled, the simulator models the hardware Phase-Locked Loop:
- **Reference Clock:** Must be provided (typically $100 \text{ MHz}$ to $500 \text{ MHz}$).
- **VCO Bounds:** The internal Voltage Controlled Oscillator must operate within a specific frequency range.
- **Multiplier/Divider Bounds:** The `Reference Clock` $\times$ `Multiplier` must equal the desired `Tile Sample Rate`.

The simulator's `validate_pll()` function enforces that a valid integer multiplier exists to bridge the reference clock to the target sample rate. If no valid multiplier exists, the UI throws a validation error, exactly as the Xilinx XRFdc API would during initialization.

## 2. AXI4-Stream Fabric Limits

Once the signal is sampled and downconverted, it is passed from the RF-ADC hard block to the Programmable Logic (PL) across an AXI4-Stream interface.

The PL in a Zynq UltraScale+ device has a maximum clock speed constraint (typically around $500 \text{ MHz}$). The clock speed driving the AXI4-Stream interface is determined by:
$$ \text{AXI Clock} = \frac{\text{Output Sample Rate}}{\text{AXI Words per Clock}} $$

- **Output Sample Rate:** The final sample rate after Decimation.
- **AXI Words per Clock:** The width of the AXI bus (e.g., 1, 2, 4, 8, or 16 samples per clock cycle).

If the resulting AXI Clock exceeds $500 \text{ MHz}$, the design will fail physical timing closure in Vivado. The simulator computes this in real-time. If the AXI clock limit is violated, the user is required to either:
1. Increase the Decimation Factor (lowering the Output Sample Rate).
2. Increase the AXI Words/Clock (widening the bus to lower the clock frequency).

## 3. Tile and Block Hierarchy

The ZU48DR architecture groups ADCs into **Tiles**, each containing multiple **Blocks**.
- **Tile:** Contains the PLL and the master clock distribution. All blocks inside a Tile share the exact same `Sample Rate`.
- **Block:** Contains the individual ADC, DDC mixer, NCO, and decimation filters. Each block can have independent mixer frequencies and decimation factors, but their initial input $F_s$ is locked to the Tile.

The simulator mimics this hierarchy accurately. Changing the Sample Rate forces a re-computation of every block inside that tile, enforcing identical time-domain sampling parameters across parallel channels.

> [!WARNING]
> **Nyquist Zone Grouping:** In real hardware, the Nyquist Zone is a hardware register that configures the analog input bandwidth. The simulator enforces this at the Block level, reflecting the underlying `XRFdc_SetNyquistZone` API behavior.
