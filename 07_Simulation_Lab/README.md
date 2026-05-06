# 07 Simulation Lab

This folder contains simulation experiments and adapters for validating the
main integration flow without live hardware.

## Current content

- `hybrid_controller_sim/`: simulation support around stable
  `hybrid_controller` interfaces.

## Boundary

- Simulation may depend on stable `hybrid_controller` APIs.
- `hybrid_controller` must not depend on simulation code for normal runtime.
- Simulation should not import private implementation from sibling numbered
  algorithm folders. If an algorithm behavior is needed, use a copied stable
  interface in `hybrid_controller` or a local fixture.
