# Vendored Sources

`hybrid_controller` is the final integration runtime. It should not import
private code directly from sibling numbered folders. When a subsystem has been
debugged and stabilized elsewhere, copy only the required runtime code here and
record the copy in this file.

## Entry format

```text
Date:
Purpose:
Source:
Destination:
Source revision:
Copied behavior:
Omitted behavior:
Validation:
Owner notes:
```

## Current baseline

No new copied code is recorded by this manifest yet. Existing local packages
such as `hybrid_controller/vision`, `hybrid_controller/robot`, and
`hybrid_controller/ssvep` are treated as the current integration baseline until
future promotions are explicitly logged here.

The next expected promotion candidate is the recognition-to-grasp debug flow
after validation in:

```text
04_Communication_And_Integration/2026-05_vision_grasp_flow_debug/
```
