# Soft Copyright UI References

This note records reference material used when designing
`08_SoftCopyright_UI`.

## Official Copyright Sources

- China Copyright Protection Center / China copyright registration service:
  `https://www.ccopyright.com.cn/`
- China copyright registration platform:
  `https://register.ccopyright.com.cn/registration.html#/index`
- Computer Software Protection Regulations:
  `https://www.gov.cn/gongbao/content/2002/content_61861.htm`
- Measures for Computer Software Copyright Registration:
  `https://www.gov.cn/zhengce/2002-02/20/content_5724627.htm`
- National Copyright Administration PDF for the registration measures:
  `https://www.ncac.gov.cn/xxfb/flfg/bmgz/202410/P020241015604759788122.pdf`

## Implemented V1.0 Workbench Boundary

- `08_SoftCopyright_UI` is a hardware-free PyQt workbench for software
  copyright screenshots, status review, and material preparation.
- Buttons in the workbench only open directories, locate files, or display
  commands. They do not launch real robot MOVE/PICK/PLACE actions and do not
  write runtime profiles.
- The workbench reads existing status from repository paths, profile files,
  schema files, and `docs/softcopyright/` material drafts.
- The MI classifier is represented through a thin contract until the new
  classifier is merged: training entry, realtime entry, and
  `datasets/profiles/MI/current_mi_profile.json`.

## Interface Rationale

- BCI online tools should expose live recognition state, confidence, idle or
  no-control state, and timing/false-positive indicators.
- Robot supervisory interfaces should expose safety gates, robot state,
  perception validity, and operator override controls.
- The V1.0 software-copyright UI therefore uses a workbench layout with
  acquisition, training, online control, vision/robot, and material-preparation
  pages.
