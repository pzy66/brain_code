# JetMax Wi-Fi References

This folder records PC-side networking references used while debugging the
JetMax/Hiwonder Wi-Fi disconnect issue. These references are not robot-camera
sender references and do not change the locked camera chain.

## Sources Used

- Microsoft Learn: Network Connectivity Status Indicator overview for Windows
  - URL: https://learn.microsoft.com/en-us/windows-server/networking/ncsi/ncsi-overview
  - Used for: interpreting Windows NCSI active/passive probes and local-only vs Internet connectivity status.
- Microsoft Learn: Network Connection Status Indicator troubleshooting guidance
  - URL: https://learn.microsoft.com/en-us/troubleshoot/windows-server/networking/troubleshoot-ncsi-guidance
  - Used for: checking how Windows evaluates network connectivity and why NCSI results affect applications/services.
- Intel: Advanced Intel Wireless Adapter Settings
  - URL: https://www.intel.com/content/www/us/en/support/articles/000005585/wireless/legacy-intel-wireless-products.html
  - Used for: interpreting Roaming Aggressiveness, MIMO Power Save, Packet Coalescing, preferred band, and transmit power.
- Intel: Wi-Fi Roaming Aggressiveness Setting
  - URL: https://www.intel.com/content/www/us/en/support/articles/000005546/wireless/legacy-intel-wireless-products.html
  - Used for: explaining why low roaming aggressiveness can reduce unnecessary AP scanning during fixed robot debugging.

## Current Project Interpretation

The JetMax AP is a robot-local network. The PC only needs the robot subnet
route, for example `192.168.149.0/24`, to reach control/video services.
Internet default routing and public DNS should normally stay on Ethernet or
another Internet-facing adapter.

When Windows assigns the JetMax Wi-Fi a default route and DNS server
`192.168.149.1`, NCSI can classify that interface as local-only/limited. On
this machine, Windows WLAN AutoConfig also logged limited-connectivity recovery
and driver-initiated disconnects. That evidence points to a PC-side
Windows/Intel Wi-Fi recovery path, not to the JetMax official camera sender.

## Diagnostic Tool

Use this read-only tool first:

```powershell
python -m hybrid_controller.tools.diagnose_jetmax_wifi_windows
```

The tool reads Windows adapter, route, DNS, advanced property, and event-log
state. It does not ping the robot, SSH to the robot, scan ports, pull camera
video, or move the arm.
