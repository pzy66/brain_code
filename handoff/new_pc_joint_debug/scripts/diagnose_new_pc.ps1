param(
    [string]$RobotHost = "192.168.149.1"
)

$ErrorActionPreference = "Continue"

function Test-Port {
    param(
        [string]$HostName,
        [int]$Port,
        [string]$Name
    )
    $result = Test-NetConnection -ComputerName $HostName -Port $Port -WarningAction SilentlyContinue
    [pscustomobject]@{
        Check = $Name
        Target = "$HostName`:$Port"
        OK = [bool]$result.TcpTestSucceeded
        Detail = if ($result.TcpTestSucceeded) { "open" } else { "closed or blocked" }
    }
}

Write-Host "=== BrainRobotWorkbench new PC diagnostics ==="
Write-Host "Robot host: $RobotHost"
Write-Host ""

$pingOk = Test-Connection -ComputerName $RobotHost -Count 2 -Quiet -ErrorAction SilentlyContinue
[pscustomobject]@{
    Check = "ping"
    Target = $RobotHost
    OK = [bool]$pingOk
    Detail = if ($pingOk) { "reachable" } else { "not reachable" }
} | Format-Table -AutoSize

@(
    (Test-Port -HostName $RobotHost -Port 22 -Name "SSH"),
    (Test-Port -HostName $RobotHost -Port 9091 -Name "ROS rosbridge"),
    (Test-Port -HostName $RobotHost -Port 8080 -Name "Camera web_video_server"),
    (Test-Port -HostName $RobotHost -Port 8888 -Name "Legacy TCP optional")
) | Format-Table -AutoSize

Write-Host ""
Write-Host "=== Serial ports ==="
try {
    Get-CimInstance Win32_SerialPort |
        Select-Object DeviceID, Name, Description, Manufacturer |
        Format-Table -AutoSize
} catch {
    Write-Host "Cannot query Win32_SerialPort: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "=== Network adapters with IPv4 ==="
try {
    Get-NetIPConfiguration |
        Where-Object { $_.IPv4Address } |
        Select-Object InterfaceAlias, InterfaceDescription, @{Name="IPv4";Expression={$_.IPv4Address.IPAddress}}, @{Name="Gateway";Expression={$_.IPv4DefaultGateway.NextHop}} |
        Format-Table -AutoSize
} catch {
    Write-Host "Cannot query network adapters: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "If SSH is open but rosbridge 9091 is closed, run on the robot:"
Write-Host "  cd /home/hiwonder/brain_code/hybrid_controller/robot"
Write-Host "  bash run_hybrid_controller_ros_runtime.sh"
