param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$SdkName = "",
    [string]$PythonExe = ""
)

$projectRootPath = (Resolve-Path -LiteralPath $ProjectRoot).Path
$ideaDir = Join-Path $projectRootPath ".idea"
$moduleFile = Join-Path $ideaDir "brain_code.iml"
$modulesFile = Join-Path $ideaDir "modules.xml"
$miscFile = Join-Path $ideaDir "misc.xml"
$nameFile = Join-Path $ideaDir ".name"

if (-not $PythonExe) {
    $resolveScript = Join-Path $PSScriptRoot "resolve_brain_python.ps1"
    if (-not (Test-Path -LiteralPath $resolveScript)) {
        Write-Error "[configure-pycharm] Missing interpreter resolver: $resolveScript"
        exit 1
    }
    $PythonExe = (& $resolveScript | Select-Object -Last 1)
}

$pythonExe = [string]$PythonExe
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Error "[configure-pycharm] Missing interpreter: $pythonExe"
    exit 1
}

$pythonVersionJson = & $pythonExe -c "import json, sys; print(json.dumps({'major': sys.version_info.major, 'minor': sys.version_info.minor, 'micro': sys.version_info.micro}))"
if ($LASTEXITCODE -ne 0) {
    Write-Error "[configure-pycharm] Failed to inspect interpreter version: $pythonExe"
    exit 1
}
$pythonVersion = $pythonVersionJson | ConvertFrom-Json
$pythonSdkVersion = "Python $($pythonVersion.major).$($pythonVersion.minor).$($pythonVersion.micro)"
if (-not $SdkName) {
    $sdkPrefix = if ($pythonExe -match 'brain-vision') { "brain-vision" } elseif ($pythonExe -match '\\\.venv\\') { "brain_code-venv" } else { "brain-python" }
    $SdkName = "$sdkPrefix-$($pythonVersion.major)$($pythonVersion.minor)"
}

function New-XmlElement {
    param(
        [xml]$Document,
        [string]$Name,
        [hashtable]$Attributes = @{}
    )

    $element = $Document.CreateElement($Name)
    foreach ($key in $Attributes.Keys) {
        $attribute = $Document.CreateAttribute($key)
        $attribute.Value = [string]$Attributes[$key]
        [void]$element.Attributes.Append($attribute)
    }
    return $element
}

function Register-PyCharmSdk {
    param(
        [string]$ProjectRootPath,
        [string]$InterpreterPath,
        [string]$SdkDisplayName,
        [string]$SdkVersion
    )

    $roamingJetBrains = Join-Path $env:APPDATA "JetBrains"
    if (-not (Test-Path -LiteralPath $roamingJetBrains)) {
        Write-Warning "[configure-pycharm] JetBrains roaming config not found: $roamingJetBrains"
        return @()
    }

    $configDirs = Get-ChildItem -LiteralPath $roamingJetBrains -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "PyCharm*" }

    $updatedTables = @()
    foreach ($configDir in $configDirs) {
        $optionsDir = Join-Path $configDir.FullName "options"
        if (-not (Test-Path -LiteralPath $optionsDir)) {
            continue
        }

        $tablePath = Join-Path $optionsDir "jdk.table.xml"
        if (-not (Test-Path -LiteralPath $tablePath)) {
            [System.IO.File]::WriteAllText(
                $tablePath,
                "<application><component name=`"ProjectJdkTable`" /></application>",
                [System.Text.UTF8Encoding]::new($false)
            )
        }

        [xml]$xml = Get-Content -LiteralPath $tablePath -Raw
        $application = $xml.SelectSingleNode("/application")
        if (-not $application) {
            $application = $xml.AppendChild($xml.CreateElement("application"))
        }

        $component = $xml.SelectSingleNode("/application/component[@name='ProjectJdkTable']")
        if (-not $component) {
            $component = $application.AppendChild((New-XmlElement -Document $xml -Name "component" -Attributes @{ name = "ProjectJdkTable" }))
        }

        $staleNodes = @($component.SelectNodes("jdk")) | Where-Object {
            $nameNode = $_.SelectSingleNode("name")
            $homeNode = $_.SelectSingleNode("homePath")
            ($nameNode -and $nameNode.GetAttribute("value") -eq $SdkDisplayName) -or
            ($homeNode -and $homeNode.GetAttribute("value") -eq $InterpreterPath)
        }
        foreach ($node in $staleNodes) {
            [void]$component.RemoveChild($node)
        }

        $venvRoot = Split-Path -Parent $InterpreterPath
        $jdk = New-XmlElement -Document $xml -Name "jdk" -Attributes @{ version = "2" }
        [void]$jdk.AppendChild((New-XmlElement -Document $xml -Name "name" -Attributes @{ value = $SdkDisplayName }))
        [void]$jdk.AppendChild((New-XmlElement -Document $xml -Name "type" -Attributes @{ value = "Python SDK" }))
        [void]$jdk.AppendChild((New-XmlElement -Document $xml -Name "version" -Attributes @{ value = $SdkVersion }))
        [void]$jdk.AppendChild((New-XmlElement -Document $xml -Name "homePath" -Attributes @{ value = $InterpreterPath }))

        $roots = New-XmlElement -Document $xml -Name "roots"
        $classPath = New-XmlElement -Document $xml -Name "classPath"
        $classComposite = New-XmlElement -Document $xml -Name "root" -Attributes @{ type = "composite" }
        $rootCandidates = @(
            (Join-Path $venvRoot "DLLs"),
            (Join-Path $venvRoot "Lib"),
            (Join-Path $venvRoot "Lib\site-packages"),
            (Join-Path $venvRoot "Scripts")
        )
        foreach ($rootPath in $rootCandidates) {
            if (Test-Path -LiteralPath $rootPath) {
                $url = "file://" + ($rootPath -replace "\\", "/")
                [void]$classComposite.AppendChild((New-XmlElement -Document $xml -Name "root" -Attributes @{ url = $url; type = "simple" }))
            }
        }
        [void]$classComposite.AppendChild(
            (New-XmlElement -Document $xml -Name "root" -Attributes @{
                url = 'file://$APPLICATION_HOME_DIR$/plugins/python-ce/helpers/typeshed/stdlib'
                type = "simple"
            })
        )
        [void]$classPath.AppendChild($classComposite)
        [void]$roots.AppendChild($classPath)

        $sourcePath = New-XmlElement -Document $xml -Name "sourcePath"
        [void]$sourcePath.AppendChild((New-XmlElement -Document $xml -Name "root" -Attributes @{ type = "composite" }))
        [void]$roots.AppendChild($sourcePath)
        [void]$jdk.AppendChild($roots)

        $additional = New-XmlElement -Document $xml -Name "additional" -Attributes @{
            ASSOCIATED_PROJECT_PATH = $ProjectRootPath
            SDK_UUID = [guid]::NewGuid().ToString()
        }
        [void]$jdk.AppendChild($additional)

        [void]$component.AppendChild($jdk)
        $xml.Save($tablePath)
        $updatedTables += $tablePath
    }

    return $updatedTables
}

New-Item -ItemType Directory -Force -Path $ideaDir | Out-Null

$miscXml = @"
<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component
    name="ProjectRootManager"
    version="2"
    project-jdk-name="$SdkName"
    project-jdk-type="Python SDK"
  />
</project>
"@

$moduleXml = @"
<?xml version="1.0" encoding="UTF-8"?>
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager">
    <content url="file://`$MODULE_DIR$/..">
      <excludeFolder url="file://`$MODULE_DIR$/../.git" />
      <excludeFolder url="file://`$MODULE_DIR$/../.venv" />
      <excludeFolder url="file://`$MODULE_DIR$/../.pytest_cache" />
      <excludeFolder url="file://`$MODULE_DIR$/../.mypy_cache" />
      <excludeFolder url="file://`$MODULE_DIR$/../.ruff_cache" />
    </content>
    <orderEntry type="jdk" jdkName="$SdkName" jdkType="Python SDK" />
    <orderEntry type="sourceFolder" forTests="false" />
  </component>
</module>
"@

$modulesXml = @"
<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectModuleManager">
    <modules>
      <module fileurl="file://`$PROJECT_DIR$/.idea/brain_code.iml" filepath="`$PROJECT_DIR$/.idea/brain_code.iml" />
    </modules>
  </component>
</project>
"@

[System.IO.File]::WriteAllText($miscFile, $miscXml, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText($moduleFile, $moduleXml, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText($modulesFile, $modulesXml, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText($nameFile, "brain_code`n", [System.Text.UTF8Encoding]::new($false))

$updatedTables = Register-PyCharmSdk `
    -ProjectRootPath $projectRootPath `
    -InterpreterPath $pythonExe `
    -SdkDisplayName $SdkName `
    -SdkVersion $pythonSdkVersion

Write-Output "[configure-pycharm] ProjectRoot = $projectRootPath"
Write-Output "[configure-pycharm] Interpreter = $pythonExe"
Write-Output "[configure-pycharm] SDK Name = $SdkName"
Write-Output "[configure-pycharm] SDK Version = $pythonSdkVersion"
foreach ($tablePath in $updatedTables) {
    Write-Output "[configure-pycharm] Updated SDK table: $tablePath"
}
Write-Output "[configure-pycharm] Updated .idea project files."
