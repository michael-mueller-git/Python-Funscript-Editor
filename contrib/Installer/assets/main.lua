
processHandleMTFG = nil
processHandleConfigDir = nil
processHandleLogFile = nil
logfileExist = false
updateCounter = 0
scriptIdx = 0
mtfgVersion = "0.0.0"
status = "MTFG not running"

function start_funscript_generator()
    if processHandleMTFG then
        print('MTFG already running')
    end

    scriptIdx = ofs.ActiveIdx()
    local tmpFile = ofs.ExtensionDir() .. "/funscript_actions.csv"
    local video = player.CurrentVideo()
    local script = ofs.Script(scriptIdx)
    local currentTimeMs = player.CurrentTime() * 1000
    local cmd = ofs.ExtensionDir() .. "/funscript-editor/funscript-editor.exe"

    print("cmd: ", cmd)
    print("tmpFile: ", tmpFile)
    print("video: ", video)
    print("currentScriptIdx: ", scriptIdx)
    print("currentTimeMs: ", currentTimeMs)

    local next_action = ofs.ClosestActionAfter(script, currentTimeMs / 1000)
    if next_action and script.actions[next_action].at < (currentTimeMs + 500) then
        next_action = ofs.ClosestActionAfter(script, script.actions[next_action].at / 1000)
    end

    print("nextAction: ", next_action and tostring(script.actions[next_action].at) or "nil")

   if next_action then
        processHandleMTFG = ofs.CreateProcess(
            cmd, "--generator",
            "-s", tostring(currentTimeMs),
            "-e", tostring(script.actions[next_action].at),
            "-i", video,
            "-o", tmpFile
        )
    else
        processHandleMTFG = ofs.CreateProcess(
            cmd, "--generator",
            "-s", tostring(currentTimeMs),
            "-i", video,
            "-o", tmpFile
        )
    end
    status = "MTFG running"
end


function import_funscript_generator_result()
    status = "MTFG not running"
    local tmpFile = ofs.ExtensionDir() .. "/funscript_actions.csv"
    local f = io.open(tmpFile)
    if not f then
        print('Funscript Generator output file not found')
        return
    end

    script = ofs.Script(scriptIdx)
    local k = 1
    for line in f:lines() do
        -- first line is header
        if k > 1 then
            for at, pos in string.gmatch(line, "(%w+);(%w+)") do
                ofs.AddAction(script, at, pos, true)
            end
        end
        k = k + 1
    end
    f:close()

    -- save changes
    ofs.Commit(script)
end


function init()
    ofs.Bind("start_funscript_generator", "execute the funcript generator")
    local f = io.open(ofs.ExtensionDir().."\\funscript-editor\\funscript_editor\\VERSION.txt")
    if f then
        for line in f:lines() do
            if string.find(string.lower(line), "v") then
                mtfgVersion = string.lower(line):gsub("v", "")
            end
        end
    end
end


function update(delta)
    updateCounter = updateCounter + 1
    if processHandleMTFG and not ofs.IsProcessAlive(processHandleMTFG) then
        print('funscript generator completed import result')
        processHandleMTFG = nil
        import_funscript_generator_result()
    end
    if math.fmod(updateCounter, 1000) then
        local f = io.open("C:/Temp/funscript_editor.log")
        if f then
            logfileExist = true
        else
            logfileExist = false
        end
    end
end


function gui()
    ofs.Text("Status: "..status)
    ofs.Text("Version: "..mtfgVersion)
    ofs.Text("Action:")

    ofs.SameLine()
    if not processHandleMTFG then
        
        if ofs.Button("Start MTFG") then
            start_funscript_generator()
        end
    else
        if ofs.Button("Kill MTFG") then
            os.execute("taskkill /f /im funscript-editor.exe")
        end
    end

    ofs.SameLine()
    if ofs.Button("Open Config") then
        processHandleConfigDir = ofs.CreateProcess("explorer.exe", ofs.ExtensionDir().."\\funscript-editor\\funscript_editor\\config")
    end

    if logfileExist then
          ofs.SameLine()
          if ofs.Button("Open Log") then
             processHandleLogFile = ofs.CreateProcess("notepad.exe", "C:/Temp/funscript_editor.log")
          end
    end

    
end
