
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
        return
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

    -- delete temporary file
    os.remove(tmpFile)
end


function invert_selected()
    local script = ofs.Script(ofs.ActiveIdx())
    for idx, action in ipairs(script.actions) do
        if action.selected then
            action.pos = clamp(100 - action.pos, 0, 100)
        end
    end
    ofs.Commit(script)
end


function align_bottom_points(align_value)
    local script = ofs.Script(ofs.ActiveIdx())

    -- get min value in selection if no align_value was given
    if align_value < 0 then
        align_value = 99
        for idx, action in ipairs(script.actions) do
            if action.selected then
                if action.pos < align_value then
                    align_value = action.pos
                end
            end
        end
    end

    -- align bottom points
    for idx, action in ipairs(script.actions) do
        if action.selected then
            local bottom_point = true

            local next_action = ofs.ClosestActionAfter(script, action.at / 1000)
            if next_action then
                if script.actions[next_action].pos <= action.pos then
                    bottom_point = false
                end
            end

            local prev_action = ofs.ClosestActionBefore(script, action.at / 1000)
            if prev_action then
                if script.actions[prev_action].pos <= action.pos then
                    bottom_point = false
                end
            end

            if bottom_point then
                action.pos = align_value
            end
        end
    end

    ofs.Commit(script)
end


function align_top_points(align_value)
    local script = ofs.Script(ofs.ActiveIdx())

    -- get max value in selection if no align_value was given
    if align_value < 0 then
        align_value = 1
        for idx, action in ipairs(script.actions) do
            if action.selected then
                if action.pos > align_value then
                    align_value = action.pos
                end
            end
        end
    end

    -- align top points
    for idx, action in ipairs(script.actions) do
        if action.selected then
            local top_point = true

            local next_action = ofs.ClosestActionAfter(script, action.at / 1000)
            if next_action then
                if script.actions[next_action].pos >= action.pos then
                    top_point = false
                end
            end

            local prev_action = ofs.ClosestActionBefore(script, action.at / 1000)
            if prev_action then
                if script.actions[prev_action].pos >= action.pos then
                    top_point = false
                end
            end

            if top_point then
                action.pos = align_value
            end
        end
    end

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

    -- if logfileExist then
    --       ofs.SameLine()
    --       if ofs.Button("Open Log") then
    --          processHandleLogFile = ofs.CreateProcess("notepad.exe", "C:/Temp/funscript_editor.log")
    --       end
    -- end

    ofs.Separator()
    ofs.Text("Post-Processing:")
    ofs.SameLine()
    if ofs.Button("Invert") then
        invert_selected()
    end

    ofs.SameLine()
    if ofs.Button("Align Bottom Points") then
        align_bottom_points(-1)
    end

    ofs.SameLine()
    if ofs.Button("Align Top Points") then
        align_top_points(-1)
    end

end
