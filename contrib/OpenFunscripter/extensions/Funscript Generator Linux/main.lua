-- Version: 0.0.1
configFile = ofs.ExtensionDir() .. "/config"
pythonScript = ""


function funscript_generator()
    local tmpFile = ofs.ExtensionDir() .. "/funscript_actions.csv"
    local video = player.CurrentVideo()
    local script = ofs.Script(ofs.ActiveIdx())
    local currentTimeMs = player.CurrentTime() * 1000

    print("tmpFile: ", tmpFile)
    print("video: ", video)
    print("currentScriptIdx: ", ofs.ActiveIdx())
    print("currentTimeMs: ", currentTimeMs)

    local next_action = ofs.ClosestActionAfter(script, currentTimeMs / 1000)
    if next_action and script.actions[next_action].at < (currentTimeMs + 500) then
        next_action = ofs.ClosestActionAfter(script, script.actions[next_action].at / 1000)
    end

    if next_action then
        print("nextAction: ", script.actions[next_action].at)
    else
        print("nextAction: nil")
    end
    local command = 'python3 "'
            ..pythonScript
            ..'" --generator -s '
            ..( next_action == nil and tostring(currentTimeMs) or tostring(currentTimeMs)..' -e '..tostring(script.actions[next_action].at) )
            ..' -i "'
            ..video
            ..'" -o "'
            ..tmpFile
            ..'"'

    print(command)
    os.execute(command)
    -- ofs.SilentCmd(command, false)

    local f = io.open(tmpFile)
    if not f then
        print('lua: funscript generator output file not found')
        return
    end

    script = ofs.Script(ofs.ActiveIdx())
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


function load_config()
    print("try load config: ", configFile)

    local f = io.open(configFile)
    if not f then
        print('extension config file not found')
        return
    end

    for line in f:lines() do
        for k, v in string.gmatch(line, "([^=]+)=(([^=]+))") do
            if k == "pythonScript" then
                print("set pythonScript to", v)
                pythonScript = v
            end
        end
    end

    f:close()
end


function save_config()
    -- print("save config to: ", configFile)
    local f = io.open(configFile, "w")
    f:write("pythonScript="..pythonScript)
    f:close()
end


function init()
    load_config()
    ofs.Bind("funscript_generator", "execute the funcript generator")
end


function update(delta)

end


function gui()
    pythonScript, valueChanged = ofs.Input("pythonScript", pythonScript)
    if valueChanged then
        save_config()
    end
end
