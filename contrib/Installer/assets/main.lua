
configFile = ofs.ExtensionDir() .. "/config"
pythonFunscriptGenerator = ofs.ExtensionDir .. "/funscript-editor/funscript-editor.exe"

function funscript_generator()
    local tmpFile = ofs.ExtensionDir() .. "/funscript_actions.csv"
    local video = player.CurrentVideo()
    local script = ofs.Script(ofs.ActiveIdx())
    local currentTimeMs = player.CurrentTime() * 1000

    print("funscriptGenerator", pythonFunscriptGenerator)
    print("tmpFile: ", tmpFile)
    print("video: ", video)
    print("currentScriptIdx: ", ofs.ActiveIdx())
    print("currentTimeMs: ", currentTimeMs)

    local next_action = ofs.ClosestActionAfter(script, currentTimeMs)
    if next_action and next_action.at < currentTimeMs + 500.0 then
        next_action = ofs.ClosestActionAfter(script, next_action.at)
    end

    if next_action then
        print("nextAction: ", next_action.at) -- TODO is this in seconds?
    else
        print("nextAction: nil")
    end

    local command = '"'
            ..pythonFunscriptGenerator
            ..'" --generator -s '
            ..( next_action == nil and tostring(currentTimeMs) or tostring(currentTimeMs)..' -e '..tostring(next_action.at) )
            ..' -i "'
            ..video
            ..'" -o "'
            ..tmpFile
            ..'"'


    print(command)
    ofs.SilentCmd(command, false)

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



function init()
    ofs.Bind("funscript_generator", "execute the funcript generator")
end


function update(delta)

end


function gui()

end
