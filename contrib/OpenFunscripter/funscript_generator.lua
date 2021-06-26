Settings = {}
Settings.FunscriptGenerator = 'C:/Users/win10/Desktop/funscript editor/funscript-editor.exe' -- Path to your Python Funscript Editor installation
Settings.TmpFile = 'C:/Users/win10/AppData/Local/Temp/funscript_actions.csv' -- file where to temporary store the result (must be a file not a directory!)
SetSettings(Settings)

-- Version: 1.1.0
function GetActions(video)
    local at = {}
    local pos = {}
    local command = '""'..Settings.FunscriptGenerator..'" --generator -s '..tostring(CurrentTimeMs)..' -i "'..video..'" -o "'..Settings.TmpFile..'"'
    print(command)
    os.execute(command)
    local f = io.open(Settings.TmpFile)
    if not f then
        print('lua: funscript generator output file not found')
        return at, pos
    end
    local k = 1
    for line in f:lines() do
        -- first line is header
        if k > 1 then
            for k, v in string.gmatch(line, "(%w+);(%w+)") do
                at[#at+1] = k
                pos[#pos+1] = v
            end
        end
        k = k + 1
    end
    f:close()
    return at, pos
end

print('start funscript generator')
local video = VideoFilePath -- NOTE: do not use special character in the video path!
print('video file: ', video)
local actions = {GetActions(video)}
for i = 1, #actions[1] do
    CurrentScript:AddActionUnordered(tonumber(actions[1][i]), tonumber(actions[2][i]), true, 0)
end
print('done')
