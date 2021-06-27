Settings = {}
Settings.PythonScript = "/home/arch/repos/python-funscript-editor/funscript-editor.py"
Settings.TmpFile = "/tmp/funscript_actions.csv"
SetSettings(Settings)

-- Version: 1.0.0
function GetActions(video)
    local at = {}
    local pos = {}
    local command = 'python3 "'..Settings.PythonScript..'" --generator -s '..tostring(CurrentTimeMs)..' -i "'..video..'" -o "'..Settings.TmpFile..'"'
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