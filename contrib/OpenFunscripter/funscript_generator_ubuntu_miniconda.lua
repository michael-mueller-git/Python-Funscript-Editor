Settings = {}
Settings.PythonScript = "/home/ubuntu/Repos/Python-Funscript-Editor/funscript-editor.py"
Settings.TmpFile = "/tmp/funscript_actions.csv"
Settings.CondaEnvName = "funscript-editor"
SetSettings(Settings)

-- Version: 1.4.0
-- Note: I have not test this script!
function GetActions(video)
    local at = {}
    local pos = {}
    local next_action = CurrentScript:GetClosestActionAfter(CurrentTimeMs)
    if next_action and next_action.at < CurrentTimeMs + 500.0 then
        next_action = CurrentScript:GetClosestActionAfter(next_action.at)
    end
    local command = 'bash -i -c \'conda activate '..Settings.CondaEnvName..'; python3 "'..Settings.PythonScript..'" --generator -s '..(next_action == nil and tostring(CurrentTimeMs) or tostring(CurrentTimeMs)..' -e '..tostring(next_action.at))..' -i "'..video..'" -o "'..Settings.TmpFile..'"\''
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