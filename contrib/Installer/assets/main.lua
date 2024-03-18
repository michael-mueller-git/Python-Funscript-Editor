json = require "json"
-- MTFG LUA Wrappper Version 2.0.4

-- global var
processHandleMTFG = nil
processHandleConfigDir = nil
processHandleLogFile = nil
logfileExist = false
updateCounter = 0
scriptIdx = 1
mtfgVersion = "0.0.0"
status = "MTFG not running"
multiaxis = false
tmpFileName = "funscript_actions.json"
tmpFileExists = false
enableLogs = false
stopAtNextActionPoint = true
filterSimilarTimestamps = true
scriptNames = {}
scriptNamesCount = 0
scriptAssignment = {x={idx=1}, y={idx=1}, distance={idx=1}, roll={idx=1}}

function exists(file)
   return os.rename(file, file)
end

function get_platform()
    if ofs.ExtensionDir():find("^/home/") ~= nil then
        local home = os.getenv( "HOME" )
        print("User Home: ", home)
        if exists("/nix/store") then
            return "Linux, Nix"
        elseif exists(home.."/miniconda3/envs/funscript-editor") then
            return "Linux, Conda"
        elseif exists(home.."/anaconda3/envs/funscript-editor") then
            return "Linux, Conda"
        else
            return "Linux, Python"
        end
    else
        return "Windows"
    end
end

platform = get_platform()

function binding.start_funscript_generator()
    exec_mtfg(false)
end

function binding.gainplus()
    gain_selected(1.05)
end

function binding.gainmin()
    gain_selected(0.95)
end

function binding.continueMinMax()
    continue_minmax(-1)
end

function binding.dTrend_0()
    detrend_selected(1,4,0)
end

function binding.norm_3()
    normalize_selection(3,50)
end

function binding.norm_7()
    normalize_selection(7,50)
end

function binding.norm_15()
    normalize_selection(15,50)
end

function exec_mtfg(no_tracking)
    if processHandleMTFG then
        print('MTFG already running')
        return
    end

    scriptIdx = ofs.ActiveIdx()
    local tmpFile = ofs.ExtensionDir() .. "/" .. tmpFileName
    local video = player.CurrentVideo()
    local script = ofs.Script(scriptIdx)
    local currentTime = player.CurrentTime()
    local fps = player.FPS()

    local next_action = nil
    if stopAtNextActionPoint then
        next_action, _ = script:closestActionAfter(currentTime)
        if next_action and next_action.at < (currentTime + 0.5) then
            next_action, _ = script:closestActionAfter(next_action.at)
        end
    end

    print("tmpFile: ", tmpFile)
    print("video: ", video)
    print("fps", fps)
    print("currentScriptIdx: ", scriptIdx)
    print("currentTime: ", currentTime)
    print("nextAction: ", next_action and tostring(next_action.at) or "nil")

    local cmd = ""
    local args = {}

    if platform == "Windows" then
        cmd = ofs.ExtensionDir() .. "/funscript-editor/funscript-editor.exe"
    elseif platform == "Linux, Python" then
        cmd = "/usr/bin/python3"
        table.insert(args, ofs.ExtensionDir() .. "/Python-Funscript-Editor/funscript-editor.py")
    elseif platform == "Linux, Conda" then
        cmd = "/usr/bin/bash"
        table.insert(args, ofs.ExtensionDir() .. "/Python-Funscript-Editor/conda_wrapper.sh")
    elseif platform == "Linux, Nix" then
        if exists(ofs.ExtensionDir() .. "/Python-Funscript-Editor/nix_wrapper.sh") then
            print("nix: use local repository")
            os.execute("chmod +x \"" .. ofs.ExtensionDir() .. "/Python-Funscript-Editor/nix_wrapper.sh" .. "\"")
            cmd = ofs.ExtensionDir() .. "/Python-Funscript-Editor/nix_wrapper.sh"
        else
            print("nix: use remote repository")
            file = io.open("/tmp/nix-mtfg.sh", "w")
            file:write("#!/usr/bin/env bash\n")
            file:write("unset LD_LIBRARY_PATH\n")
            file:write("nix run github:michael-mueller-git/Python-Funscript-Editor --refresh -- \"$@\"\n")
            file:close()
    
            os.execute("chmod +x /tmp/nix-mtfg.sh")
        
            cmd = "/tmp/nix-mtfg.sh"
        end
    else
        print("ERROR: Platform Not Implemented (", platform, ")")
    end

    table.insert(args, "--generator")

    if multiaxis then;
        table.insert(args, "--multiaxis")
    end

    if enableLogs then;
        table.insert(args, "--logs")
    end

    table.insert(args, "-s")
    table.insert(args, tostring(math.floor(currentTime*1000)))
    table.insert(args, "-i")
    table.insert(args, video)
    table.insert(args, "-o")
    table.insert(args, tmpFile)

    if next_action then
        table.insert(args, "-e")
        table.insert(args, tostring(math.floor(next_action.at*1000.0)))
    end

    if no_tracking then
        table.insert(args, "--no-tracking")
    end

    print("cmd: ", cmd)
    print("args: ", table.unpack(args))

    processHandleMTFG = Process.new(cmd, table.unpack(args))

    status = "MTFG running"
end


function delete_range(script, start_of_range, end_of_range)
    local fps = player.FPS()
    local frame_time = 1.0/fps
    for idx, action in ipairs(script.actions) do
      if action.at >= (start_of_range - frame_time) and action.at <= (end_of_range + frame_time) then
        script:markForRemoval(idx)
      end
    end
    print("delete range", start_of_range, end_of_range)
    script:removeMarked()
end


function tableLength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end


function import_funscript_generator_json_result()
    status = "MTFG not running"
    local tmpFile = ofs.ExtensionDir() .. "/" .. tmpFileName
    local f = io.open(tmpFile)
    if not f then
        print('Funscript Generator json output file not found')
        return
    end

    local content = f:read("*a")
    f:close()
    json_body = json.decode(content)
    actions = json_body["actions"]

    local fps = player.FPS()
    local frame_time = 1.0/fps
    print("Frame Time:", frame_time)
    local filtered = 0
    if multiaxis then
        local i = 1
        while i <= ofs.ScriptCount() do
            name = ofs.ScriptName(i)
            for k,v in pairs(scriptAssignment) do
                if name and name == scriptNames[v.idx] then
                    if actions[k] then
                        script = ofs.Script(i)
                        if #(actions[k]) < 2 then
                            print("not enough data available")
                        else
                            delete_range(script, actions[k][1]["at"] / 1000.0, actions[k][#(actions[k])]["at"] / 1000.0)
                            for _, action in pairs(actions[k]) do
                                local closest_action, _ = script:closestAction(action["at"])
                                local new_action = Action.new(action["at"]/1000.0, action["pos"], true)
                                if filterSimilarTimestamps and closest_action and math.abs(closest_action.at - new_action.at) <= frame_time then
                                    filtered = filtered + 1
                                else
                                    script.actions:add(new_action)
                                end
                            end
                        end
                        script:commit()
                    end
                end
            end
       	    i = i + 1
        end
    else
        script = ofs.Script(scriptIdx)
        for metric, actions_metric in pairs(actions) do
            print('add ', metric, ' to ', ofs.ScriptName(scriptIdx))
            if #actions_metric < 2 then
                print("not enough data available")
            else
                delete_range(script, actions_metric[1]["at"] / 1000.0, actions_metric[#actions_metric]["at"] / 1000.0)
                for _, action in pairs(actions_metric) do
                    local closest_action, _ = script:closestAction(action["at"]/1000.0)
                    local new_action = Action.new(action["at"]/1000.0, action["pos"], true)
                    if filterSimilarTimestamps and closest_action and math.abs(closest_action.at - new_action.at) <= frame_time then
                        filtered = filtered + 1
                    else
                        script.actions:add(new_action)
                    end
                end
            end
        end

        script:commit()
    end

    if filterSimilarTimestamps then
        print('filtered timestamps', filtered)
    end

end


function invert_selected()
    local script = ofs.Script(ofs.ActiveIdx())
    for idx, action in ipairs(script.actions) do
        if action.selected then
            action.pos = clamp(100 - action.pos, 0, 100)
        end
    end
    script:commit()
end


function gain_selected(gain)
    local script = ofs.Script(ofs.ActiveIdx())
    print('pos gain',gain)
    for idx, action in ipairs(script.actions) do
        if action.selected then
            action.pos = clamp((action.pos-50)*gain+50, 0, 100)
            --action.pos = clamp(100 - action.pos, 0, 100)
        end
    end
    script:commit()
end


function minmaxpos(start,stop,script)
    local minpos=100
	local maxpos=0
    for i=start,stop,1 do
	     minpos = math.min(minpos,script.actions[i].pos)
		 maxpos = math.max(maxpos,script.actions[i].pos)
	end
	return minpos, maxpos
end


function detrend_selected(order,n,adjust_gain)
    local script = ofs.Script(ofs.ActiveIdx())
    print('deTrend')
    local idxlo,idxhi
    script:sort()
    -- find first and last index
    for idx, action in ipairs(script.actions) do
        if action.selected then
            idxlo = idxlo or idx
            idxhi = idx
        end
    end
	if idxlo then
	    min0,max0 = minmaxpos(idxlo, idxlo+n, script)
	    min1,max1 = minmaxpos(idxhi-n, idxhi, script)
	    pos0 = 0.5*(min0+max0)
	    pos1 = 0.5*(min1+max1)
	    at0 = script.actions[idxlo].at
	    at1 = script.actions[idxhi].at
	    range0 = max0-min0
	    range1 = max1-min1
        for idx, action in ipairs(script.actions) do
            if action.selected then
		        center = pos0 + (pos1 - pos0) * (action.at - at0) / (at1 - at0)
			    if adjust_gain>0 then
			        range = range0 + (range1 - range0) * (action.at - at0) / (at1 - at0)
			    else
			        range=50
			    end
                action.pos = clamp(50*(action.pos - center)/range +50, 0, 100)
            end
        end
    script:commit()
    end
end

function align_bottom_points(align_value)
    local script = ofs.Script(ofs.ActiveIdx())
    script:sort()

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

    print('align bottom points to', align_value)

    -- align bottom points
    for idx, action in ipairs(script.actions) do
        if action.selected then
            local bottom_point = true

            local next_action, _ = script:closestActionAfter(action.at)
            if next_action then
                if next_action.pos <= action.pos then
                    bottom_point = false
                end
            end

            local prev_action, _ = script:closestActionBefore(action.at)
            if prev_action then
                if prev_action.pos <= action.pos then
                    bottom_point = false
                end
            end

            if bottom_point then
                action.pos = align_value
            end
        end
    end

    script:commit()
end

function align_top_points(align_value)
    local script = ofs.Script(ofs.ActiveIdx())
    script:sort()

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

    print('align top points to', align_value)

    -- align top points
    for idx, action in ipairs(script.actions) do
        if action.selected then
            local top_point = true

            local next_action, _ = script:closestActionAfter(action.at)
            if next_action then
                if next_action.pos >= action.pos then
                    top_point = false
                end
            end

            local prev_action, _ = script:closestActionBefore(action.at)
            if prev_action then
                if prev_action.pos >= action.pos then
                    top_point = false
                end
            end

            if top_point then
                action.pos = align_value
            end
        end
    end

    script:commit()
end

function continue_minmax(window)
    local script = ofs.Script(ofs.ActiveIdx())
    local maxpos,minpos
    script:sort()

    -- get min and max valuea just before the selection in selection
    stop=false
    for idx, action in ipairs(script.actions) do
        if action.selected and not stop then
            minpos=100
            maxpos=0
            act1 = script:closestActionBefore(action.at)
            maxpos=math.max(act1.pos,maxpos)
            minpos=math.min(act1.pos,minpos)
            act2 = script:closestActionBefore(act1.at)
            maxpos=math.max(act2.pos,maxpos)
            minpos=math.min(act2.pos,minpos)
            act3 = script:closestActionBefore(act2.at)
            maxpos=math.max(act3.pos,maxpos)
            minpos=math.min(act3.pos,minpos)
            act4 = script:closestActionBefore(act3.at)
            maxpos=math.max(act4.pos,maxpos)
            minpos=math.min(act4.pos,minpos)
            stop=true
        end
    end


    print('align selected points')

    -- align top points
    for idx, action in ipairs(script.actions) do
        if action.selected then
            local top_point = true

            local next_action, _ = script:closestActionAfter(action.at)
            if next_action then
                if next_action.pos >= action.pos then
                    top_point = false
                end
            end

            local prev_action, _ = script:closestActionBefore(action.at)
            if prev_action then
                if prev_action.pos >= action.pos then
                    top_point = false
                end
            end
            if top_point then
                action.pos = maxpos
            end
        end
    end
    -- align bottom points
    for idx, action in ipairs(script.actions) do
        if action.selected then
            local bottom_point = true

            local next_action, _ = script:closestActionAfter(action.at)
            if next_action then
                if next_action.pos <= action.pos then
                    bottom_point = false
                end
            end

            local prev_action, _ = script:closestActionBefore(action.at)
            if prev_action then
                if prev_action.pos <= action.pos then
                    bottom_point = false
                end
            end

            if bottom_point then
                action.pos = minpos
            end
        end
    end    
    script:commit()
end

function normalize_selection(windowsize,targetrange)
    local script = ofs.Script(ofs.ActiveIdx())
    local mav = {}
    local dev = {}
    --local devmav = {}
    local idxlo,idxhi
    
    script:sort()

    -- calculate moving averages
    hwindow = math.floor(windowsize/2)
    
    for idx, action in ipairs(script.actions) do
        if action.selected then
            idxlo = idxlo or idx
            idxhi = idx
        end
    end

    for idx, action in ipairs(script.actions) do
        if action.selected then
            mavsum=0
            for ii = -hwindow,hwindow,1 do
                if idx+ii <= idxhi and idx+ii >= idxlo then
                    mavsum = mavsum + (script.actions[idx+ii].pos or 0)
                end
            end
            mavsum=mavsum-0.5*(script.actions[math.max(idx-hwindow,idxlo)].pos + script.actions[math.min(idx+hwindow,idxhi)].pos)
            --mav[idx] = mavsum/(math.min(idx-idxlo,hwindow)+math.min(idxhi-idx,hwindow)-1)
            mav[idx] = mavsum/(windowsize-1)
            dev[idx] = math.abs(action.pos-mav[idx])
        end
    end
    -- calculate moving average of absolute deviation
    for idx, action in ipairs(script.actions) do
        if action.selected then
            devsum=0
            for ii = -hwindow,hwindow-1,1 do
                if idx+ii <= idxhi and idx+ii >= idxlo then
                    devsum = devsum + (dev[idx+ii] or 0)
                end
            end
            devmav = devsum/(math.min(idx-idxlo,hwindow)+math.min(idxhi-idx,hwindow))
            action.pos = clamp((50+0.005*targetrange*(action.pos - mav[idx])/devmav*100),0,100)
            --devmav[idx] = devsum/((math.min(idx-idxlo,hwindow)+math.min(idxhi-idx,hwindow)+1)
        end
    end       
    print('normalize points')

    script:commit()
end


function init()
    print("OFS Version:", ofs.Version())
    print("Detected OS: ", platform)
    local version_file_path = ""
    if platform == "Windows" then
        version_file_path = ofs.ExtensionDir().."\\funscript-editor\\funscript_editor\\VERSION.txt"
    else
        if platform ~= "Linux, Nix" then
            local mtfg_repo_path = ofs.ExtensionDir().."/Python-Funscript-Editor"
            local cmd = "git -C "..mtfg_repo_path.." describe --tags `git -C "..mtfg_repo_path.." rev-list --tags --max-count=1` > "..mtfg_repo_path.."/funscript_editor/VERSION.txt"
            -- print("cmd: ", cmd)
            os.execute(cmd)
            version_file_path = mtfg_repo_path.."/funscript_editor/VERSION.txt"
        end
    end
    if platform == "Linux, Nix" then
        mtfgVersion = "nixpkgs" 
    else
        local f = io.open(version_file_path)
        if f then
            for line in f:lines() do
                if string.find(string.lower(line), "v") then
                    mtfgVersion = string.lower(line):gsub("v", "")
                end
            end
            f:close()
        end
    end
end

function update_logfile_exists()
        local logfile = ""
        if platform == "Windows" then
            logfile = "C:/Temp/funscript_editor.log"
        else
            logfile = "/tmp/funscript_editor.log"
        end
        local f = io.open(logfile)
        if f then
            logfileExist = true
            f:close()
        else
            logfileExist = false
        end
end


function is_empty(s)
  return s == nil or s == ''
end


function update_script_names()
    local i = 1
    scriptNamesCount = 0
    scriptNames = {'ignore'}
    scriptNamesCount = scriptNamesCount + 1
    while i <= ofs.ScriptCount() do
        name = ofs.ScriptName(i)
        if not is_empty(name) then
            table.insert(scriptNames, name)
            scriptNamesCount = scriptNamesCount + 1
       end
       i = i + 1
    end
end

function update_tmp_file_exists()
    local tmpFile = ofs.ExtensionDir() .. "/" .. tmpFileName
    local f = io.open(tmpFile)
    if f then
        tmpFileExists = true
        f:close()
    else
        tmpFileExists = false
    end
end


function update(delta)
    updateCounter = updateCounter + 1
    if processHandleMTFG and not processHandleMTFG:alive() then
        print('funscript generator completed import result')
        processHandleMTFG = nil
        import_funscript_generator_json_result()
    end
    if math.fmod(updateCounter, 100) == 1 then
        update_logfile_exists()
        update_script_names()
        update_tmp_file_exists()
    end
end


function gui()
    ofs.Text("Status: "..status)
    ofs.Text("Version: "..mtfgVersion)
    ofs.Text("Action:")

    ofs.SameLine()
    if not processHandleMTFG then
        if ofs.Button("Start MTFG") then
            exec_mtfg(false)
        end
        ofs.SameLine()
        if ofs.Button("Reprocess Data") then
            exec_mtfg(true)
        end
    else
        if ofs.Button("Kill MTFG") then
            if platform == "Windows" then
                os.execute("taskkill /f /im funscript-editor.exe")
            else
                os.execute("pkill -f funscript-editor.py")
            end
        end
    end

    ofs.SameLine()
    if ofs.Button("Open Config") then
        if platform == "Windows" then
            processHandleConfigDir = Process.new("explorer.exe", ofs.ExtensionDir().."\\funscript-editor\\funscript_editor\\config")
        else
            local cmd = '/usr/bin/dbus-send --session --print-reply --dest=org.freedesktop.FileManager1 --type=method_call /org/freedesktop/FileManager1 org.freedesktop.FileManager1.ShowItems array:string:"file://'
                ..ofs.ExtensionDir()..'/Python-Funscript-Editor/funscript_editor/config/" string:""'
            -- print("cmd: ", cmd)
            os.execute(cmd)
        end
    end

    if logfileExist then
        if platform == "Windows" then
            ofs.SameLine()
            if ofs.Button("Open Log") then
                 processHandleLogFile = Process.new("notepad.exe", "C:/Temp/funscript_editor.log")
            end
        else
            ofs.SameLine()
            if ofs.Button("Open Log") then
                processHandleLogFile = Process.new("/usr/bin/xdg-open", "/tmp/funscript_editor.log")
            end
        end
    end

    if tmpFileExists then
        ofs.SameLine()
        if ofs.Button("Force Import") then
            scriptIdx = ofs.ActiveIdx()
            import_funscript_generator_json_result()
        end
    end

    ofs.Separator()
    ofs.Text("Options:")
    stopAtNextActionPoint, _ = ofs.Checkbox("Stop tracking at next existing point", stopAtNextActionPoint)
    enableLogs, _ = ofs.Checkbox("Enable logging", enableLogs)
    multiaxis, _ = ofs.Checkbox("Enable multiaxis", multiaxis)

    if multiaxis then
        ofs.Separator()
        ofs.Text("Multiaxis Output Assignment:")
        local comboNum = 1
        for k,v in pairs(scriptAssignment) do
            ofs.Text("  o "..k.." ->")
            ofs.SameLine()
            if v.idx > scriptNamesCount then
                v.idx = 1
            end
            v.idx, _ = ofs.Combo("#"..tostring(comboNum), v.idx, scriptNames)
            comboNum = comboNum + 1
        end
    end

    ofs.Separator()

    local enable_post_processing = true
    if enable_post_processing then
        ofs.Text("Post-Processing:")
        ofs.SameLine()
        if ofs.Button("Norm.3") then
             normalize_selection(3,50)
        end
		ofs.SameLine()
        if ofs.Button("Norm.7") then
             normalize_selection(7,50)
        end
        ofs.SameLine()
        if ofs.Button("Norm.15") then
             normalize_selection(15,50)
        end
        ofs.SameLine()
        if ofs.Button("Gain +") then
            gain_selected(1.05)
        end
        ofs.SameLine()
        if ofs.Button("Gain -") then
            gain_selected(1/1.05)
        end
        if ofs.Button("Invert") then
            gain_selected(-1)
        end        
        ofs.SameLine()
        if ofs.Button("deTrend_0") then
            detrend_selected(1,4,0)
        end        
        ofs.SameLine()
        if ofs.Button("deTrend_1") then
            detrend_selected(1,4,1)
        end         ofs.SameLine()
        if ofs.Button("Continue minmax") then
            continue_minmax(-1)
        end
        if ofs.Button("Align Bottom") then
            align_bottom_points(-1)
        end
        ofs.SameLine()
        if ofs.Button("Align Top") then
             align_top_points(-1)
        end
    end

end
