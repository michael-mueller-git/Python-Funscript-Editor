# Open Funscripter Integration

A hacky lua script to use the python funscript generator scripts in Open Funscripter. Copy the lua script to `data/lua` in your OFS directory. Then adjust the python script path in OFS or direct in the source code. Finally add an shortcut for the script in OFS. The script was only tested on Linux!

**NOTE:** We should use luasockets for the communication but i could not get them to work. I install the lua socket module with `pacman -Sy lua-socket` and compile OFS with dynamic linked library support for lua. But when i load the socket module i got an exception `Symbol not found: some_symbol_name`.

Below the code i used for my first lua socket test:

```lua
package.cpath ="/usr/lib/lua/5.4/?.so;" .. package.cpath
package.path = "/usr/share/lua/5.4/?.lua;" .. package.path

local HOST, PORT = "localhost", 9090
local socket = require('socket')

client, err = socket.connect(HOST, PORT)
client:setoption('keepalive', true)

-- Attempt to ping the server once a second
start = os.time()
while true do
  now = os.time()
  if os.difftime(now, start) >= 1 then
    data = client:send("Hello World")
    -- Receive data from the server and print out everything
    s, status, partial = client:receive()
    print(data, s, status, partial)
    start = now
  end
end
```
