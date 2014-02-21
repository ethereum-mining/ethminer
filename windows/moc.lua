
local function quote(x)
	return '"' .. x .. '"'
end

local function toForward(x)
	return x:gsub('\\', '/')
end

-- arguments are in this order
local cmd = arg[1]
local outFile = arg[2]
local includes = toForward(arg[3])
local defines = arg[4]
local inFile = arg[5]

-- build list of includes
local includes2 = ""
for i in string.gmatch(includes, "[^;]+") do
  includes2 = includes2.." -I "..quote(i)
end
includes = includes2;

-- build list of defines
local defines2 = ""
for i in string.gmatch(defines, "[^;]+") do
  defines2 = defines2.." -D"..i
end
defines = defines2

-- moc doesn't compile boost correctly, so skip those headers
workarounds=' -DBOOST_MP_CPP_INT_HPP -DBOOST_THREAD_WEK01082003_HPP'

-- build command
cmd = quote(cmd).." -o "..quote(outFile)..includes..defines..workarounds..' '..quote(inFile)
print(cmd)
os.execute(quote(cmd))

