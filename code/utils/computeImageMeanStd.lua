require 'image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Utility to compute ')
cmd:text()
cmd:text('Options:')
 ------------ General options --------------------
cmd:option('-dir',        '', 'Directory with the images')
cmd:text()

-- Parsing Arguments
opt = cmd:parse(arg)

-- Main Program --
local ext = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
local findOptions = ' -iname "*_leftImg8bit.' .. ext[1] .. '"'
for i=2,#ext do
   findOptions = findOptions .. ' -o -iname "*' .. ext[i] .. '"'
end

-- Find all the images using the find command
local f = io.popen('find -L ' .. opt.dir .. "/*" .. findOptions)

-- Generate a list of all the images and their class
local means = {}
local stdevs = {}
while true do
   local line = f:read('*line')
   if not line then break end

   local filename = paths.basename(line)
   local im = image.load(line, 3, 'float')

   -- means
   local ms = im:view(3, -1):mean(2):squeeze()
   table.insert(means, {ms[1], ms[2], ms[3]})

   -- stdev
   local ds = im:view(3, -1):std(2):squeeze()
   table.insert(stdevs, {ds[1], ds[2], ds[3]})
end

f:close()

print('Found ', #means, ' images')

local means = torch.FloatTensor(means)
local stdevs = torch.FloatTensor(stdevs)

local mmeans = means:mean(1):squeeze()
local stdevs = stdevs:mean(1):squeeze()

print(string.format('Means: %.4f, %.4f, %.4f', mmeans[1], mmeans[2], mmeans[3]))
print(string.format('Stdev: %.4f, %.4f, %.4f', stdevs[1], stdevs[2], stdevs[3]))
