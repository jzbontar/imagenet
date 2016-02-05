require 'torch'

local jdb = {}

local Write = torch.class('jdb.Write', jdb)
function Write:__init(fname, size)
   self.fname = fname
   self.idx = 1
   self.offset = torch.LongTensor(size + 1)
   self.offset[1] = 1
   self.file = io.open(fname .. '.data' , 'w')
end

function Write:close()
   torch.save(self.fname .. '.offset', self.offset:resize(self.idx):clone())
   self.file:close()
end

function Write:append(data)
   self.offset[self.idx + 1] = self.offset[self.idx] + string.len(data)
   self.idx = self.idx + 1
   self.file:write(data)
end

local Read = torch.class('jdb.Read', jdb)
function Read:__init(fname)
   self.offset = torch.load(fname .. '.offset')
   self.file = torch.DiskFile(fname .. '.data', 'r'):binary()
end

function Read:get(idx)
   self.file:seek(self.offset[idx])
   local storage = self.file:readByte(self.offset[idx + 1] - self.offset[idx])
   return torch.ByteTensor(storage)
end

function Read:size()
   return self.offset:nElement() - 1
end

return jdb
