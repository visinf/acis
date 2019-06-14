--
-- Helper class to keep track of mean and deviation
--

local stats = {}

-- Get the standard deviation of a table
stats.std = function(t)
  local m
  local vm
  local sum = 0
  local count = 0
  local result

  m = stats.mean( t )

  for k,v in pairs(t) do
    if type(v) == 'number' then
      vm = v - m
      sum = sum + (vm * vm)
      count = count + 1
    end
  end

  result = math.sqrt(sum / (count-1))

  return result
end

-- Get the mean value of a table
stats.mean = function(t)
  local sum = 0
  local count = 0

  for k,v in pairs(t) do
    if type(v) == 'number' then
      sum = sum + v
      count = count + 1
    end
  end

  return (sum / count)
end

stats.write = function(filename, stat, count)
  local fn = io.open(filename, 'w')
  local fnData = ''
  if count then
    for i = 1,stat:nElement() do
      fnData = fnData .. string.format("%d \"T = %d\" %4.3f\n", i - 1, i, stat[i] / count[i])
    end
  else
    for i = 1,stat:nElement() do
      fnData = fnData .. string.format("%d \"T = %d\" %4.3f\n", i - 1, i, stat[i])
    end
  end
  fn:write(fnData)
  fn:close()
end

stats.report = function(stat, tag, epoch, iter, dump, savedir, logger)
  local adic = {}
  for i,v in ipairs(stat.dic) do
    table.insert(adic, math.abs(v))
  end

  local iou = stats.mean(stat.ious)
  local bce = stats.mean(stat.bces)

  print(tag)
  print('--------------')
  print(string.format(' MWCov: %4.3f ±%4.3f', stats.mean(stat.wtCov),  stats.std(stat.wtCov)))
  print(string.format('MUWCov: %4.3f ±%4.3f', stats.mean(stat.uwtCov), stats.std(stat.uwtCov)))
  print(string.format('   IoU: %4.3f ±%4.3f', iou,   stats.std(stat.ious)))
  print(string.format('    AP: %4.3f ±%4.3f', stats.mean(stat.aps),    stats.std(stat.aps)))
  print(string.format('  Dice: %4.3f ±%4.3f', stats.mean(stat.dices),  stats.std(stat.dices)))
  print(string.format('   BCE: %4.3e ±%4.3e', bce,   stats.std(stat.bces)))
  print(string.format('   CLS: %4.3e ±%4.3e', stats.mean(stat.cls),    stats.std(stat.cls)))
  print(string.format('  aDiC: %4.3f ±%4.3f', stats.mean(adic),        stats.std(adic)))
  print(string.format('   DiC: %4.3f ±%4.3f', stats.mean(stat.dic),    stats.std(stat.dic)))
  print(string.format('  Accy: %4.3f ±%4.3f', stats.mean(stat.accs),    stats.std(stat.accs)))
  print(string.format('  Conf: %4.3f ±%4.3f', stats.mean(stat.confs),    stats.std(stat.confs)))
  print('------------------------\n')

  if logger then
    logger:add_scalar_value("data/dice",      stats.mean(stat.dices),  -1, iter)
    if iou == iou then logger:add_scalar_value("data/iou",       iou,  -1, iter) end
    if bce == iou then logger:add_scalar_value("data/mask_loss", bce,  -1, iter) end
    logger:add_scalar_value("data/muwcov",    stats.mean(stat.uwtCov), -1, iter) 
  end

  if dump then
    local dir = paths.concat(savedir, string.format('eval_%s_%03d', tag, epoch))
    if not paths.dirp(dir) and not paths.mkdir(dir) then
       cmd:error('error: unable to create checkpoint directory: ' .. dir .. '\n')
    end
    print('Saving stats in ' .. dir)
    -- saving the stats in a file
    stats.write(paths.concat(dir, 'iou.data'), stat.iou, stat.counts)
    stats.write(paths.concat(dir, 'ap.data'), stat.ap, stat.counts)
    stats.write(paths.concat(dir, 'dice.data'), stat.dice, stat.counts)
    stats.write(paths.concat(dir, 'bce.data'), stat.bce, stat.counts)
    stats.write(paths.concat(dir, 'reward.data'), stat.reward, stat.counts)
    stats.write(paths.concat(dir, 'critic.data'), stat.critic, stat.counts)
    stats.write(paths.concat(dir, 'cdiff.data'), stat.cdiff, stat.counts)
    stats.write(paths.concat(dir, 'count.data'), stat.counts)
  end

  return stats.mean(stat.dices)
end

stats.validate = function(trainer, opt, set, epoch, iter, dump, loader, logger)
  trainer:clearStates()

  local stat = {}
  
  stat.op = {}
  stat.accs = {}
  stat.confs = {}
  stat.cls = {}
  stat.dic = {}
  stat.ious = {}
  stat.aps = {}
  stat.dices = {}
  stat.bces = {}
  stat.wtCov = {}
  stat.uwtCov = {}

  stat.sizeCorr = torch.ones(opt.max_seq_length, opt.max_seq_length)

  stat.counts = torch.zeros(opt.max_seq_length)
  stat.iou = torch.zeros(opt.max_seq_length)
  stat.ap = torch.zeros(opt.max_seq_length)
  stat.dice = torch.zeros(opt.max_seq_length)
  stat.bce = torch.zeros(opt.max_seq_length)
  stat.reward = torch.zeros(opt.max_seq_length)
  stat.critic = torch.zeros(opt.max_seq_length)
  stat.cdiff = torch.zeros(opt.max_seq_length)
  stat.segSize = torch.zeros(opt.max_seq_length)
  
  for t = 1,opt.numTestRuns do
    xlua.progress(t, opt.numTestRuns)
    trainer:test(loader, epoch, 0.5, dump, gen, stat, t)
  end
  
  return stats.report(stat, set, epoch, iter, dump, opt.save, logger)
end


return stats
