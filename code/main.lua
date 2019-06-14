require 'common'
require 'os'

require 'utils/ContextManager'

local models = require 'models/init'
local DataLoader = require 'dataloader'
local Trainer = require 'train'
local opts = require 'opts'
local crayon = require 'crayon'
local stats = require 'utils/stats'
local sched = require 'schedules'

require 'colormap'
colormap:setStyle('parula')

torch.setdefaulttensortype('torch.FloatTensor')

--TODO ?
--torch.setnumthreads(1)

local opt = opts.parse(arg)

opt.mode = 'train'
--opt.mode = 'preproc'

-- create log file
inGPUMode = opt.nGPU > 1
if inGPUMode==1 then
  cutorch.setDevice(opt.gpu_id)
end

cudnn.verbose = true

if opt.manualSeed ~= 0 then
  print('Setting the seed ' .. opt.manualSeed)
  torch.manualSeed(opt.manualSeed)
  cutorch.manualSeedAll(opt.manualSeed)
  math.randomseed(opt.manualSeed)
else
  torch.seed()
  cutorch.seed()
  math.randomseed(os.time())
end

local gen = torch.Generator()
torch.manualSeed(gen, opt.manualSeed)

-- enable/disable debugging
-- off by default
nngraph.setDebug(false)

-- Create model
local fromEpoch = 1
local model = {}
local optimState

if opt.continue ~= '' then
  local ver
  ver, fromEpoch = unpack(string.split(opt.continue, ','))
  fromEpoch = tonumber(fromEpoch)

  local snapPrefix = string.format('checkpoints/%s_%s/%03d_', opt.dataset, ver, fromEpoch)

  print('Loading model from ' .. snapPrefix .. '*')

  model.Actor = torch.load(snapPrefix .. 'actor.model')

  if opt.cudnn == 'deterministic' then
    print('Actor: deterministic')
    model.Actor:apply(function(m)
       if m.setMode then m:setMode(1,1,1) end
    end)
  end

  -- shortcut
  model.Actor.KLDLoss = model.Actor.kldiv.data.module

  print('-- lambdaKL = ' .. model.Actor.KLDLoss.KLDK)
  if model.Actor.KLDLoss.KLDK ~= opt.lambdaKL then
    model.Actor.KLDLoss.KLDK = opt.lambdaKL
    print('-- Setting lambdaKL = ' .. model.Actor.KLDLoss.KLDK)
  end

  local decVer, decEpoch = unpack(string.split(opt.decoder, ','))
  local decPrefix = string.format('checkpoints/%s_%s/%s_', opt.dataset, decVer, decEpoch)
  local decoderPath = decPrefix .. 'cvae_dec.t7'
  if not paths.filep(decoderPath) then
    decoderPath = decPrefix .. 'decoder.model'
  end

  print('Loading decoder from ', decoderPath)
  model.Decoder = torch.load(decoderPath)

  optimState = {}
  if opt.learnDecoder then
    optimState.Decoder = torch.load(decPrefix .. 'decoder_opt.t7')
  else
    model.Decoder:evaluate()
  end

  if opt.cudnn == 'deterministic' then
    print('Decoder: deterministic')
    model.Decoder:apply(function(m)
       if m.setMode then m:setMode(1,1,1) end
    end)
  end

  print('Loading Optimisation State')
  optimState.Actor = torch.load(snapPrefix .. 'actor_opt.t7')

  if opt.grad == 'ac' then

    -- initialising either the control net or the critic
    local criticNetPath = snapPrefix .. 'critic.model'
    if paths.filep(criticNetPath) and not opt.reset_critic then
      model.Critic = torch.load(criticNetPath)
      optimState.Critic = torch.load(snapPrefix .. 'critic_opt.t7')
    else
      print('Warning: Critic NOT found. Starting from sctatch.')
      model.Critic = models.setup('critic', opt, checkpoint)
      optimState.Critic   = {learningRate = opt.betaCritic * opt.learning_rate,
                             weightDecay = opt.weightDecay}
    end

    if opt.cudnn == 'deterministic' then
      print('Critic: deterministic')
      model.Critic:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
    end

  end

  fromEpoch = fromEpoch + 1
else
  if opt.grad == 'ac' then
    model.Critic = models.setup('critic', opt, checkpoint)
  end

  -- Loading pre-trained CNN
  local ver, preEpoch = unpack(string.split(opt.decoder, ','))
  local snapPrefix = string.format('checkpoints/%s_%s/%s_', opt.dataset, ver, preEpoch)

  local encoderPath = snapPrefix .. 'cvae_enc.t7'
  local decoderPath = snapPrefix .. 'cvae_dec.t7'

  if paths.filep(decoderPath)then
    print('Loading decoder from ', decoderPath)
    model.Decoder = torch.load(decoderPath)
  else
    print('Initialising decoder')
    model.Decoder = models.setup('cvae_decoder', opt, checkpoint)
    model.Decoder:evaluate()
  end

  model.Actor = models.setup('cvae_encoder', opt, checkpoint)

  -- shortcut
  if model.Actor.kldiv then
    model.Actor.KLDLoss = model.Actor.kldiv.data.module
    print('-- lambdaKL = ' .. model.Actor.KLDLoss.KLDK)
    if model.Actor.KLDLoss.KLDK ~= opt.lambdaKL then
      model.Actor.KLDLoss.KLDK = opt.lambdaKL
      print('-- Setting lambdaKL = ' .. model.Actor.KLDLoss.KLDK)
    end
  end
end

-- turning on sampling
if model.Actor.sampler then
  print('Found sampler: sampler.fixed = ', model.Actor.sampler.data.module.fixed)
  model.Actor.sampler.data.module.fixed = opt.noSampling
  print('Set sampler.fixed = ', model.Actor.sampler.data.module.fixed)
else
  error('Actor sampler not found!')
end


if opt.learnControl or opt.learnOp then
  if opt.continue ~= '' and not opt.reset_control then
    local ctlVer, ctlEpoch = unpack(string.split(opt.continue, ','))
    ctlEpoch = tonumber(ctlEpoch)

    local snapPrefixCtl = string.format('checkpoints/%s_%s/%03d_', opt.dataset, ctlVer, ctlEpoch)
    local controlNetPath = snapPrefixCtl .. 'control.model'
    assert(paths.filep(controlNetPath), 'ControlNet not found')
    model.Control = torch.load(controlNetPath)
    optimState.Control = torch.load(snapPrefixCtl .. 'control_opt.t7')
    if optimState.Control then
      optimState.Control.learningRate = opt.learning_rate
    else
      optimState.Control = {}
    end
  else
    print('Warning: Starting ControlNet from sctatch.')
    model.Control = models.setup('control', opt, checkpoint)
    if optimState then
      optimState.Control = {}
    end
  end

  if opt.cudnn == 'deterministic' then
    print('Control: deterministic')
    model.Control:apply(function(m)
       if m.setMode then m:setMode(1,1,1) end
    end)
  end
end



local criterion = {}
if opt.loss == 'bce' then
  print('Actor Loss: Using BCE')
  criterion.BCE = nn.BCECriterion(nil, false)
else
  print('Actor Loss: Using Dice')
  criterion.BCE = nn.DICECriterion()
end
if opt.nGPU > 0 then criterion.BCE = criterion.BCE:cuda() end

criterion.TD = nn.MSECriterion()
criterion.TD.sizeAverage = false

criterion.MSE = nn.MSECriterion()

criterion.CE = nn.CrossEntropyCriterion()
criterion.CE.sizeAverage = false

if opt.nGPU > 0 then
  criterion.TD = criterion.TD:cuda()
  criterion.CE = criterion.CE:cuda()
  criterion.MSE = criterion.MSE:cuda()
end

function createExperiment(client, name)
  local ok, err = pcall(function () return client:create_experiment(name) end)
  if not ok then
    client:remove_experiment(name)
    ok, err = pcall(function () return client:create_experiment(name) end)
  end
  assert(ok, err)
  return err
end

--
-- Logging
-- Setting up crayon
--
local crayon_client = crayon.CrayonClient(opt.crayonHost, opt.crayonPort)
local loggers = {}
loggers.train = createExperiment(crayon_client, opt.config_id .. string.format("/train_%02d", opt.seq_length))
loggers.train_val = createExperiment(crayon_client, opt.config_id .. "/train_val")
loggers.val = createExperiment(crayon_client, opt.config_id .. "/val")

local baselineSchedule = {
  {"epoch"}
}


-- Data
-- Dataloaders
--
local trainLoader = DataLoader.create(opt, 'train')

local trainLoaderCritic
if opt.grad == 'ac' then
  trainLoaderCritic = DataLoader.create(opt, 'train')
end

local valLoader = DataLoader.create(opt, 'val')

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState, loggers)

if opt.checkOnly then
  stats.validate(trainer, opt, 'val', fromEpoch - 1, opt.dump, valLoader)
  os.exit(1)
end

if opt.grad == 'ac' then
  opt.critic_warmup = trainLoader:size() * (fromEpoch - 1) + opt.critic_warmup
  print('Critic Warm-up after ' .. (opt.critic_warmup / trainLoader:size()) .. ' epochs')
end

print('Starting training... From epoch ', fromEpoch)
local sched_id = 1
for s = 1,(#sched.btrunc - 1) do
  if fromEpoch >= sched.btrunc[sched_id + 1].epoch then
    sched_id = sched_id + 1
  end
end
print("SchedID: ", sched_id, " | Next Epoch ", sched.btrunc[sched_id].epoch)

for epoch = fromEpoch, opt.nEpochs do
  
   -- Train for a single epoch
   if opt.grad == 'fixed' then

     if sched.btrunc[sched_id] ~= nil and 
        epoch % sched.btrunc[sched_id].epoch == 0 then

       local params = sched.btrunc[sched_id]
       print(">>> [BLTrunc] Next step schedule: ", params)
       opt.learning_rate = params.lr
       if params.len ~= opt.seq_length then
         opt.seq_length = params.len
         opt.loadSeqLength = params.load_len
         loggers.train = createExperiment(crayon_client, opt.config_id .. string.format("/train_%02d", opt.seq_length))
       end
       local optimState = trainer.optimState
       trainer = Trainer(model, criterion, opt, optimState, loggers)
       sched_id = sched_id + 1
     end

     trainer:trainBaseline(epoch, trainLoader)
   else

     if sched.ac[sched_id] ~= nil and 
        epoch % sched.ac[sched_id].epoch == 0 then

       local params = sched.ac[sched_id]
       print(">>> [AC] Next step schedule: ", params)
       opt.learning_rate = params.lr
       if params.len ~= opt.seq_length then
         opt.seq_length = params.len
         opt.loadSeqLength = params.load_len
         loggers.train = createExperiment(crayon_client, opt.config_id .. string.format("/train_%02d", opt.seq_length))
       end
       local optimState = trainer.optimState

       -- setting KL
       model.Actor.KLDLoss.KLDK = params.kld
       print('-- Setting lambdaKL = ' .. model.Actor.KLDLoss.KLDK)

       trainer = Trainer(model, criterion, opt, optimState, loggers)
       sched_id = sched_id + 1
     end

     trainer:train(epoch, trainLoader, trainLoaderCritic)
   end

   trainer:clearStates()

   local checkpoint = (epoch % opt.checkpoint_after == 0)

   if checkpoint then
     trainer:saveModels(epoch)
   end

   -- Validation? Checkpoint?
   if epoch % opt.validation_after == 0 then
     local dump = checkpoint
     stats.validate(trainer, opt, 'val', epoch, trainer.iter, dump, valLoader, loggers.val)

     if opt.valTrain then
       stats.validate(trainer, opt, 'train', epoch, trainer.iter, false, trainLoader, loggers.train_val)
     end
   end

   trainer:clearStates()
end
