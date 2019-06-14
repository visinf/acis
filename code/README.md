## ACIS/code

### Crayon

#### Installation

Install crayon client:
```
luarocks install crayon OPENSSL_INCDIR=~/.linuxbrew/include OPENSSL_DIR=~/.linuxbrew
```

A slightly modified crayon server is already supplied in ```<acis>/code/crayon/server```
The modification simply includes additional arguments:
- ```--tb-port``` specified the tensorboard port;
- ```--logdir``` specified the log directory of the tensorboard server.

#### Run
Go to ```<acis>/code```. Then,

1. start tensorboard
```
nohup tensorboard --logdir tensorboard --port 6038 > tensorboard/tb.log 2>&1 &
```
2. start crayon
```
nohup python crayon/server/server.py --port 6039 --logdir tensorboard --tb-port 6038 > tensorboard/crayon.log 2>&1 &
```

The crayon can now be accessed through port 6039.
It will in turn access tensorboard via port 6038 and save data in ```tensorboard``` directory.

#### Managing experiments
Session example:

```lua
-- initialise the client
cc = crayon.CrayonClient("localhost", 6039)

-- get a list of experiment names
cc:get_experiment_names()

-- remove experiments
cc:remove_experiment("cvppp_600_btrunc/train")
cc:remove_experiment("cvppp_600_btrunc/train_val")
cc:remove_experiment("cvppp_600_btrunc/val")
```
