import numpy as np
import pylab as pl
from random import randrange
# custom modules
from utils     import Options
from simulator import Simulator

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

# TODO: load your agent
agent =None

# 1. control loop
if opt.disp_on:
    win_all = None
    win_pob = None
epi_step = 0    # #steps in current episode
nepisodes = 0   # total #episodes executed
nepisodes_solved = 0
action = 0     # action to take given by the network

# start a new game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
for step in range(opt.eval_steps):

    # check if episode ended
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
    else:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: here you would let your agent take its action
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # this just gets a random action
        action = randrange(opt.act_num)
        state = sim.step(action)

        epi_step += 1

    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)

    if step % opt.prog_freq == 0:
        print step

    if opt.disp_on:
        if win_all is None:
            pl.figure()
            win_all = pl.imshow(state.screen)
            pl.figure()
            win_pob = pl.imshow(state.pob)
        else:
            win_all.set_data(state.screen)
            win_pob.set_data(state.pob)
        pl.pause(opt.disp_interval)
        pl.draw()

# 2. calculate statistics
print float(nepisodes_solved) / float(nepisodes)
# 3. TODO perhaps  do some additional analysis
