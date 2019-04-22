"""Optimisation class"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

# pylint: disable=R0914, R0912


import random
import logging

import deap.algorithms
import deap.tools
import functools
import pickle
import numpy as np

logger = logging.getLogger('__main__')


def _evaluate_invalid_fitness(toolbox, population, eval_stat = 600):
    '''Evaluate the individuals with an invalid fitness

    Returns the count of individuals with invalid fitness
    '''
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses_with_times = toolbox.map(functools.partial(toolbox.evaluate,\
                                  timeout_stat = eval_stat), invalid_ind)
    fitnesses = [fitness_ for fitness_,_ in fitnesses_with_times]
    eval_times = [times_ for _,times_ in fitnesses_with_times]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    return len(invalid_ind),eval_times


def _update_history_and_hof(halloffame, history, population):
    '''Update the hall of fame with the generated individuals

    Note: History and Hall-of-Fame behave like dictionaries
    '''
    if halloffame is not None:
        halloffame.update(population)

    history.update(population)


def _record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)


def _get_offspring(parents, toolbox, cxpb, mutpb):
    '''return the offsprint, use toolbox.variate if possible'''
    if hasattr(toolbox, 'variate'):
        return toolbox.variate(parents, toolbox, cxpb, mutpb)
    return deap.algorithms.varAnd(parents, toolbox, cxpb, mutpb)


def eaAlphaMuPlusLambdaCheckpoint(
        population,
        toolbox,
        mu,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        cp_frequency=1,
        cp_filename=None,
        continue_cp=False,
        eval_stat_default = 600,
        **kwargs):
    r"""This is the :math:`(~\alpha,\mu~,~\lambda)` evolutionary algorithm

    Args:
        population(list of deap Individuals)
        toolbox(deap Toolbox)
        mu(int): Total parent population size of EA
        cxpb(float): Crossover probability
        mutpb(float): Mutation probability
        ngen(int): Total number of generation to run
        stats(deap.tools.Statistics): generation of statistics
        halloffame(deap.tools.HallOfFame): hall of fame
        cp_frequency(int): generations between checkpoints
        cp_filename(string): path to checkpoint filename
        continue_cp(bool): whether to continue
    """
    eval_time_stats = []
    
    if continue_cp:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(cp_filename, "r"))
        population = cp["population"]
        parents = cp["parents"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        history = cp["history"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        start_gen = 1
        parents = population[:]
        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        history = deap.tools.History()

        # TODO this first loop should be not be repeated !
        invalid_count,eval_times = _evaluate_invalid_fitness(toolbox, population)
        _update_history_and_hof(halloffame, history, population)
        _record_stats(stats, logbook, start_gen, population, invalid_count)

    eval_time_stats.extend(eval_times)
    eval_time_stats = [int(eval_time_) if eval_time_ is not None else eval_stat_default \
                       for eval_time_ in eval_time_stats]
    
    # Begin the generational process
    for gen in range(start_gen + 1, ngen + 1):
        offspring = _get_offspring(parents, toolbox, cxpb, mutpb)

        population = parents + offspring
        eval_stat = int(np.percentile(eval_time_stats,80))
        invalid_count,eval_times = _evaluate_invalid_fitness(toolbox, offspring,
                             eval_stat =eval_stat)
        _update_history_and_hof(halloffame, history, population)
        _record_stats(stats, logbook, gen, population, invalid_count)

        # Select the next generation parents
        parents = toolbox.select(population, mu)
        logbook_stream = logbook.stream
        logger.info(logbook_stream)

        if(cp_filename and cp_frequency and
           gen % cp_frequency == 0):
            cp = dict(population=population,
                      generation=gen,
                      parents=parents,
                      halloffame=halloffame,
                      history=history,
                      logbook=logbook,
                      rndstate=random.getstate())
            pickle.dump(cp, open(cp_filename, "wb"))
            logger.debug('Wrote checkpoint to %s', cp_filename)
            
            # Writing the generation statistics in a file
            f =  open('logbook_info.txt','a')
            f.write('%s %s \n'%(logbook_stream, cp_filename.split('.')[0]))
            f.close()
            
        if kwargs.get('cp_backup') and gen % kwargs.get('cp_backup_frequency',5) == 0:
            cp_backup = kwargs.get('cp_backup')
            pickle.dump(cp, open(cp_backup, "wb"))
            logger.debug('Wrote checkpoint backup to %s',cp_backup)
            
        eval_time_stats.extend(eval_times)
        eval_time_stats = [int(eval_time_) if eval_time_ is not None else eval_stat_default \
                       for eval_time_ in eval_time_stats]
        logger.debug('evaluation time stats =  %s',eval_stat)
        if len(eval_time_stats) > 2*len(population):
            eval_time_stats = eval_time_stats[-2*len(population):]
        
    return population, halloffame, logbook, history
