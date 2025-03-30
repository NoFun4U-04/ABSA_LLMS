from mapper import *
import pandas as pd
import os

def get_output(labels, task: str):
    if task == 'triplet':
        pass
    elif task == 'quadruplet':
        return get_quadruplet_output(labels)
    elif task == 'pair':
        return labels

def get_triplet_io(triplet):
    pass

def get_quadruplet_output(labels, prompt_format=1):

    def get_quadruplet_1(quad):
        ac, at, sp, ot = quad.split(',')

        if (at == 'null') and (ot == 'null'):
            completion = f"{mapping_category(os.environ['domain'], ac, 'vie')} {SENTIMENT_ENG2VIET[sp]}"
        else:
            at = 'nó' if at == 'null' else at
            ot = f'#{SENTIMENT_ENG2VIET[sp]}' if ot == 'null' else ot

            completion = f"{mapping_category(os.environ['domain'], ac, 'vie')} {SENTIMENT_ENG2VIET[sp]} vì {at} {ot}"
        
        return completion

    def get_quadruplet_2(quad):
        ac, at, sp, ot = quad.split(',')

        if (at == 'null') and (ot == 'null'):
            completion = f"{mapping_category(os.environ['domain'], ac, 'vie')}::{SENTIMENT_ENG2VIET[sp]}"
        else:
            completion = f"{mapping_category(os.environ['domain'], ac, 'vie')}::{SENTIMENT_ENG2VIET[sp]}::{at}::{ot}"
        
        return completion
    
    quads = labels.split(';')
    quadruplets = [] 
    for quad in quads:
        quad = quad.strip().strip('{}')
        if prompt_format == 1:
            quadruplet = get_quadruplet_1(quad)
        elif prompt_format == 2:
            quadruplet = get_quadruplet_2(quad)
        quadruplets.append(quadruplet)
    output_text = ' và '.join(quadruplets)
    
    return output_text


def read_data(domain, task):
    if task == 'pair':
        df_train = pd.read_csv(f'../data/Pair/{domain}/Train.csv')
        df_test = pd.read_csv(f'../data/Pair/{domain}/Test.csv')
    elif task == 'triplet':
        pass
    elif task == 'quadruplet':
        df_train = pd.read_csv(f'../data/Quadruplet/{domain}/Train.csv')
        df_test = pd.read_csv(f'../data/Quadruplet/{domain}/Test.csv')
    
    return df_train, df_test