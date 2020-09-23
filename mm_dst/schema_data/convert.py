#!/usr/bin/env python3
"""
    Script for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.
"""
import json
import re
import os
import argparse
import ipdb 

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = '<SOM>'
END_OF_MULTIMODAL_CONTEXTS = '<EOM>'
START_BELIEF_STATE = '=> Belief State :'
END_OF_BELIEF = '<EOB>'
END_OF_SENTENCE = '<EOS>'
START_OF_SLOT = ' [ '
END_OF_SLOT = ' ] '

def convert_json_to_flattened(
        input_path_json,
        output_path_target, context):
    """
        Input: JSON representation of the dialogs
        Output: line-by-line stringified representation of each turn
    """
    output = open(output_path_target,'a') 
    
    #import ipdb; ipdb.set_trace()
    with open(input_path_json, 'r') as f_in:
        ds = json.load(f_in)

    result = ""
    previousUtt = ""
    for data in ds : 
        contextSize =  0
        #previousUtt = ""
        for key in data['turns'] :  
            speaker = key['speaker'] 
            utterance = key['utterance']
            contextSize += 1 
            if speaker == 'SYSTEM' : ## For SYSTEM utterance
                result += ' ' + utterance + ' ' +  '\n'  
                output.write(result)  ## print to output file 
                previousUtt += "SYSTEM : {}".format(utterance) + ' ' ## store utterance history
                result = previousUtt ## initialize result 
                if contextSize >= context*2 : ## if bigger than context size 
                    contextSize = 0   # reset utterance history and context size 
                    result = previousUtt
                    previousUtt = "" 
            else :    ## For USER utterance
                result+= "User : {}".format(utterance)
                previousUtt  += "User : {}".format(utterance) ## store utterance history 
                result+= ' ' + START_OF_MULTIMODAL_CONTEXTS+' '  ## Start of multimodal
                result+=END_OF_MULTIMODAL_CONTEXTS + ' '    ## multimodal is empty for now.
                previousUtt+= ' ' + START_OF_MULTIMODAL_CONTEXTS+' '
                previousUtt+=END_OF_MULTIMODAL_CONTEXTS + ' '
                result+=START_BELIEF_STATE   ## start belief state
                for element in key['frames'] :  ## loop through frames 
                    for action in element['actions'] : 
                        if action['act'] == 'GOODBYE' or action['act'] == 'THANK_YOU' : 
                            result += ' err chitchat '   ## convert DA GOODBYE and THANKYOU to CHITCHAT
                        else : 
                            result += ' da ' + action['act'].replace('_'," ").lower()  ## convert DA to simmc format

                        if action['slot'] is not None : 
                            if len(action['values'])==0 :  ## if slot value is not defined
                                result += ' ' + action['slot'].lower()  ## make slot name into object 
                                result += START_OF_SLOT ## make empty slot 
                            else : 
                                result += START_OF_SLOT     ## if slot value is defined
                                result += action['slot'].lower() 
                                result += ' = ' 
                                for values in action['values'] :  ## loop through slot value 
                                    lst = re.findall('[A-Z][^A-Z]*',values) ## split for each capital letters
                                    if len(lst) == 0 or len(lst) == len(values) : 
                                        result += values   ## just add raw slot value if it has no capital, or all capital letters
                                    else :
                                        for ele in re.findall('[A-Z][^A-Z]*',values) : 
                                            result += " " + ele.lower()  # Seperate each capital letter
                            result += END_OF_SLOT 
                            result += END_OF_BELIEF  + ' '
                


if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_json',
                        help='path for input, line-separated format (.json)',required=True)
    parser.add_argument('--output_path',
                        help='output text file path (.txt)', required=True)
    parser.add_argument('--context',
                        help='context size', type=int,default=1)
    args = parser.parse_args()

    convert_json_to_flattened(args.input_path_json,args.output_path,args.context)