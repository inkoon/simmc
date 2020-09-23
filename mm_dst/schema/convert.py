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
# DSTC style dataset fieldnames
FIELDNAME_DIALOG = 'dialogue'
FIELDNAME_USER_UTTR = 'transcript'
FIELDNAME_ASST_UTTR = 'system_transcript'
FIELDNAME_BELIEF_STATE = 'belief_state'
FIELDNAME_STATE_GRAPH_0 = 'state_graph_0'
FIELDNAME_VISUAL_OBJECTS = 'visual_objects'

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = '<SOM>'
END_OF_MULTIMODAL_CONTEXTS = '<EOM>'
START_BELIEF_STATE = '=> Belief State :'
END_OF_BELIEF = '<EOB>'
END_OF_SENTENCE = '<EOS>'

TEMPLATE_PREDICT = '{context} {START_BELIEF_STATE} '
TEMPLATE_TARGET = '{context} {START_BELIEF_STATE} {belief_state} ' \
    '{END_OF_BELIEF} {response} {END_OF_SENTENCE}'
TEMPLATE_TARGET_NORESP = '{context} {START_BELIEF_STATE} {belief_state} {END_OF_SENTENCE}'


def convert_json_to_flattened(
        input_path_json,
        output_path_target):
    """
        Input: JSON representation of the dialogs
        Output: line-by-line stringified representation of each turn
    """

    import ipdb; ipdb.set_trace()
    with open(input_path_json, 'r') as f_in:
        ds = json.load(f_in)
    i = 0
    previousSystemUtt = ' '
    for data in ds : 
       for key in data['turns'] :  
           speaker = key['speaker'] 
           utterance = key['utterance']
           if speaker == 'System' : 
               previousSystemUtt = utterance 
               break 
           for element in key['frames'] : 
               for action in element['actions'] : 
                   print(action)
                   print(action['action'])
                   print(action['slot'])

               for slot in element['slots'] : 
                   print(slot)

               print(element['service'])
           

def parse_flattened_results_from_file(path):
    results = []
    with open(path, 'r') as f_in:
        for line in f_in:
            parsed = parse_flattened_result(line)
            results.append(parsed)

    return results


def parse_flattened_result(to_parse):
    """
        Parse out the belief state from the raw text.
        Return an empty list if the belief state can't be parsed

        Input:
        - A single <str> of flattened result
          e.g. 'User: Show me something else => Belief State : DA:REQUEST ...'

        Output:
        - Parsed result in a JSON format, where the format is:
            [
                {
                    'act': <str>  # e.g. 'DA:REQUEST',
                    'slots': [
                        <str> slot_name,
                        <str> slot_value
                    ]
                }, ...  # End of a frame
            ]  # End of a dialog
    """
    dialog_act_regex = re.compile(r'([\w:?.?]*)  *\[([^\]]*)\]')
    slot_regex = re.compile(r'([A-Za-z0-9_.-:]*)  *= ([^,]*)')

    belief = []

    # Parse
    splits = to_parse.strip().split(START_BELIEF_STATE)
    if len(splits) == 2:
        to_parse = splits[1].strip()
        splits = to_parse.split(END_OF_BELIEF)

        if len(splits) == 2:
            # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
            to_parse = splits[0].strip()

            for dialog_act in dialog_act_regex.finditer(to_parse):
                d = {
                    'act': dialog_act.group(1),
                    'slots': []
                }

                for slot in slot_regex.finditer(dialog_act.group(2)):
                    d['slots'].append(
                        [
                            slot.group(1).strip(),
                            slot.group(2).strip()
                        ]
                    )

                if d != {}:
                    belief.append(d)

    return belief


if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_json',
                        help='path for input, line-separated format (.json)')
    parser.add_argument('--output_dir',
                        help='output directory for saving analysis summary files')
    parser.add_argument('--limit',
                        help='percentage', type=float,default=0.3)
    args = parser.parse_args()
   
    convert_json_to_flattened(args.input_path_json,args.output_dir)