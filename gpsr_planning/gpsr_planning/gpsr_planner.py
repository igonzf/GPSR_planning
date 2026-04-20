#!/usr/bin/env python3

# MIT License

# Copyright (c) 2023  Miguel Ángel González Santamarta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
from typing import Tuple
from llama_ros.langchain import ChatLlamaROS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import datetime as dt
import re
from itertools import product


class GpsrPlanner:

    def __init__(
        self,
        robot_actions_path: str = "robot_actions_template.json",
        waypoints_path: str = "waypoints.json",
        objects_path: str = "objects.json",
        names_path: str = "names.json"
    ) -> None:

        self.robot_actions = json.load(open(robot_actions_path))
        self.waypoints_path = waypoints_path

        # Load data files for $ref definitions
        with open(objects_path) as f:
            obj_data = json.load(f)
        with open(waypoints_path) as f:
            wp_data = json.load(f)
        with open(names_path) as f:
            names_data = json.load(f)

        self.placeholder_values = {
            "<items>": sorted([o.replace(" ", "_") for o in obj_data['items']]),
            "<categories>": sorted([c.replace(" ", "_") for c in obj_data['categories']]),
            "<waypoints>": sorted([w.replace(" ", "_") for w in wp_data]),
            "<names>": sorted(names_data['names'])
        }

        self.create_grammar()
        # self.load_waypoints()

        self.llm = ChatLlamaROS(
            temp=0.60,
            top_p=0.8,
            top_k=20,
            min_p=0,
            grammar_schema=self.grammar_schema
        )
        
        print("Grammar schema:")
        print(self.grammar_schema)

        is_lora_added = False

        chat_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a robot named Tiago who is participating in the Robocup with the Gentlebots team from Spain, "
                "made up of the Rey Juan Carlos University of Madrid and the University of León. "
                
                + ("You have to generate plans, sequence of actions, to achieve goals. "
                # "Use the least number of actions as possible and try to speak as much as you can. " # Limiting the number of actions can lead to worse results
                "Use only the actions listed below. " if not is_lora_added else "")
                
                + ("The output should be a JSON object with a key 'actions' containing a list of actions. "
                   "Each action has 'explanation_of_next_actions', explaining the reason for the action, "
                   "and an action-specific key (e.g. find_object) with its parameters. "
                   "Only output the JSON object without any additional explanatory text or steps. " 
                if not is_lora_added else '') +

                # theoretically better but worse results
                # "The format of the output of the plan should be {{explanation_of_next_actions, action}}[], "
                # "where explanation_of_next_actions is a string with an explanation on why you choose the action, the action object key is the action name and the value is an object with the action parameters."
                
                "Planning rules:\n"
                "- The robot always starts at the instruction point.\n"
                "- All task actions are executed at waypoints.\n"
                "- Rooms, furniture, and tables are waypoints.\n"
                "- If a task refers to a room, furniture, table, or any other waypoint different from the current one, you must insert a move_to action before the task action.\n"
                "- Do not perform perception, search, counting, inspection, delivery, or interaction actions at a waypoint unless the robot is already there.\n"
                "- Keep track of the current waypoint after each move_to.\n"
                "- If the robot is already at the required waypoint, do not add an unnecessary move_to.\n"
                "- For commands such as counting, finding, describing, inspecting, answering about, or interacting with people or objects in a room, first move to that room if needed.\n"
                "- For commands asking to report back to the user, return to the instruction point before speaking unless the command clearly specifies another destination.\n"
                "- If they don't give a location, you don't have to move."

                # "Use the move_to action before each action that requires changing the waypoint and remember your current waypoint. "
                # "Answer only to the arguments you are asked for. "
                "Today is {day}, tomorrow is {tomorrow} and the time is {time_h}. "
                # "You start at the instruction point. "
                "\n\n"

                + ("ACTIONS:\n"
                "{actions_descriptions}" if not is_lora_added else '')
            ),
            HumanMessagePromptTemplate.from_template(
                "You are at the instruction point, generate a plan to achieve your goal: {prompt}"
            )
        ])

        

        # create a chain with the llm and the prompt template
        self.chain = chat_prompt_template | self.llm | StrOutputParser()

    def cancel(self) -> None:
        self.llm.cancel()

    def send_prompt(self, prompt: str) -> Tuple[dict | str]:

        prompt = prompt + " "
        prompt = re.sub(r'\b(?:tell|say)\s+me\b', lambda match: match.group(0).replace("me", "at the instruction point"), prompt, flags=re.IGNORECASE)
        prompt = prompt.replace("to to", "to").replace("them", "him").strip()
        
        today_dt = dt.datetime.now()
        day_suffix = lambda day: 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        day = today_dt.strftime(f"%A {today_dt.day}{day_suffix(today_dt.day)}")
        
        time_h = today_dt.strftime("%H:%M")
        tomorrow = (today_dt + dt.timedelta(days=1)).strftime("%A")

        response = self.chain.invoke({
            "prompt": prompt,
            "actions_descriptions": self.actions_descriptions[:-1],
            "day": day,
            "tomorrow": tomorrow,
            "time_h": time_h
        })

        return json.loads(response), prompt

    def load_waypoints(self) -> None:
        self.waypoints = ""
        waypoints = json.load(open(self.waypoints_path))

        for ele in waypoints:
            self.waypoints += f"- {ele['room']}\n"
            for l in ele["locations"]:
                self.waypoints += f"- {l}\n"

    def create_grammar(self) -> None:
        self.actions_descriptions = ""
        actions_refs = []

        # Map placeholder strings to $defs names
        placeholder_to_def = {
            "<items>": "items",
            "<categories>": "categories",
            "<waypoints>": "waypoints",
            "<names>": "names"
        }

        # Build $defs from data files
        schema_defs = {}
        for placeholder, def_name in placeholder_to_def.items():
            schema_defs[def_name] = {
                "type": "string",
                "enum": self.placeholder_values[placeholder]
            }

        # Collect inline choice arrays and deduplicate by value
        inline_defs = {}  # tuple(choices) -> def_name
        def_names_used = set(schema_defs.keys())

        for robot_act in self.robot_actions:
            for arg_name, arg_def in robot_act['args'].items():
                if 'choices' in arg_def and isinstance(arg_def['choices'], list):
                    key = tuple(arg_def['choices'])
                    if key not in inline_defs:
                        name = arg_name
                        if name in def_names_used:
                            name = f"{robot_act['name']}_{arg_name}"
                        inline_defs[key] = name
                        def_names_used.add(name)

        for choices, def_name in inline_defs.items():
            schema_defs[def_name] = {
                "type": "string",
                "enum": list(choices)
            }

        # Helper: resolve choices to a $ref
        def choices_to_ref(choices):
            if isinstance(choices, str) and choices in placeholder_to_def:
                return {"$ref": f"#/$defs/{placeholder_to_def[choices]}"}
            elif isinstance(choices, list):
                key = tuple(choices)
                if key in inline_defs:
                    return {"$ref": f"#/$defs/{inline_defs[key]}"}
            return {"type": "string", "enum": choices}

        # Build action definitions using $ref
        action_definitions = {}
        for robot_act in self.robot_actions:
            self.actions_descriptions += f"- {robot_act['name']}: {robot_act['description']}\n"
            actions_refs.append({"$ref": f"#/$defs/{robot_act['name']}"})

            action_args = {
                "type": "object",
                "properties": {},
                "required": []
            }

            if robot_act['arg_case'] == 'allOf' and len(robot_act['args'].keys()) != 0:
                for arg in robot_act["args"]:
                    if "choices" in robot_act["args"][arg]:
                        action_args["properties"][arg] = choices_to_ref(robot_act["args"][arg]["choices"])
                    else:
                        prop = {"type": robot_act["args"][arg]["type"]}
                        for attr in ("minLength", "maxLength"):
                            if attr in robot_act["args"][arg]:
                                prop[attr] = robot_act["args"][arg][attr]
                        action_args["properties"][arg] = prop
                    action_args['required'].append(arg)
                if 'result_key_required' in robot_act and robot_act['result_key_required']:
                    action_args['properties']['result_key'] = {"type": "string"}
                    action_args['required'].append('result_key')

            elif robot_act['name'] == 'find_object':
                action_args['oneOf'] = []

                item_list = {'item': ['category', 'specific_item'], 'specific_item': ['specific_item'], 'category': ['category'], 'none': []}
                size_list = {'size': ['size'], 'weight': ['weight'], 'none': []}

                args_ref = {}
                for arg in robot_act["args"]:
                    if "choices" in robot_act["args"][arg]:
                        args_ref[arg] = choices_to_ref(robot_act["args"][arg]["choices"])

                for item, size in product(list(item_list.keys()), size_list):
                    if item == 'none' and size == 'none':
                        continue
                    props = {k: args_ref[k] for k in [*item_list[item], *size_list[size]]}
                    action_to_add = {
                        "properties": props,
                        "required": list(item_list[item]) + list(size_list[size])
                    }
                    if 'result_key_required' in robot_act and robot_act['result_key_required']:
                        action_to_add['properties']['result_key'] = {"type": "string"}
                        action_to_add['required'].insert(0, 'result_key')
                    action_args['oneOf'].append(action_to_add)

            elif robot_act['arg_case'] == 'anyOf':
                for arg in robot_act["args"]:
                    if "choices" in robot_act["args"][arg]:
                        action_args['properties'][arg] = choices_to_ref(robot_act["args"][arg]["choices"])
                    else:
                        action_args['properties'][arg] = {"type": robot_act["args"][arg]["type"]}

                if 'result_key_required' in robot_act and robot_act['result_key_required']:
                    action_args['properties']['result_key'] = {"type": "string"}
                    action_args['required'].append('result_key')

            elif robot_act['arg_case'] == 'oneOf':
                action_args['oneOf'] = []
                del action_args['properties']
                del action_args['required']

                for search_by_option in robot_act['args']['search_by']['choices']:
                    option_obj = {
                        "properties": {
                            "search_by": {
                                "const": search_by_option
                            }
                        },
                        "required": ['search_by']
                    }

                    if search_by_option != 'none':
                        option_obj['properties'][search_by_option] = choices_to_ref(
                            robot_act['args'][search_by_option]['choices']
                        )

                    # idk if we should let this here
                    if 'previously_found' in robot_act['args']:
                        # option_obj["required"].append('previously_found')
                        option_obj["properties"]['previously_found'] = {
                            'type': 'boolean'
                        }
                        option_obj['required'].append(search_by_option)

                    if 'result_key_required' in robot_act and robot_act['result_key_required']:
                        option_obj['properties']['result_key'] = {"type": "string"}
                        option_obj['required'].append('result_key')

                    action_args['oneOf'].append(option_obj)

            action_def = {
                "type": "object",
                "properties": {
                    "explanation_of_next_actions": {
                        "type": "string",
                        "maxLength": 200
                    },
                    robot_act['name']: action_args
                },
                "required": ['explanation_of_next_actions', robot_act['name']]
            }

            action_definitions[robot_act["name"]] = action_def

        self.grammar_schema = json.dumps({
            "$defs": {**schema_defs, **action_definitions},
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "anyOf": actions_refs,
                    },
                    "minItems": 1,
                    "maxItems": 15
                },
            },
            "required": ["actions"]
        })
