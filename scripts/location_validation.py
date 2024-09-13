#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:26:33 2024

@author: dgaio
"""

import os
import json
import logging

class LocationValidationGame:
    def __init__(self, data, directory_with_split_metadata, work_dir):
        self.data = data
        self.directory_with_split_metadata = directory_with_split_metadata
        self.work_dir = work_dir
        self.save_file = os.path.join(work_dir, 'validation_game_progress.json')
        self.load_progress()  # Load existing progress at initialization

    def fetch_metadata_for_samples(self, sample_id):
        folder_name = f"dir_{sample_id[-3:]}"
        folder_path = os.path.join(self.directory_with_split_metadata, folder_name)
        metadata_file_path = os.path.join(folder_path, f"{sample_id}_clean.txt")
        try:
            with open(metadata_file_path, 'r') as file:
                return file.read()
        except Exception as e:
            logging.error(f"Failed to fetch metadata for sample {sample_id}: {e}")
            return f"Metadata for sample {sample_id} could not be retrieved."

    def play(self):
        for sample_id, info in self.data.items():
            # Skip samples that have already been validated
            if 'answer' in info and 'comment' in info and info['answer'] and info['comment']:
                continue
            
            metadata = self.fetch_metadata_for_samples(sample_id)
            print(metadata)
            print(f"GPT Location: {info['gpt_name']}")
            print(f"Coordinates from Metadata: {info['latlon_name']}")
            print("Who is right? (G = GPT, C = Coordinates, B = Both, N = Neither, QUIT to exit):")
            user_choice = input().strip().upper()

            if user_choice == "QUIT":
                print("Exiting the game and saving progress...")
                break
            
            while user_choice not in ['G', 'C', 'B', 'N']:
                print("Invalid input. Please choose G, C, B, or N, or type QUIT to exit.")
                user_choice = input().strip().upper()

            if user_choice == "QUIT":
                print("Exiting the game and saving progress...")
                break

            user_comment = input("Add a comment: ")
            if user_comment.strip() == "" or user_choice.strip() == "":
                continue  # Skip saving this entry if the input is incomplete
            
            self.data[sample_id]['answer'] = user_choice
            self.data[sample_id]['comment'] = user_comment
            self.save_progress()

    def save_progress(self):
        with open(self.save_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def load_progress(self):
        # Load progress if a save file exists
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f:
                self.data = json.load(f)

    def get_updated_data(self):
        return self.data





