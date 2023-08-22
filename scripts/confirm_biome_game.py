#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:55:43 2023

@author: dgaio
"""

# conda create -n pygame_env python=3.10.8
# conda activate pygame_env
# conda install -c conda-forge pygame python=3.10.8
# conda install pandas
# conda install numpy


# =============================================================================
# import pygame
# import pandas as pd
# import random
# 
# # Initialize pygame
# pygame.init()
# 
# # Constants for display
# SCREEN_WIDTH = 1000
# SCREEN_HEIGHT = 800
# FONT_SIZE = 20
# LARGE_FONT_SIZE = 28
# FONT_NAME = pygame.font.match_font('arial')
# 
# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# 
# # Load the CSV
# df = pd.read_csv('/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_biome_pmid_title_abstract.csv')
# 
# # Check if confirmed_samples.csv exists, load or initialize it
# try:
#     confirmed_df = pd.read_csv('confirmed_samples.csv')
# except FileNotFoundError:
#     confirmed_df = pd.DataFrame(columns=['sample', 'biome', 'pmid', 'title', 'abstract'])
# 
# df = df[~df['sample'].isin(confirmed_df['sample'])]
# biome_counts = confirmed_df['biome'].value_counts().to_dict()
# 
# def get_random_sample(df, biome):
#     return df[df['biome'] == biome].sample(1)
# 
# def draw_text(surf, text, size, x, y):
#     font = pygame.font.Font(FONT_NAME, size)
#     text_surface = font.render(text, True, BLACK)
#     text_rect = text_surface.get_rect()
#     text_rect.midtop = (x, y)
#     surf.blit(text_surface, text_rect)
# 
# # Main loop for the game
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption('Biome Classifier Game')
# 
# running = True
# while running:
#     min_biome = min(biome_counts, key=lambda k: biome_counts.get(k, 0))
#     if min_biome not in df['biome'].values:
#         # Exit if no more samples left for the minimum biome
#         running = False
# 
#     sample = get_random_sample(df, min_biome)
#     prompt = f"Title: {sample.iloc[0]['title']} | Abstract: {sample.iloc[0]['abstract']}"
#     answer = ''
# 
#     while not answer:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 answer = 'exit'
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_y:
#                     answer = 'y'
#                 elif event.key == pygame.K_n:
#                     answer = 'n'
#                 elif event.key in [pygame.K_a, pygame.K_p, pygame.K_s, pygame.K_w, pygame.K_u, pygame.K_t]:
#                     answer = event.unicode
# 
#         screen.fill(WHITE)
#         draw_text(screen, "Do you confirm the biome is correct? (y/n) or assign a new biome (a/p/s/w/u/t)", LARGE_FONT_SIZE, SCREEN_WIDTH / 2, 50)
#         draw_text(screen, prompt, FONT_SIZE, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 4)
#         pygame.display.flip()
# 
#     # Update DataFrames
#     if answer in ['y', 'a', 'p', 's', 'w', 'u', 't']:
#         if answer == 'y':
#             confirmed_df = confirmed_df.append(sample, ignore_index=True)
#         else:
#             sample['biome'] = answer
#             confirmed_df = confirmed_df.append(sample, ignore_index=True)
#         biome_counts = confirmed_df['biome'].value_counts().to_dict()
#         df = df[~df['sample'].isin(confirmed_df['sample'])]
# 
# # Save the final results
# confirmed_df.to_csv('confirmed_samples.csv', index=False)
# 
# pygame.quit()
# =============================================================================











# =============================================================================
# FILENAME = '/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_biome_pmid_title_abstract.csv'
# OUTPUT_FILENAME = 'updated_' + FILENAME
# SUBSET_FILENAME = 'random_subset_' + FILENAME
# 
# def get_bin_counts(df):
#     return df['biome'].value_counts()
# 
# def create_unique_pmid_subset(df):
#     # Drop rows with NaN pmids, then drop duplicates keeping only the first occurrence
#     subset = df.dropna(subset=['pmid']).drop_duplicates(subset=['pmid'], keep='first')
# 
#     print("Unique PMIDs per biome after removing duplicates:")
#     print(subset['biome'].value_counts())
# 
#     # Maintain an approximately equal number of each biome
#     min_biomes = subset['biome'].value_counts().min()
#     print(f"Minimum biomes: {min_biomes}")
# 
#     samples_from_each_biome = [subset[subset['biome'] == biome].sample(min_biomes) for biome in subset['biome'].unique()]
#     
#     return pd.concat(samples_from_each_biome)
# 
# def review_sample(sample):
#     print("\nSample ID:", sample.iloc[0]['sample'])
#     print("Title:", sample.iloc[0]['title'])
#     print("Abstract:", sample.iloc[0]['abstract'])
#     is_correct = input(f"Is the biome '{sample.iloc[0]['biome']}' correct? (y/n/q) ")
# 
#     if is_correct == 'q':
#         return None, False
# 
#     if is_correct == 'n':
#         new_biome = input("Enter the new biome (a for animal, p for plant, s for soil, w for water, u for unknown, t for trash): ")
#         sample['biome'] = new_biome
#     sample['checked'] = True
# 
#     return sample, True
# 
# def propose_sample(df):
#     counts = get_bin_counts(df)
#     reviewed_pmids = df[df['checked']]['pmid'].unique()
# 
#     target_biomes = ['a', 's', 'w', 'p', 'u']
#     min_count = counts.min() if not counts.empty else 0
# 
#     potential_samples = df[
#         ~df['checked'] & 
#         ~df['title'].isna() & 
#         ~df['abstract'].isna() & 
#         ~df['pmid'].isin(reviewed_pmids)
#     ]
# 
#     for biome in target_biomes:
#         if counts.get(biome, 0) == min_count and len(potential_samples[potential_samples['biome'] == biome]) > 0:
#             return potential_samples[potential_samples['biome'] == biome].sample(1)
# 
#     return potential_samples.sample(1)
# 
# 
# def game(df, game_subset):
#     print("\nWelcome to the biome review game!")
#     
#     while True:
#         counts = get_bin_counts(game_subset)
#         print("\nCurrent counts:", counts.to_dict())
# 
#         sample = propose_sample(game_subset)
#         sample, continue_game = review_sample(sample)
# 
#         if not continue_game:
#             break
# 
#         # Corrected this section
#         game_subset.loc[sample.index[0], 'biome'] = sample['biome'].iloc[0]
#         game_subset.loc[sample.index[0], 'checked'] = sample['checked'].iloc[0]
# 
#     # Update the original dataframe based on reviewed pmids from the subset
#     for index, row in game_subset[game_subset['checked']].iterrows():
#         df.loc[df['pmid'] == row['pmid'], 'biome'] = row['biome']
#         df.loc[df['pmid'] == row['pmid'], 'checked'] = True
# 
#     df.to_csv(OUTPUT_FILENAME)
#     game_subset.to_csv(SUBSET_FILENAME)
# 
# 
# 
# if __name__ == "__main__":
#     df = pd.read_csv(FILENAME)
#     if 'checked' not in df.columns:
#         df['checked'] = False
# 
#     try:
#         game_subset = pd.read_csv(SUBSET_FILENAME)
#     except FileNotFoundError:
#         game_subset = create_unique_pmid_subset(df)
#         game_subset['checked'] = False
# 
#     game(df, game_subset)
# =============================================================================


import pygame
import pandas as pd
import sys


# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (255, 255, 255)
FONT_COLOR = (0, 0, 0)
FONT_SIZE = 24
TEXT_COLOR = (1, 0, 0)

# Set up screen and fonts
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Biome Review Game")
font = pygame.font.SysFont(None, FONT_SIZE)

class Button:
    def __init__(self, x, y, w, h, text, action=None, color=(100, 100, 100)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action  # action is a function that will be called when the button is clicked
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)  # use the button color
        rendered_text = font.render(self.text, True, FONT_COLOR)
        text_rect = rendered_text.get_rect(center=self.rect.center)
        screen.blit(rendered_text, text_rect)

    def handle_click(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the click position is inside the button's rectangle
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()
                    

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        if font.size(current_line + ' ' + word)[0] <= max_width:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    
    return lines

def draw_text(screen, text, font, color, x, y, max_width):
    lines = wrap_text(text, font, max_width)
    for line in lines:
        rendered_text = font.render(line, True, color)
        screen.blit(rendered_text, (x, y))
        y += font.get_height()  # Adjust y for the next line

def set_biome(self, biome):
    self.df.loc[self.current_sample.name, 'biome'] = biome
    self.df.loc[self.current_sample.name, 'checked'] = True
    self.show_biome_buttons = False
    self.move_to_next_sample()
    
def handle_biome(self, chosen_biome):
    # Save the chosen biome to the current sample (you might adjust this part depending on your data structure)
    self.df.loc[self.current_sample.name, 'biome'] = chosen_biome
    
    # Mark it as checked and move to the next sample
    self.df.loc[self.current_sample.name, 'checked'] = True
    self.move_to_next_sample()

    # Set the flag back to False so the regular buttons are shown for the next sample
    self.show_biome_buttons = False



class Game:
    def __init__(self, filename, font):
        self.df = pd.read_csv(filename)
        if 'checked' not in self.df.columns:
            self.df['checked'] = False
        self.current_sample_idx = 0
        self.current_sample = self.get_next_sample()
        self.setup_buttons()
        self.font = font
        self.show_biome_buttons = False  # set this flag to control showing the biome buttons
        
        
    def get_next_sample(self):
        # Filter unchecked samples that don't have NaN in the title
        valid_samples = self.df[(~self.df['checked']) & (pd.notna(self.df['title']))]
        if self.current_sample_idx < len(valid_samples):
            return valid_samples.iloc[self.current_sample_idx]
        else:
            return None  # No more valid samples left

    def yes_action(self):
        # For demonstration, just marking it as checked
        self.df.loc[self.current_sample.name, 'checked'] = True
        self.move_to_next_sample()


    def no_action(self):
        # You can modify biome or other actions here.
        # For demonstration, just marking it as checked
        self.df.loc[self.current_sample.name, 'checked'] = True
        self.move_to_next_sample()
        self.show_biome_buttons = True # Display biome choices instead of immediately moving to the next sample
    
        
    def move_to_next_sample(self):
        self.current_sample_idx += 1
        self.current_sample = self.get_next_sample()
        if self.current_sample is None:
            global running
            running = False

    def setup_buttons(self):
        self.quit_button = Button(3 * SCREEN_WIDTH // 4 - 50, 500, 100, 50, "Quit", quit_action)
        
        self.regular_buttons = [
            Button(SCREEN_WIDTH // 4 - 50, 450, 100, 50, "Yes", self.yes_action),
            Button(SCREEN_WIDTH // 2 - 50, 450, 100, 50, "No", self.no_action)
        ]

        self.biome_buttons = [
            Button(SCREEN_WIDTH // 5 - 50, 500, 100, 50, "Animal", lambda: self.handle_biome("animal")),
            Button(2 * SCREEN_WIDTH // 5 - 50, 500, 100, 50, "Plant", lambda: self.handle_biome("plant")),
            Button(3 * SCREEN_WIDTH // 5 - 50, 500, 100, 50, "Water", lambda: self.handle_biome("water")),
            Button(4 * SCREEN_WIDTH // 5 - 50, 500, 100, 50, "Soil", lambda: self.handle_biome("soil")),
            Button(5 * SCREEN_WIDTH // 5 - 50, 500, 100, 50, "unknown", lambda: self.handle_biome("unknown")),
            Button(6 * SCREEN_WIDTH // 5 - 50, 500, 100, 50, "trash", lambda: self.handle_biome("trash"))
        ]


    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        if self.current_sample is not None:
            draw_text(screen, f"Sample ID: {self.current_sample['sample']}", font, TEXT_COLOR, 10, 50, SCREEN_WIDTH - 20)
            draw_text(screen, f"Biome: {self.current_sample['biome']}", font, TEXT_COLOR, 10, 100, SCREEN_WIDTH - 20)
            draw_text(screen, f"Title: {self.current_sample['title']}", font, TEXT_COLOR, 10, 150, SCREEN_WIDTH - 20)
            draw_text(screen, f"Abstract: {self.current_sample['abstract']}", font, TEXT_COLOR, 10, 200, SCREEN_WIDTH - 20)
        else:
            draw_text(screen, "No more valid samples left.", font, TEXT_COLOR, 10, 150, SCREEN_WIDTH - 20)
        
        # Show the regular buttons or biome buttons based on the flag
        button_list = self.biome_buttons if self.show_biome_buttons else self.regular_buttons
        for button in button_list:
            button.draw(screen)

        # Always draw the quit button
        self.quit_button.draw(screen)
            

    def handle_event(self, event):
        # Check event for all buttons
        button_list = self.biome_buttons if self.show_biome_buttons else self.regular_buttons
        
        for button in button_list:
            button.handle_click(event)

        # Also check for the quit button
        self.quit_button.handle_click(event)

def quit_action():
    global running
    running = False

def game_loop(game_instance):
    global running
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            game_instance.handle_event(event)

        game_instance.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()   # Forcefully exit the script


if __name__ == "__main__":
    game = Game("/Users/dgaio/cloudstor/Gaio/MicrobeAtlasProject/sample.info_biome_pmid_title_abstract.csv", font)
    game_loop(game)

# continue - implement latest gpt 



