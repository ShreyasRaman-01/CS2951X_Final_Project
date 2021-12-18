'''
This is a script that loads an ATARI game of a specified choice onto a GUI + records game play
to generate a dataset for WSDDN object detection model

Also generates a text file capturing the coordinates of relevant environment positions
'''
import gym
import os


#specify game type to render e.g. mspacman-v0 or breakout-v0 or spaceinvaders-v0
game = str(input("Enter game: "))

#create required gym environment
env = gym.make('MsPacman-v0')

#print size of actions space
print(env.action_space)
print(env.observation_space)

image_save_dir = os.path.join('.',game)

#sample a total of 10 games over a 1000 steps each using user inputs (keyboard)
for episode in range(10):

    #reset environment for each epoch
    observation = env.reset()

    for step in range(1000):

        a = int(input("Action: "))

        if a >= len(env.action_space[0]):
            print("invalid action")
            continue


        #first render environment and save the rendering
        env.render()




        #take the action chosen
        action = env.action_space[a]
        env.step(action)
