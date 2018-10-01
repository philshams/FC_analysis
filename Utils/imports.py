# Plotting
import matplotlib.pyplot as plt  # used for debugging
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

# Math
import numpy as np
from scipy import misc
import pandas as pd
from scipy.spatial import distance
from math import factorial, atan2, degrees

# Various stuff
from collections import namedtuple

# Image and video processing
import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.fx import crop

# SYS
import os
from termcolor import colored
import sys
import time
import warnings
import gc
from tqdm import tqdm
import datetime
from warnings import warn

# UI
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# parallel processing
from multiprocessing.dummy import Pool as ThreadPool


# Utils
from Utils.loadsave_funcs import load_yaml
from Utils.maths import calc_acceleration, calc_ang_velocity, calc_ang_acc, calc_distance_2d
from Utils.decorators import clock
from Utils.utils_classes import Session_metadata, DataBase, Exploration
from Utils.loadsave_funcs import save_data
from Utils.Messaging import slack_chat_messenger
from Utils.utils_classes import Trial, Cohort

# Processing
from Processing.Processing_maze import mazeprocessor,  mazecohortprocessor  # , ProcessingTrialsMaze, ProcessingSessionMaze
from Processing.Processing_utils import *
from multiprocessing.dummy import Pool as ThreadPool

# file IO
from tempfile import mkdtemp
from nptdms import TdmsFile
import psutil
import pyexcel
import yaml


# Config
from Config import save_name
from Config import cohort_options










