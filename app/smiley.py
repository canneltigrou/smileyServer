#%%
import os

from flask import redirect, render_template, request, send_file, session, url_for



#from matplotlib.patches import Ellipse
#from scipy.interpolate import make_interp_spline
#import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt


print("Hello world")

class Smiley:
  ratio_X = 1
  ratio_Y = 1
  rotation = 0

  # we suppose here the axis origin id at the bottom left corner.
  # ellipse coordinate :

  left_eye_center = (-0.3,0.2)     # center of the left eye
  left_eye_W = 0.2                # width of the left eye
  left_eye_H = 0.05               # height of the left eye
  left_eye_rot = 0                # rotation of the left eye

  rigth_eye_center = (0.3,0.2)    # center of the right eye
  right_eye_W = 0.2               # width of the right eye
  right_eye_H = 0.05              # height of the right eye
  right_eye_rot = 0               # rotation of the right eye

  # for now, lets have an ellipse for the mouth
  mouth_center = (0,-0.2)    # center of the mouth
  mouth_W = 0.02              # width of the mouth
  mouth_H = 0.1               # height of the mouth
  mouth_rot = 0               # rotation of the mouth

  # we can have a smouth line for the eyebrows
  # so we need to keep the vector of points
  left_eyebrow = [(-0.4, 0.3), (-0.35, 0.4), (-0.3, 0.41), (-0.25, 0.4), (-0.2, 0.3)]
  right_eyebrow = [(0.2, 0.3), (0.25, 0.4), (0.30, 0.41), (0.35, 0.4), (0.4, 0.3)]

  def draw_smiley(self, ax):
    #plt.figure(figsize=(10, 10))
    #ax = plt.gca()

    # first of all, the base transformation of the data points is needed
    base = ax.transData #plt.gca().transData
    rot = mpl.transforms.Affine2D().rotate_deg(self.rotation)
    t = mpl.transforms.Affine2D().translate(-0.5,-0.5)

    # let's create the background face (big ellipse of 1,1, centered, yellow.)
    background_face = Ellipse(xy=(0, 0), width=1, height=1, 
                            edgecolor='y', fc='y')
    ax.add_patch(background_face)

    ######################  EYES    #########################################
    right_eye_ellipse = Ellipse(xy=self.rigth_eye_center, width=self.right_eye_W * self.ratio_X, 
                              height=self.right_eye_H * self.ratio_Y, edgecolor='k', fc='w')
    right_eye_ellipse.set_transform(rot + base)
    ax.add_patch(right_eye_ellipse)

    left_eye_ellipse = Ellipse(xy=self.left_eye_center, width=self.left_eye_W * self.ratio_X, 
                              height=self.left_eye_H * self.ratio_Y, edgecolor='k', fc='w')
    left_eye_ellipse.set_transform(rot + base)
    ax.add_patch(left_eye_ellipse)

    ###################### EYEBROWS   #####################################
    # left:
    x = []
    y = []
    for i in range(len(self.left_eyebrow)):
      x.append(self.left_eyebrow[i][0])
      y.append(self.left_eyebrow[i][1])
    x_new = np.linspace(min(x), max(x), 40)
    a_BSpline = make_interp_spline(x,y)
    y_new = a_BSpline(x_new)
    ax.plot(x_new, y_new, c='k', linewidth=2, transform= rot + base)
    # right:
    x = []
    y = []
    for i in range(len(self.right_eyebrow)):
      x.append(self.right_eyebrow[i][0])
      y.append(self.right_eyebrow[i][1])
    x_new = np.linspace(min(x), max(x), 40)
    a_BSpline = make_interp_spline(x,y)
    y_new = a_BSpline(x_new)
    ax.plot(x_new, y_new, c = 'k', linewidth=2, transform= rot + base)

    ###################   MOUTH   ###########################################
    mouth_ellipse = Ellipse(xy=self.mouth_center, width=self.mouth_W * self.ratio_X, 
                              height=self.mouth_H * self.ratio_Y, edgecolor='r', fc='k')
    mouth_ellipse.set_transform(rot + base)
    ax.add_patch(mouth_ellipse)

    #ax.set_transform(rot + base)
    #ax.rotate_around(5, 5, 20)
    plt.savefig("smiley.jpg", format = 'jpg')
    return ax

def displayDefaultSmiley():
  return 42
  # print("bouh")
  # smileyTest = Smiley()
  # ax = plt.gca()
  # smileyTest.draw_smiley(ax)
  # return send_file(ax)


smileyTest = Smiley()
ax = plt.gca()
smileyTest.draw_smiley(ax)


