import os
import cv2
import dlib

from flask import redirect, render_template, request, send_file, session, url_for

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import make_interp_spline
import numpy as np


detector = dlib.get_frontal_face_detector()
# the index is here begin at 0 and not 1 like in the image. we have to decrease by 1 the index
left_eye_index = (36, 37, 38, 39, 40, 41)
right_eye_index = (42, 43, 44, 45, 46, 47)
left_eyebrow_index = (17, 18, 19, 20, 21)
right_eyebrow_index = (22, 23, 24, 25, 26)
intern_lips_index = (60, 61, 62, 63, 64, 65, 66, 67)
extern_lips_index = (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)

predictor = dlib.shape_predictor("app/static/shape_predictor_68_face_landmarks.dat")


def getShape(image):
  dets = detector(image, 1)
  # take the first face
  d = dets[0]
  # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # face_image = image[d.top():d.bottom(), d.left():d.right()]
  shape = predictor(image, d)
  return shape


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
    ax.axis('off')
    plt.savefig("app/static/smiley.jpg", format = 'jpg')
    return ax


def displayDefaultSmiley():
  smileyTest = Smiley()
  ax = plt.gca()
  smileyTest.draw_smiley(ax)
  # return send_file(ax)


# smileyTest = Smiley()
# ax = plt.gca()
# smileyTest.draw_smiley(ax)

def drawImageWithLandmarks(image, shape):
  plt.imshow(image,cmap='gray')
  for i in range(68):
    point = (shape.part(i).x, shape.part(i).y)
    plt.scatter(point[0], point[1], s=5, c='red', marker='o')
  plt.show()


def rotate_image(image):
  OUTER_EYES_AND_NOSE = [36, 45, 33]; #{left eye, right eye, nose}
  # we would like to have the Y coordinate of the 2 eyes with the same value
  shape = getShape(image)
  right_eye = (shape.part(OUTER_EYES_AND_NOSE[0]).x, shape.part(OUTER_EYES_AND_NOSE[0]).y)
  left_eye = (shape.part(OUTER_EYES_AND_NOSE[1]).x, shape.part(OUTER_EYES_AND_NOSE[1]).y)
  # compute the angle between the eye centroids
  dY = right_eye[1] - left_eye[1]
  dX = right_eye[0] - left_eye[0]
  angle = np.degrees(np.arctan2(dY, dX)) - 180
  # compute center (x, y)-coordinates (i.e., the median point)
  # between the two eyes in the input image
  eyesCenter = ((left_eye[0] + right_eye[0]) // 2,
  (left_eye[1] + right_eye[1]) // 2)
  scale = 1
  # grab the rotation matrix for rotating and scaling the face
  M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  (w,h) = gray_image.shape #(desiredFaceWidth, desiredFaceHeight)
  rotated_image = cv2.warpAffine(gray_image, M, (w, h),flags=cv2.INTER_CUBIC)
  return rotated_image, angle

def updateSmiley(smiley, image):

  ######################################################################
  # We will compute the different size of the face attribute 
  # relatively to the size of the image we are working with.
  # Then we will compute the ratio to make them at a relative size for the smiley.
  #######################################################################

  rotated_image,angle = rotate_image(image)
  rotated_shape = getShape(rotated_image)

  # let's compute the ratio.
  # about x : let's take the points 1 and 17 from the landmarks, 
  # and compare to the figure size we have set : 1
  face_width = (rotated_shape.part(16).x - rotated_shape.part(0).x)
  smiley.ratio_X = 1 / face_width

  # about y : let's take the points 37 and 9 from the landmarks, and multiply by 3/2 
  # and compare to the figure size we have set : 10
  face_height = 1.5 * (rotated_shape.part(8).y - rotated_shape.part(right_eye_index[0]).y)
  smiley.ratio_Y = 1 / face_height

  # the rotation is the inverse of the rotation done 
  smiley.rotation = -angle

  #######################################################################
  ###################       EYES     ####################################
  #######################################################################
  # to do so, we will compute the height and the width of the eye
  # using the shape variable
  smiley.right_eye_H = np.mean([rotated_shape.part(right_eye_index[1]).y, rotated_shape.part(right_eye_index[2]).y])\
                  - np.mean([rotated_shape.part(right_eye_index[4]).y, rotated_shape.part(right_eye_index[5]).y])
  smiley.right_eye_W = rotated_shape.part(right_eye_index[3]).x - rotated_shape.part(right_eye_index[0]).x
  # lets compute the position. 
  # I set the y coordinate of the eyes to be always at 2/3 of the face
  # I suppose the coordinate of the shape.part[0] is at -0.5 
  y_pos = 2/3 - 0.5
  x_pos = abs(np.mean([rotated_shape.part(right_eye_index[3]).x, rotated_shape.part(right_eye_index[0]).x]) - rotated_shape.part(0).x)\
              / face_width - 0.5
  smiley.right_eye_center = (x_pos, y_pos) 


  smiley.left_eye_H = np.mean([rotated_shape.part(left_eye_index[1]).y, rotated_shape.part(left_eye_index[2]).y])\
                  - np.mean([rotated_shape.part(left_eye_index[4]).y, rotated_shape.part(left_eye_index[5]).y])
  smiley.left_eye_W = rotated_shape.part(left_eye_index[3]).x - rotated_shape.part(left_eye_index[0]).x
  # lets compute the position.
  y_pos = 2/3 - 0.5
  x_pos = abs(np.mean([rotated_shape.part(left_eye_index[3]).x, rotated_shape.part(left_eye_index[0]).x]) - rotated_shape.part(0).x)\
              / face_width - 0.5
  smiley.left_eye_center = (x_pos, y_pos)


  #######################################################################
  ###################   EYEBROWS     ####################################
  #######################################################################
  # left eyebrow:
  pos_vect = smiley.left_eyebrow
  for i in range(5):
    x_pos = abs(rotated_shape.part(left_eyebrow_index[i]).x - rotated_shape.part(0).x)\
              / face_width - 0.5
    y_pos = abs(rotated_shape.part(left_eyebrow_index[i]).y - rotated_shape.part(8).y)\
              / face_height - 0.5
    pos_vect[i] = (x_pos, y_pos)
  #smiley.left_eyebrow = pos_vect

  # right eyebrow:
  pos_vect2 = smiley.right_eyebrow
  for i in range(5):
    x_pos = abs(rotated_shape.part(right_eyebrow_index[i]).x - rotated_shape.part(0).x)\
              / face_width - 0.5
    y_pos = abs(rotated_shape.part(right_eyebrow_index[i]).y - rotated_shape.part(8).y)\
              / face_height - 0.5
    pos_vect2[i] = (x_pos, y_pos)
  #smiley.right_eyebrow = pos_vect2


  #######################################################################
  ###################     MOUTH      ####################################
  #######################################################################
  mouth_index = intern_lips_index
  # to do so, we will compute the height and the width of the eye
  # using the shape variable
  smiley.mouth_H = abs(rotated_shape.part(mouth_index[2]).y -rotated_shape.part(mouth_index[6]).y)
  smiley.mouth_W = abs(rotated_shape.part(mouth_index[0]).x - rotated_shape.part(mouth_index[4]).x)
  # lets compute the position. 
  y_pos = abs(rotated_shape.part(mouth_index[6]).y - rotated_shape.part(8).y) \
              / face_height - 0.5
  x_pos = abs(rotated_shape.part(mouth_index[2]).x - rotated_shape.part(0).x)\
              / face_width - 0.5
  smiley.mouth_center = (x_pos, y_pos)
